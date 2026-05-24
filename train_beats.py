"""
BEATs + SupMin 对比学习 — 肠音二分类训练
集成 ABCD 四个改进:
  A: Attention Pooling (4 queries × cross-attn)
  B: 物理特征 concat 注入分类头
  C: BEATs 最后2层微调 (lr=1e-5)
  D: 物理特征预测辅助任务 (MSE, weight=0.1)
"""
import inspect
import os, sys, argparse
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

torch.set_float32_matmul_precision('high')

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'TTC'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'TTC', 'data'))

from models.beats_cont import BEATsContrastive
from audio_dataset import AudioDataModule, AudioThreeViews


def make_audio_transform(audio_length, training):
    sig = inspect.signature(AudioThreeViews)
    if 'training' in sig.parameters:
        return AudioThreeViews(audio_length, training=training)
    print("[WARN] AudioThreeViews has no 'training' parameter; using legacy random transform for all splits.")
    return AudioThreeViews(audio_length)


def main():
    parser = argparse.ArgumentParser()
    # --- 数据 ---
    parser.add_argument('--train_dir', type=str, default='data_changyin_audio/train')
    parser.add_argument('--val_dir', type=str, default='data_changyin_audio/val')
    parser.add_argument('--audio_length', type=int, default=48000)
    # --- BEATs ---
    parser.add_argument('--beats_code_path', type=str, default='./BEATs',
                       help='BEATs 源码目录 (含 BEATs.py, backbone.py)')
    parser.add_argument('--beats_ckpt_path', type=str, default=None,
                       help='BEATs 权重 .pt 路径 (默认在 code_path 下查找)')
    # 方案C: 微调
    parser.add_argument('--finetune_layers', type=int, default=2,
                       help='解冻 BEATs 最后 N 层: 0=冻结, 2=推荐, -1=全部')
    parser.add_argument('--beats_lr', type=float, default=1e-5,
                       help='BEATs 微调学习率 (应远小于主 LR)')
    # 方案A: Attention Pooling
    parser.add_argument('--attn_queries', type=int, default=4,
                       help='Attention pooling 的 query token 数量')
    parser.add_argument('--no_multilayer_fusion', action='store_true',
                       help='关闭 BEATs 多层特征融合 (只用最后一层)')
    parser.add_argument('--fusion_layers', type=str, default='1,4,8,12',
                       help='融合哪些层, 逗号分隔, 如 1,4,8,12')
    parser.add_argument('--fusion_expand', type=int, default=2,
                       help='融合后扩宽倍数: 1=768d, 2=1536d (推荐)')
    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--ratio_sup_majority', type=float, default=0.0,
                       help='0.0=SupMin, -1.0=SupCon')
    # 方案D: 辅助任务
    parser.add_argument('--aux_weight', type=float, default=0.1,
                       help='物理特征预测 MSE 权重')
    # 硬件
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # 输出
    parser.add_argument('--name', type=str, default='changyin_beats_abcd')
    parser.add_argument('--log_dir', type=str, default='logs')

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # --- 数据 ---
    dm = AudioDataModule(
        train_dir=args.train_dir, val_dir=args.val_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        audio_length=args.audio_length,
        train_transform=make_audio_transform(args.audio_length, training=True),
        val_transform=make_audio_transform(args.audio_length, training=False),
    )

    # --- 模型 (含所有四个改进) ---
    model = BEATsContrastive(
        beats_model_path=args.beats_code_path,
        beats_ckpt_path=args.beats_ckpt_path,
        beats_finetune_layers=args.finetune_layers,    # C
        multi_layer_fusion=not args.no_multilayer_fusion,
        fusion_layers=[int(x) for x in args.fusion_layers.split(',')],
        fusion_expand_ratio=args.fusion_expand,
        beats_lr=args.beats_lr,                         # C
        attn_pool_queries=args.attn_queries,            # A
        aux_loss_weight=args.aux_weight,                # D
        output_dim=128,
        max_epochs=args.epochs, lr=args.lr,
        temperature=args.temperature,
        warmup_epochs=args.warmup_epochs,
        optimizer_name='adamw',
        batch_size=args.batch_size,
        ratio_supervised_majority=args.ratio_sup_majority,
    )

    # --- 回调 ---
    checkpoint_cb = ModelCheckpoint(
        monitor='online_val_acc', mode='max', save_top_k=3,
        filename='best-{epoch:02d}-{online_val_acc:.3f}')
    lr_monitor = LearningRateMonitor()

    # --- 训练 ---
    logger = CSVLogger(save_dir=args.log_dir, name=args.name)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator,
        devices=args.devices, logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10, check_val_every_n_epoch=5,
    )

    print(f"\n{'='*60}")
    print(f" BEATs + ABCD 改进 — 肠音二分类")
    print(f" A: Attention Pooling ({args.attn_queries} queries)")
    print(f" A+: Multi-layer fusion (enabled={not args.no_multilayer_fusion}, layers={args.fusion_layers}, expand={args.fusion_expand}x)")
    print(f" B: 物理特征 concat (7-dim)")
    print(f" C: BEATs 微调 (最后 {args.finetune_layers} 层, lr={args.beats_lr})")
    print(f" D: 物理特征预测 MSE (weight={args.aux_weight})")
    print(f" Loss: SupMin (ratio={args.ratio_sup_majority})")
    print(f"{'='*60}\n")

    trainer.fit(model, dm)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best online_val_acc: {checkpoint_cb.best_model_score:.4f}")


if __name__ == '__main__':
    main()
