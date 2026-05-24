"""
训练 FreqCNN — 从零训练的细粒度频谱编码器
不需要 BEATs 预训练权重, 不需要 HuggingFace
"""
import os, sys, argparse
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TTC'))
from models.freq_cnn_cont import FreqCNNContrastive
from data.audio_dataset import AudioDataModule, AudioThreeViews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data_changyin_audio/train')
    parser.add_argument('--val_dir', default='data_changyin_audio/val')
    parser.add_argument('--audio_length', type=int, default=64000, help='4s@16kHz=64000')
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ratio_sup_majority', type=float, default=0.0)
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='changyin_freqcnn')
    parser.add_argument('--log_dir', default='logs')
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    dm = AudioDataModule(
        train_dir=args.train_dir, val_dir=args.val_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        audio_length=args.audio_length,
        train_transform=AudioThreeViews(args.audio_length),
        val_transform=AudioThreeViews(args.audio_length))

    model = FreqCNNContrastive(
        base_ch=args.base_ch, output_dim=128,
        max_epochs=args.epochs, lr=args.lr,
        temperature=args.temperature, warmup_epochs=args.warmup_epochs,
        optimizer_name='adamw', batch_size=args.batch_size,
        ratio_supervised_majority=args.ratio_sup_majority,
        aux_loss_weight=args.aux_weight)

    ckpt = ModelCheckpoint(monitor='val.recall_rare', mode='max', save_top_k=3,
                           filename='best-{epoch:02d}-{val.recall_rare:.3f}')
    logger = CSVLogger(save_dir=args.log_dir, name=args.name)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices,
        logger=logger, callbacks=[ckpt, LearningRateMonitor()],
        log_every_n_steps=10, check_val_every_n_epoch=5)

    print(f"\n{'='*60}")
    print(f" FreqCNN — 从零训练细粒度频谱编码器")
    print(f" Mel: 256 bins (~31Hz/bin), 频率轴 256→128→64→32")
    print(f" 参数量: ~5M (BEATs的 1/18)")
    print(f"{'='*60}\n")
    trainer.fit(model, dm)
    print(f"\nBest: {ckpt.best_model_path}  recall_rare={ckpt.best_model_score:.4f}")


if __name__ == '__main__':
    main()
