"""
只重训分类头 — encoder 冻结, 分类头从头学
用法:
  python finetune_head.py \
      --ckpt logs/changyin_beats_abcd_v1/best-*.ckpt \
      --train_dir data_changyin_audio/train \
      --val_dir data_changyin_audio/val \
      --beats_code_path ./BEATs \
      --epochs 30 --accelerator auto
"""
import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TTC'))
from models.beats_cont import BEATsContrastive

# 直接 import, 绕过 data/__init__.py (它拖了 cardiac.py 等不需要的依赖)
import importlib.util
_dataset_path = os.path.join(os.path.dirname(__file__), 'TTC', 'data', 'audio_dataset.py')
spec = importlib.util.spec_from_file_location("audio_dataset", _dataset_path)
audio_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_dataset)
AudioDataModule = audio_dataset.AudioDataModule
AudioThreeViews = audio_dataset.AudioThreeViews


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class HeadOnlyModule(pl.LightningModule):
    """只包装分类头, encoder 完全冻结"""

    def __init__(self, trained_model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.encoder = trained_model  # 完整 BEATsContrastive, 冻结

        # 冻结所有 encoder 参数
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 新建分类头 (替换原来的)
        pool_dim = self.encoder.hparams.get('attn_pool_queries', 4) * 768
        phys_dim = 7
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim + phys_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

        self.val_preds, self.val_labels = [], []

    def forward(self, x, phys_feats):
        # 直接用 encoder 的 forward, 但跳过它的 classifier
        pooled, _, _ = self.encoder(x, phys_feats)  # _encode_audio + forward
        phys_norm = (phys_feats - self.encoder.phys_mean.to(x.device)) / \
                    (self.encoder.phys_std.to(x.device) + 1e-8)
        combined = torch.cat([pooled, phys_norm], dim=1)
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        audios, phys_feats, labels = batch
        logits = self(audios[2], phys_feats)  # 只用弱视图
        loss = self.focal(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()

        # per-class
        for cid in [0, 1]:
            mask = labels == cid
            if mask.sum() > 0:
                rec = (logits[mask].argmax(1) == cid).float().mean()
                self.log(f'train.recall_{cid}', rec.item(), prog_bar=(cid == 1))

        self.log('train.loss', loss.item(), prog_bar=True)
        self.log('train.acc', acc.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audios, phys_feats, labels = batch
        logits = self(audios[2], phys_feats)
        self.val_preds.append(logits)
        self.val_labels.append(labels)

    def on_validation_epoch_end(self):
        if not self.val_preds: return
        preds = torch.cat(self.val_preds).argmax(1)
        labs = torch.cat(self.val_labels)
        acc = (preds == labs).float().mean()
        self.log('val.acc', acc.item(), prog_bar=True)

        for cid, cname in [(0, 'n'), (1, 'rare')]:
            mask = labs == cid
            if mask.sum() > 0:
                rec = (preds[mask] == cid).float().mean()
                self.log(f'val.recall_{cname}', rec.item(), prog_bar=(cname == 'rare'))

        self.val_preds.clear(); self.val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        return [opt], [sch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='data_changyin_audio/train')
    parser.add_argument('--val_dir', type=str, default='data_changyin_audio/val')
    parser.add_argument('--beats_code_path', type=str, default='./BEATs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # 加载训练好的模型
    print(f"Loading: {args.ckpt}")
    trained = BEATsContrastive.load_from_checkpoint(
        args.ckpt, beats_model_path=args.beats_code_path,
        strict=False, map_location='cpu')
    trained.eval()

    # 数据
    dm = AudioDataModule(
        train_dir=args.train_dir, val_dir=args.val_dir,
        batch_size=args.batch_size, num_workers=0, audio_length=48000,
        train_transform=AudioThreeViews(48000),
        val_transform=AudioThreeViews(48000),
    )

    # 分类头模型
    model = HeadOnlyModule(trained, lr=args.lr)

    ckpt_cb = ModelCheckpoint(monitor='val.recall_rare', mode='max',
                              save_top_k=1, filename='best-head-{val.recall_rare:.3f}')
    logger = CSVLogger(save_dir='logs', name='finetune_head')
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        logger=logger, callbacks=[ckpt_cb],
        log_every_n_steps=10, check_val_every_n_epoch=3,
    )

    trainer.fit(model, dm)
    print(f"\nBest: {ckpt_cb.best_model_path}  recall_rare={ckpt_cb.best_model_score:.4f}")


if __name__ == '__main__':
    main()
