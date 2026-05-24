"""
训练后评估: 对比训练前后的嵌入质量 + 分类上限
用法:
  python evaluate_model.py --ckpt logs/changyin_beats_abcd_v1/best-*.ckpt \
      --data_dir data_changyin_audio/val --beats_code_path ./BEATs
"""
import os, sys, glob, random, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SR = 16000
AUDIO_LEN = 48000
SEED = 42

# ==================== 数据加载 ====================
def load_wav(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
    if info is not None: data = data / (info.max + 1.0)
    if sr != SR: data = resample_poly(data, SR, sr)
    if len(data) < AUDIO_LEN: data = np.pad(data, (0, AUDIO_LEN - len(data)))
    else:
        start = (len(data) - AUDIO_LEN) // 2
        data = data[start:start + AUDIO_LEN]
    return data.astype(np.float32)


def compute_phys_feats_np(y, sr=SR, n_fft=1024, hop=256):
    from scipy.signal import get_window
    from scipy.fftpack import fft
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    win = get_window('hann', n_fft, fftbins=True)
    n_frames = 1 + max(0, len(y) - n_fft) // hop
    S = np.zeros((n_fft//2+1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        S[:, i] = fft(y[i*hop:i*hop+n_fft] * win)[:n_fft//2+1]
    power = np.abs(S)**2 / n_fft
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    total_en = np.sum(power, axis=0) + 1e-10
    centroid = np.sum(freqs[:,None]*power, axis=0) / total_en
    diff = freqs[:,None] - centroid[None,:]
    bandwidth = np.sqrt(np.sum(diff**2*power, axis=0) / total_en)
    lo_mask = (freqs>=200)&(freqs<=500)
    hi_mask = freqs>4000
    total_p = np.sum(power)
    return np.array([
        float(np.mean(centroid)), float(np.mean(bandwidth)),
        float(np.sum(power[lo_mask])/(total_p+1e-10)),
        float(np.sum(power[hi_mask])/(total_p+1e-10)),
        float(np.sqrt(np.mean(y**2))),
        float(np.max(np.abs(y))/(np.sqrt(np.mean(y**2))+1e-10)),
        float(np.sum(np.abs(np.diff(np.sign(y))))/(2*len(y))),
    ], dtype=np.float32)


# ==================== 模型加载 ====================
def load_model(ckpt_path, beats_code_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TTC'))
    from models.beats_cont import BEATsContrastive
    # 加载时提供 beats 路径
    model = BEATsContrastive.load_from_checkpoint(
        ckpt_path, beats_model_path=beats_code_path,
        strict=False, map_location='cpu')
    model.eval()
    return model


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='训练好的 .ckpt')
    parser.add_argument('--data_dir', type=str, required=True, help='val WAV 目录')
    parser.add_argument('--beats_code_path', type=str, default='./BEATs')
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs('eval_output', exist_ok=True)

    # --- 1. 加载模型 ---
    print(f"Loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.beats_code_path)
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    # --- 2. 收集数据 ---
    print(f"Loading data from {args.data_dir}")
    samples = []
    for cls_name in sorted(os.listdir(args.data_dir)):
        cls_dir = os.path.join(args.data_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        cls_id = int(cls_name.split('_')[-1])
        for wf in glob.glob(os.path.join(cls_dir, '*.wav')):
            samples.append((wf, cls_id))

    # 限制总数 + 确保每类都有
    class_samples = {0: [], 1: []}
    random.shuffle(samples)
    for s in samples:
        if len(class_samples[s[1]]) < args.max_samples // 2:
            class_samples[s[1]].append(s)

    selected = class_samples[0] + class_samples[1]
    random.shuffle(selected)
    print(f"  Selected: class_0={len(class_samples[0])}, class_1={len(class_samples[1])}")

    # --- 3. 提取嵌入 ---
    print("Extracting embeddings...")
    projections = []   # 128-dim 对比嵌入
    pooled = []        # 3072-dim pooled 特征
    logits_list = []   # 分类 logits
    labels = []
    phys_feats_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(selected), 32)):
            batch = selected[i:i+32]
            audios = [torch.from_numpy(load_wav(wf)).unsqueeze(0) for wf, _ in batch]
            audio_batch = torch.stack(audios, dim=0).to(device)  # [B, 1, T]

            # 物理特征
            phys_batch = torch.stack([
                torch.from_numpy(compute_phys_feats_np(load_wav(wf)))
                for wf, _ in batch
            ], dim=0).to(device)

            # 前向
            pool, proj, logits = model(audio_batch, phys_batch)

            projections.append(proj.cpu().numpy())
            pooled.append(pool.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            labels.extend([lbl for _, lbl in batch])
            phys_feats_list.append(phys_batch.cpu().numpy())

    projections = np.concatenate(projections, axis=0)
    pooled = np.concatenate(pooled, axis=0)
    logits_all = np.concatenate(logits_list, axis=0)
    labels = np.array(labels)
    phys_feats = np.concatenate(phys_feats_list, axis=0)

    print(f"  Projections: {projections.shape}, Pooled: {pooled.shape}")

    # --- 4. 分类准确率 ---
    preds = logits_all.argmax(1)
    acc = (preds == labels).mean()
    bal_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"模型分类结果 (直接前向, val集)")
    print(f"{'='*60}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Balanced Acc:    {bal_acc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    for i, name in enumerate(['class_0(n)', 'class_1(rare)']):
        rec = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        prec = cm[i,i] / cm[:,i].sum() if cm[:,i].sum() > 0 else 0
        print(f"  {name}: recall={rec:.4f}, precision={prec:.4f}")

    # --- 5. 嵌入质量: LR 分类上限 ---
    print(f"\n{'='*60}")
    print(f"嵌入分类上限 (Logistic Regression)")
    print(f"{'='*60}")

    # 分层拆分
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)

    for feat_name, feat_matrix in [('projection(128d)', projections), ('pooled(>768d)', pooled)]:
        try:
            train_idx, test_idx = next(splitter.split(feat_matrix, labels))
        except:
            continue
        X_tr, X_te = feat_matrix[train_idx], feat_matrix[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                random_state=args.seed)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        bal = balanced_accuracy_score(y_te, y_pred)
        cm_lr = confusion_matrix(y_te, y_pred)
        rec0 = cm_lr[0,0]/cm_lr[0].sum() if cm_lr[0].sum()>0 else 0
        rec1 = cm_lr[1,1]/cm_lr[1].sum() if cm_lr[1].sum()>0 else 0

        print(f"  {feat_name}:")
        print(f"    Balanced Acc: {bal:.4f}")
        print(f"    class_0 recall: {rec0:.4f}, class_1 recall: {rec1:.4f}")

    # 同时评估物理特征的分类上限做对比
    try:
        tr_i, te_i = next(splitter.split(phys_feats, labels))
        Xp_tr, Xp_te = phys_feats[tr_i], phys_feats[te_i]
        yp_tr, yp_te = labels[tr_i], labels[te_i]
        sp = StandardScaler()
        clfp = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=args.seed)
        clfp.fit(sp.fit_transform(Xp_tr), yp_tr)
        yp_pred = clfp.predict(sp.transform(Xp_te))
        bal_p = balanced_accuracy_score(yp_te, yp_pred)
        print(f"  phys_feats(7d) only:")
        print(f"    Balanced Acc: {bal_p:.4f}")
    except: pass

    # --- 6. t-SNE 可视化 ---
    print("\nGenerating t-SNE...")
    # 用 projection (128d) 做 t-SNE
    tsne_n = min(500, (labels==0).sum())
    tsne_idx = np.concatenate([
        np.random.choice(np.where(labels==0)[0], tsne_n, replace=False),
        np.where(labels==1)[0],
    ])
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30)
    emb_2d = tsne.fit_transform(StandardScaler().fit_transform(projections[tsne_idx]))
    tsne_labs = labels[tsne_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # t-SNE
    colors = {0: '#2196F3', 1: '#FF5722'}
    names = {0: f'n ({len(class_samples[0])})', 1: f'rare ({len(class_samples[1])})'}
    for cls_id in [0, 1]:
        mask = tsne_labs == cls_id
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=colors[cls_id], alpha=0.6, s=15 if cls_id==0 else 40,
                       label=names[cls_id], edgecolors='black' if cls_id==1 else 'none',
                       linewidth=0.3)
    axes[0].set_title(f't-SNE (训练后 projection 128d)', fontsize=12)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # 到 rare 中心的距离分布
    rare_center = np.mean(projections[labels==1], axis=0)
    for cls_id, cls_name in [(0, 'n'), (1, 'rare')]:
        mask = labels == cls_id
        dists = np.linalg.norm(projections[mask] - rare_center, axis=1)
        axes[1].hist(dists, bins=30, alpha=0.5, label=cls_name,
                    color=colors[cls_id], density=True)
    axes[1].set_xlabel('到 rare 类中心的欧氏距离')
    axes[1].set_title('嵌入空间中的距离分布', fontsize=12)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # 物理特征空间 (已知可分的维度)
    from sklearn.decomposition import PCA as PCA2
    pca_phys = PCA2(n_components=2)
    phys_2d = pca_phys.fit_transform(StandardScaler().fit_transform(phys_feats[tsne_idx]))
    for cls_id in [0, 1]:
        mask = tsne_labs == cls_id
        axes[2].scatter(phys_2d[mask, 0], phys_2d[mask, 1],
                       c=colors[cls_id], alpha=0.6, s=15 if cls_id==0 else 40,
                       label=names[cls_id], edgecolors='black' if cls_id==1 else 'none',
                       linewidth=0.3)
    axes[2].set_title(f'物理特征 PCA (7d, 作为对比基线)', fontsize=12)
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle(f'训练后模型评估 — Val Acc={acc:.3f} BalAcc={bal_acc:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('eval_output/01_trained_model_tsne.png', dpi=150)
    plt.close()
    print("  -> eval_output/01_trained_model_tsne.png")

    # --- 7. 汇总 JSON ---
    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
        "class_0_n": len(class_samples[0]),
        "class_1_rare": len(class_samples[1]),
    }
    with open('eval_output/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Charts in eval_output/\n")


if __name__ == '__main__':
    main()
