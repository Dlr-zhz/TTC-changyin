"""
BEATs 特征分离度验证
提取肠音事件的 embedding → PCA/t-SNE → LR 分类上限
支持两种模式:
  原始 BEATs:  python probe_beats.py --beats_code_path ./BEATs
  训练后模型:  python probe_beats.py --beats_code_path ./BEATs --model_ckpt logs/.../best.ckpt
"""
import os, sys, glob, random, argparse, json
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 常量 ====================
SR_TARGET = 16000
AUDIO_LENGTH = 48000
MAX_EVENTS_PER_CLASS = {"n": 2000, "g": 179, "h": 24}

# ==================== 物理特征 ====================
def compute_phys_feats_np(y, sr=SR_TARGET, n_fft=1024, hop=256):
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

# ==================== BEATs 加载 (原始) ====================
def load_beats(beats_code_path, beats_ckpt_path=None):
    """
    beats_code_path: BEATs 源码目录 (包含 BEATs.py, backbone.py 等)
    beats_ckpt_path: 预训练权重路径 (默认在 code_path 下查找)
    """
    print(f"Loading BEATs...")
    print(f"  Code path: {beats_code_path}")

    # 检查必要文件是否存在
    required_files = ['BEATs.py', 'backbone.py']
    for f in required_files:
        fpath = os.path.join(beats_code_path, f)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"缺少文件: {fpath}\n"
                f"请 clone BEATs 完整源码:\n"
                f"  git clone https://github.com/microsoft/unilm.git\n"
                f"  BEATs 源码在 unilm/beats/ 目录下"
            )

    # 添加源码路径
    if beats_code_path not in sys.path:
        sys.path.insert(0, beats_code_path)

    from BEATs import BEATs, BEATsConfig

    # 查找权重文件
    if beats_ckpt_path is None:
        ckpt_candidates = [
            os.path.join(beats_code_path, "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
            os.path.join(os.path.dirname(beats_code_path), "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
        ]
        for cand in ckpt_candidates:
            if os.path.exists(cand):
                beats_ckpt_path = cand
                break
        if beats_ckpt_path is None:
            raise FileNotFoundError(
                f"找不到 BEATs 预训练权重 (.pt 文件)\n"
                f"请下载: wget https://conversationhub.blob.core.windows.net/beats-share/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
            )

    print(f"  Checkpoint: {beats_ckpt_path}")
    ckpt = torch.load(beats_ckpt_path, map_location='cpu')
    cfg = BEATsConfig(ckpt['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    print("  -> BEATs loaded successfully")
    return model

# ==================== 音频加载 ====================
def load_wav(path):
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
    if info is not None: data = data / (info.max + 1.0)
    if sr != SR_TARGET:
        data = resample_poly(data, SR_TARGET, sr)
    return data


def pad_or_clip(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    else:
        # 居中截取
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]


def parse_txt(txt_path):
    events = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            try:
                st, et = float(parts[0]), float(parts[1])
                lab = parts[2] if len(parts)>=3 and parts[2] in ('n','g','h') else 'n'
                events.append((st, et, lab))
            except: continue
    return events

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='BEATs feature probe')
    parser.add_argument('--data_dir', type=str, default='data_changyin',
                       help='含有 .wav 和 .txt 标注文件的目录')
    parser.add_argument('--beats_code_path', type=str, required=True,
                       help='BEATs 源码目录 (含 BEATs.py, backbone.py 等)')
    parser.add_argument('--beats_ckpt_path', type=str, default=None,
                       help='BEATs 预训练权重 .pt 文件路径 (可选, 默认在 code_path 下查找)')
    parser.add_argument('--model_ckpt', type=str, default=None,
                       help='训练后 BEATsContrastive 的 .ckpt (embedding来源)')
    parser.add_argument('--head_ckpt', type=str, default=None,
                       help='HeadOnlyModule checkpoint (可选, 替换分类头)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = "beats_probe_trained" if (args.model_ckpt or args.head_ckpt) else "beats_probe"
    os.makedirs(out_dir, exist_ok=True)

    # --- 加载模型 ---
    trained_model = None
    needs_beats_ckpt = bool(args.model_ckpt or args.head_ckpt)
    if needs_beats_ckpt and args.beats_ckpt_path is None:
        # 探一下默认位置
        for guess in ['../model/BEATs_iter3.pt', './BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt']:
            if os.path.exists(guess):
                args.beats_ckpt_path = guess
                break

    if args.head_ckpt and not args.model_ckpt:
        # head_ckpt 是 HeadOnlyModule, 不能直接用 BEATsContrastive.load_from_checkpoint
        # 策略: 先创建原始 BEATsContrastive, 然后从 head_ckpt 加载 encoder + classifier 权重
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TTC'))
        from models.beats_cont import BEATsContrastive
        print("Creating base BEATsContrastive (weights will be overwritten)...")
        trained_model = BEATsContrastive(
            beats_model_path=args.beats_code_path,
            beats_ckpt_path=args.beats_ckpt_path)
        print(f"Loading head checkpoint weights from: {args.head_ckpt}")
        head_ckpt = torch.load(args.head_ckpt, map_location='cpu')
        head_state = head_ckpt['state_dict']
        # 'encoder.' 前缀 → BEATsContrastive 的 state_dict
        encoder_state = {}
        classifier_state = {}
        for k, v in head_state.items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v
            elif k.startswith('classifier.'):
                classifier_state[k.replace('classifier.', '')] = v
        if encoder_state:
            missing, unexpected = trained_model.load_state_dict(encoder_state, strict=False)
            print(f"  Encoder loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        if classifier_state:
            trained_model.classifier.load_state_dict(classifier_state, strict=False)
            print(f"  Classifier head replaced ({len(classifier_state)} params)")
        trained_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model = trained_model.to(device)

    elif args.model_ckpt:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TTC'))
        from models.beats_cont import BEATsContrastive
        print(f"Loading BEATsContrastive from: {args.model_ckpt}")
        trained_model = BEATsContrastive.load_from_checkpoint(
            args.model_ckpt,
            beats_model_path=args.beats_code_path,
            beats_ckpt_path=args.beats_ckpt_path,
            strict=False, map_location='cpu')
        trained_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model = trained_model.to(device)
        print(f"  -> Loaded, device={device}")

        # 如果有 head_ckpt, 额外替换分类头
        if args.head_ckpt:
            print(f"Replacing classifier from: {args.head_ckpt}")
            head_state = torch.load(args.head_ckpt, map_location='cpu')['state_dict']
            head_weights = {k.replace('classifier.', ''): v
                          for k, v in head_state.items() if k.startswith('classifier.')}
            if head_weights:
                trained_model.classifier.load_state_dict(head_weights, strict=False)
                print(f"  -> Replaced classifier ({len(head_weights)} params)")

    # --- 1. 收集所有事件 ---
    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if not wav_files:
        print(f"ERROR: 在 {args.data_dir} 下没有找到 .wav 文件")
        print(f"请检查 --data_dir 路径。当前目录: {os.getcwd()}")
        # 尝试常见路径
        candidates = [
            "../MM_class/data_changyin",
            "../../MM_class/data_changyin",
            "data_changyin",
        ]
        for cand in candidates:
            test = sorted(glob.glob(os.path.join(cand, "*.wav")))
            if test:
                print(f"  找到数据在: {cand}/ ({len(test)} WAV)")
        sys.exit(1)
    all_events = []  # [(filename, start, end, label, audio_segment)]

    print("收集所有标注事件...")
    for wf in tqdm(wav_files):
        base = os.path.splitext(wf)[0]
        tf = base + ".txt"
        if not os.path.exists(tf): continue
        try: audio = load_wav(wf)
        except: continue

        for st, et, lab in parse_txt(tf):
            s, e = max(0, int(st*SR_TARGET)), min(len(audio), int(et*SR_TARGET))
            if e <= s: continue
            all_events.append((os.path.basename(wf), st, et, lab, audio[s:e].copy()))

    # 采样策略: h和g全部取, n取2000
    events_by_class = {"n": [], "g": [], "h": []}
    for ev in all_events:
        events_by_class[ev[3]].append(ev)

    for cls in events_by_class:
        if len(events_by_class[cls]) > MAX_EVENTS_PER_CLASS[cls]:
            events_by_class[cls] = random.sample(events_by_class[cls], MAX_EVENTS_PER_CLASS[cls])

    selected = []
    for cls in ['n', 'g', 'h']:
        selected.extend(events_by_class[cls])
    random.shuffle(selected)

    print(f"Selected: n={len(events_by_class['n'])}, g={len(events_by_class['g'])}, h={len(events_by_class['h'])}")

    # --- 2. 加载模型 ---
    if trained_model is not None:
        device = next(trained_model.parameters()).device
    else:
        beats = load_beats(args.beats_code_path, args.beats_ckpt_path)
        beats.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        beats = beats.to(device)

    # --- 3. 提取 embedding ---
    source_name = "TrainedModel" if trained_model else "BEATs_raw"
    print(f"Extracting {source_name} embeddings...")
    embeddings = []
    logits_all = []
    labels = []
    class_names = {0: 'n', 1: 'g', 2: 'h'}
    label_to_id = {'n': 0, 'g': 1, 'h': 2}

    batch_size = 32
    for i in tqdm(range(0, len(selected), batch_size)):
        batch = selected[i:i+batch_size]
        audios = []
        for _, _, _, lab, seg in batch:
            y = pad_or_clip(seg, AUDIO_LENGTH)
            audios.append(torch.from_numpy(y).float())
            labels.append(label_to_id[lab])

        batch_audio = torch.stack(audios, dim=0).to(device)

        with torch.no_grad():
            if trained_model is not None:
                phys_batch = torch.stack([
                    torch.from_numpy(compute_phys_feats_np(
                        pad_or_clip(seg, AUDIO_LENGTH)))
                    for _, _, _, _, seg in batch
                ], dim=0).to(device)
                _, proj, logits_out = trained_model(batch_audio, phys_batch)
                emb = proj.cpu().numpy()
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)  # 单样本时 reshape
                logits_all.append(logits_out.cpu().numpy() if logits_out is not None else np.zeros((len(batch), 2)))
            else:
                result = beats.extract_features(batch_audio)
                feats = result[0] if isinstance(result, tuple) else result
                emb = feats.mean(dim=1).cpu().numpy()

            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    embeddings = np.concatenate(embeddings, axis=0)
    # 修复: 如果被 flatten 了, reshape 回 (samples, dim)
    if embeddings.ndim == 1:
        emb_dim = embeddings.shape[0] // len(labels)
        embeddings = embeddings.reshape(len(labels), emb_dim)
        print(f"WARNING: embeddings were flat, reshaped to {embeddings.shape}")

    labels = np.array(labels)

    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}")
    print(f"Class dist: {Counter(labels.tolist())}")

    # --- 训练后模型: 直接分类精度 ---
    if trained_model is not None and logits_all:
        logits_all = np.concatenate(logits_all, axis=0)
        preds = logits_all.argmax(1)
        direct_acc = (preds == labels).mean()
        direct_bal = balanced_accuracy_score(labels, preds)
        print(f"\n{'='*60}")
        print(f"训练后模型直接分类 (classifier head)")
        print(f"{'='*60}")
        print(f"  Accuracy:      {direct_acc:.4f}")
        print(f"  Balanced Acc:  {direct_bal:.4f}")
        cm_direct = confusion_matrix(labels, preds)
        print(f"  Confusion Matrix:\n{cm_direct}")
        for i, cls in enumerate(['n', 'g', 'h']):
            rec = cm_direct[i,i]/cm_direct[i].sum() if cm_direct[i].sum()>0 else 0
            print(f"    {cls} recall: {rec:.4f}")

    # --- 4. 线性分类器测试 (理论上限) ---
    print("\n" + "="*60)
    print(f"Logistic Regression 分类测试 ({source_name} 特征的理论上限)")
    print("="*60)

    # 分层拆分
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(embeddings, labels))

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3-fold CV 调 C
    from sklearn.model_selection import cross_val_score
    best_c, best_score = 1.0, 0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=1000, class_weight='balanced',
                                 random_state=args.seed)
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='f1_macro')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_c = C

    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight='balanced',
                             random_state=args.seed)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print(f"\nBest C={best_c}")
    print(classification_report(y_test, y_pred, target_names=['n','g','h'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 各类 recall
    cm = confusion_matrix(y_test, y_pred)
    for i, cls in enumerate(['n', 'g', 'h']):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {cls} recall: {recall:.4f}")

    # --- 5. 可视化 ---
    print("\nGenerating plots...")

    # PCA
    pca = PCA(n_components=2)
    emb_2d_pca = pca.fit_transform(StandardScaler().fit_transform(embeddings))

    # t-SNE (只用一部分 n, 全量 g/h)
    tsne_n = min(500, (labels == 0).sum())
    tsne_idx = np.concatenate([
        np.random.choice(np.where(labels == 0)[0], tsne_n, replace=False),
        np.where(labels == 1)[0],
        np.where(labels == 2)[0],
    ])
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30)
    emb_tsne_sub = tsne.fit_transform(StandardScaler().fit_transform(embeddings[tsne_idx]))
    tsne_labels = labels[tsne_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'n': '#2196F3', 'g': '#FF9800', 'h': '#F44336'}
    markers = {'n': '.', 'g': '^', 'h': '*'}

    # PCA 全量
    for cls_id, cls_name in class_names.items():
        mask = labels == cls_id
        axes[0].scatter(emb_2d_pca[mask, 0], emb_2d_pca[mask, 1],
                       c=colors[cls_name], marker=markers[cls_name],
                       alpha=0.5 if cls_name == 'n' else 0.9,
                       s=5 if cls_name == 'n' else 80,
                       label=f"{cls_name} ({mask.sum()})",
                       edgecolors='black' if cls_name != 'n' else 'none',
                       linewidth=0.5)
    axes[0].set_title(f'PCA (var={pca.explained_variance_ratio_.sum():.1%})', fontsize=12)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # t-SNE
    for cls_id, cls_name in class_names.items():
        mask_tsne = tsne_labels == cls_id
        axes[1].scatter(emb_tsne_sub[mask_tsne, 0], emb_tsne_sub[mask_tsne, 1],
                       c=colors[cls_name], marker=markers[cls_name],
                       alpha=0.5 if cls_name == 'n' else 0.9,
                       s=5 if cls_name == 'n' else 80,
                       label=f"{cls_name} ({mask_tsne.sum()})",
                       edgecolors='black' if cls_name != 'n' else 'none',
                       linewidth=0.5)
    axes[1].set_title('t-SNE (subsampled n)', fontsize=12)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # 各类到 n 类中心的距离分布
    n_center = np.mean(embeddings[labels == 0], axis=0)
    for cls_id, cls_name in class_names.items():
        mask = labels == cls_id
        dists = np.linalg.norm(embeddings[mask] - n_center, axis=1)
        axes[2].hist(dists, bins=30, alpha=0.5, label=f'{cls_name}',
                    color=colors[cls_name], density=True)
    axes[2].set_xlabel('到 n 类中心的欧氏距离')
    axes[2].set_title('各类到 n 类中心的距离分布', fontsize=12)
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle(f'{source_name} 特征分离度 — 肠音 n/g/h 三类', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_embedding_visualization.png", dpi=150)
    plt.close()
    print("  -> beats_probe/01_embedding_visualization.png")

    # --- 6. 定量评估 ---
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(embeddings[tsne_idx], tsne_labels)
    print(f"\nSilhouette Score (t-SNE subset): {sil:.4f}")

    # 类间距离矩阵
    print("\n类间距离 (标准化后类中心的欧氏距离):")
    centers = {}
    for cls_id, cls_name in class_names.items():
        centers[cls_name] = np.mean(StandardScaler().fit_transform(embeddings)[labels == cls_id], axis=0)
    for c1 in ['n', 'g', 'h']:
        row = []
        for c2 in ['n', 'g', 'h']:
            row.append(f"{np.linalg.norm(centers[c1] - centers[c2]):.3f}")
        print(f"  {c1}: {'  '.join(row)}")

    # 单特征分离度
    print("\n单维特征的 Fisher 判别比 (top-10):")
    scaler_all = StandardScaler()
    emb_scaled = scaler_all.fit_transform(embeddings)
    fisher_ratios = []
    for dim in range(embeddings.shape[1]):
        between = 0
        within = 0
        grand_mean = np.mean(emb_scaled[:, dim])
        for cls_id in range(3):
            cls_vals = emb_scaled[labels == cls_id, dim]
            cls_mean = np.mean(cls_vals)
            between += len(cls_vals) * (cls_mean - grand_mean) ** 2
            within += np.sum((cls_vals - cls_mean) ** 2)
        fisher_ratios.append((dim, between / (within + 1e-10)))
    for dim, ratio in sorted(fisher_ratios, key=lambda x: -x[1])[:10]:
        print(f"  dim_{dim:04d}: Fisher={ratio:.4f}")

    # --- 保存汇总 ---
    summary = {
        "source": source_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
        "class_dist": {k: v for k, v in Counter(labels.tolist()).items()},
    }
    if trained_model is not None and logits_all is not None:
        summary["direct_accuracy"] = float(direct_acc)
        summary["direct_balanced_acc"] = float(direct_bal)
    # LR results
    summary["lr_best_c"] = float(best_c)
    summary["lr_report"] = classification_report(y_test, y_pred, target_names=['n','g','h'],
                                                  output_dict=True, zero_division=0)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Charts in {out_dir}/, compare with beats_probe/ (baseline)")

if __name__ == "__main__":
    main()
