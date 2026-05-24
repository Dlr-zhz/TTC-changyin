"""
将 data_changyin 音频事件转为 mel频谱图, 组织为 TTC 所需的 ImageFolder 格式
二分类: class_0=n(正常), class_1=rare(g+h合并)
"""
import os, sys, glob, random, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, get_window, butter, filtfilt
from scipy.fftpack import fft
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# ==================== 配置 ====================
DATA_DIR = "../MM_class/data_changyin"
OUTPUT_DIR = "data_changyin_binary"
SR_TARGET = 16000
N_FFT = 1024
HOP = 256
N_MELS = 128
FMIN, FMAX = 0, 8000
IMG_SIZE = 224
VAL_RATIO = 0.15
SEED = 42

# h类增强参数
H_AUG_PER_SAMPLE = 45   # 每个h → 45变体, 24 → 1080
G_AUG_PER_SAMPLE = 7    # 每个g → 7变体, 179 → 1253
N_MAX_SAMPLES = 4000    # n类下采样

random.seed(SEED)
np.random.seed(SEED)

# ==================== 工具函数 ====================
def load_wav(path, target_sr=SR_TARGET):
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
    if info is not None: data = data / (info.max + 1.0)
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr)
    return data

def mel_filterbank(n_fft, sr, n_mels, fmin, fmax):
    def hz2mel(hz): return 2595 * np.log10(1 + hz/700)
    def mel2hz(m): return 700 * (10**(m/2595) - 1)
    n_freqs = n_fft // 2 + 1
    m_min, m_max = hz2mel(fmin), hz2mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    bins = np.clip(np.floor((n_fft+1)*mel2hz(m_pts)/sr).astype(int), 0, n_freqs-1)
    fb = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        if c > l: fb[i, l:c] = np.linspace(0, 1, c-l, endpoint=False)
        if r > c: fb[i, c:r] = np.linspace(1, 0, r-c, endpoint=False)
    return fb

MEL_FB = mel_filterbank(N_FFT, SR_TARGET, N_MELS, FMIN, FMAX)

def stft(y, n_fft, hop):
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + max(0, len(y) - n_fft) // hop
    win = get_window('hann', n_fft, fftbins=True)
    S = np.zeros((n_fft//2+1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        S[:, i] = fft(y[i*hop:i*hop+n_fft] * win)[:n_fft//2+1]
    return S

def audio_to_mel_image(y, sr=SR_TARGET):
    """将音频转换为224×224×3的mel频谱图"""
    S = np.abs(stft(y, N_FFT, HOP))
    power = S**2 / N_FFT
    mel_power = MEL_FB @ power
    mel_db = 10 * np.log10(np.maximum(mel_power, 1e-10))

    # 归一化到 [0, 255]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # resize 到 224×224
    from PIL import Image
    img = Image.fromarray((mel_norm * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    # 转为3通道 (灰度复制)
    img_rgb = np.stack([np.array(img)]*3, axis=-1)
    return img_rgb

# ==================== 约束增强 (h专用) ====================
H_CONSTRAINTS = {
    'centroid': (180, 480),
    'bandwidth': (60, 450),
    'lo_energy_ratio': (0.60, 1.0),
    'zcr': (0.015, 0.060),
    'crest': (1.5, 20.0),
    'hi_ratio': (0.0, 0.10),
}

def check_constraints(y):
    """检查音频是否符合h物理特征"""
    from scipy.signal import hilbert
    if len(y) < N_FFT: y = np.pad(y, (0, N_FFT - len(y)))
    S = np.abs(stft(y, N_FFT, HOP))
    power = S**2 / N_FFT
    freqs = np.fft.rfftfreq(N_FFT, 1/SR_TARGET)

    centroid = np.sum(freqs[:,None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    diff = freqs[:,None] - centroid[None, :]
    bandwidth = np.sqrt(np.sum(diff**2 * power, axis=0) / (np.sum(power, axis=0) + 1e-10))

    mask_lo = (freqs >= 200) & (freqs <= 500)
    mask_hi = freqs > 4000
    total_en = np.sum(power)
    lo_ratio = np.sum(power[mask_lo]) / (total_en + 1e-10)
    hi_ratio = np.sum(power[mask_hi]) / (total_en + 1e-10)

    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    crest = peak / (rms + 1e-10)
    zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2*len(y))

    c, bw = float(np.mean(centroid)), float(np.mean(bandwidth))
    ok = (H_CONSTRAINTS['centroid'][0] <= c <= H_CONSTRAINTS['centroid'][1] and
          H_CONSTRAINTS['bandwidth'][0] <= bw <= H_CONSTRAINTS['bandwidth'][1] and
          H_CONSTRAINTS['lo_energy_ratio'][0] <= lo_ratio <= H_CONSTRAINTS['lo_energy_ratio'][1] and
          H_CONSTRAINTS['zcr'][0] <= zcr <= H_CONSTRAINTS['zcr'][1] and
          H_CONSTRAINTS['crest'][0] <= crest <= H_CONSTRAINTS['crest'][1] and
          H_CONSTRAINTS['hi_ratio'][0] <= hi_ratio <= H_CONSTRAINTS['hi_ratio'][1])
    return ok

def aug_timestretch(y, rate_range=(0.88, 1.12)):
    from scipy.interpolate import interp1d
    rate = np.random.uniform(*rate_range)
    old_len = len(y)
    new_len = int(old_len / rate)
    f = interp1d(np.arange(old_len), y, kind='linear', assume_sorted=True,
                 bounds_error=False, fill_value=0)
    return f(np.linspace(0, old_len-1, new_len)).astype(np.float32)

def aug_pitchshift(y, cents_range=(-150, 150)):
    """通过拉伸+重采样实现音调偏移, 保持时长不变"""
    from scipy.interpolate import interp1d
    cents = np.random.uniform(*cents_range)
    rate = 2.0 ** (cents / 1200.0)  # >1=更多采样点=低音, <1=更少=高音
    old_len = len(y)
    # 线性插值拉伸
    stretched_len = max(4, int(old_len * rate))
    f = interp1d(np.arange(old_len), y.astype(np.float64), kind='linear',
                 assume_sorted=True, bounds_error=False, fill_value=0)
    stretched = f(np.linspace(0, old_len-1, stretched_len)).astype(np.float32)
    # 重采样回原长
    return resample_poly(stretched, old_len, stretched_len).astype(np.float32)[:old_len]

def aug_gain(y, db_range=(-6, 6)):
    db = np.random.uniform(*db_range)
    return y * (10**(db/20))

def aug_phase(y):
    if len(y) < N_FFT: y = np.pad(y, (0, N_FFT-len(y)))
    n_fft, hop = N_FFT, N_FFT//4
    n_frames = 1 + (len(y) - n_fft) // hop
    win = get_window('hann', n_fft, fftbins=True)
    mags, phases = [], []
    for i in range(n_frames):
        frame = y[i*hop:i*hop+n_fft] * win
        s = fft(frame)
        mags.append(np.abs(s)); phases.append(np.angle(s))
    new_phases = [ph + np.random.uniform(0, 2*np.pi, ph.shape) * 0.3 for ph in phases]
    from scipy.signal import istft
    new_specs = [m * np.exp(1j*p) for m, p in zip(mags, new_phases)]
    result = np.zeros(len(y) + n_fft)
    win_sum = np.zeros_like(result)
    for i, spec in enumerate(new_specs):
        frame = np.real(np.fft.ifft(spec)) * win
        start = i*hop
        result[start:start+n_fft] += frame
        win_sum[start:start+n_fft] += win**2
    return result[:len(y)] / np.maximum(win_sum[:len(y)], 1e-10)

def aug_low_eq(y, db_range=(-3, 3)):
    db = np.random.uniform(*db_range)
    gain = 10**(db/20)
    nyq = SR_TARGET/2
    b_lo, a_lo = butter(2, 500/nyq, btype='low')
    b_hi, a_hi = butter(2, 200/nyq, btype='high')
    lo = filtfilt(b_lo, a_lo, y)
    band = filtfilt(b_hi, a_hi, lo)
    return y + band * (gain - 1)

def generate_h_variants(seg, target_n=45):
    """对h样本生成约束通过的有效变体"""
    valid = []
    methods = [
        (aug_timestretch, 7), (aug_pitchshift, 6), (aug_gain, 5),
        (aug_phase, 4), (aug_low_eq, 3),
        (lambda s: aug_pitchshift(aug_timestretch(s)), 5),
        (lambda s: aug_gain(aug_timestretch(s)), 5),
        (lambda s: aug_gain(aug_pitchshift(s)), 4),
        (lambda s: aug_phase(aug_timestretch(s)), 3),
        (lambda s: aug_gain(aug_low_eq(s)), 3),
    ]
    for fn, target in methods:
        got = 0
        for _ in range(target * 3):
            if len(valid) >= target_n: break
            try:
                aug = fn(seg)
                if check_constraints(aug):
                    valid.append(aug)
                    got += 1
            except: continue
            if got >= target: break
        if len(valid) >= target_n: break
    return valid[:target_n]

def generate_g_variants(seg, target_n=7):
    valid = []
    methods = [
        (lambda s: aug_timestretch(s, (0.85, 1.15)), 2),
        (lambda s: aug_pitchshift(s, (-300, 300)), 2),
        (lambda s: aug_gain(s, (-8, 8)), 2),
        (lambda s: aug_pitchshift(aug_timestretch(s, (0.88, 1.12)), (-200, 200)), 1),
    ]
    for fn, target in methods:
        got = 0
        for _ in range(target * 2):
            if len(valid) >= target_n: break
            try:
                aug = fn(seg)
                if len(aug) > 64: valid.append(aug); got += 1
            except: continue
            if got >= target: break
    return valid[:target_n]

# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))

    # 收集所有事件
    n_segments = []
    g_segments = []
    h_segments = []

    for wf in tqdm(wav_files, desc="扫描音频文件"):
        base = os.path.splitext(wf)[0]
        tf = base + ".txt"
        if not os.path.exists(tf): continue
        try: audio = load_wav(wf)
        except: continue

        with open(tf, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                try:
                    st, et = float(parts[0]), float(parts[1])
                    lab = parts[2] if len(parts) >= 3 and parts[2] in ('n','g','h') else 'n'
                except: continue
                s, e = max(0, int(st*SR_TARGET)), min(len(audio), int(et*SR_TARGET))
                if e <= s: continue
                seg = audio[s:e]
                if lab == 'h': h_segments.append(seg)
                elif lab == 'g': g_segments.append(seg)
                elif lab == 'n': n_segments.append(seg)

    print(f"事件统计: n={len(n_segments)}, g={len(g_segments)}, h={len(h_segments)}")

    # 数据集划分 (分层)
    random.shuffle(n_segments)
    random.shuffle(g_segments)
    random.shuffle(h_segments)

    # n: 15% → val
    n_val_n = int(len(n_segments) * VAL_RATIO)
    n_train = n_segments[n_val_n:][:N_MAX_SAMPLES]
    n_val = n_segments[:n_val_n]

    # g: 15% → val
    g_val_n = max(1, int(len(g_segments) * VAL_RATIO))
    g_train_orig = g_segments[g_val_n:]
    g_val_orig = g_segments[:g_val_n]

    # h: 15% → val
    h_val_n = max(1, int(len(h_segments) * VAL_RATIO))
    h_train_orig = h_segments[h_val_n:]
    h_val_orig = h_segments[:h_val_n]

    print(f"Train: n={len(n_train)}, g={len(g_train_orig)}, h={len(h_train_orig)}")
    print(f"Val:   n={len(n_val)}, g={len(g_val_orig)}, h={len(h_val_orig)}")

    # 增强 h 和 g
    print("增强 h...")
    h_train_aug = []
    for seg in tqdm(h_train_orig):
        variants = generate_h_variants(seg, H_AUG_PER_SAMPLE)
        h_train_aug.extend(variants)

    h_val_aug = []
    for seg in h_val_orig:
        variants = generate_h_variants(seg, max(1, H_AUG_PER_SAMPLE//3))
        h_val_aug.extend(variants)

    print(f"h 增强: train {len(h_train_orig)}→{len(h_train_aug)}, val {len(h_val_orig)}→{len(h_val_aug)}")

    print("增强 g...")
    g_train_aug = []
    for seg in tqdm(g_train_orig):
        variants = generate_g_variants(seg, G_AUG_PER_SAMPLE)
        g_train_aug.extend(variants)

    g_val_aug = []
    for seg in g_val_orig:
        variants = generate_g_variants(seg, 2)
        g_val_aug.extend(variants)

    print(f"g 增强: train {len(g_train_orig)}→{len(g_train_aug)}, val {len(g_val_orig)}→{len(g_val_aug)}")

    # 合并 rare = g + h
    rare_train = g_train_aug + h_train_aug
    rare_val = g_val_aug + h_val_aug
    random.shuffle(rare_train)
    random.shuffle(rare_val)

    print(f"\n最终数据集:")
    print(f"  Train: class_0(n)={len(n_train)}, class_1(rare)={len(rare_train)}")
    print(f"  Val:   class_0(n)={len(n_val)}, class_1(rare)={len(rare_val)}")

    # 生成 mel 图像并保存
    def save_images(segments, split, class_name):
        img_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(img_dir, exist_ok=True)
        for i, seg in enumerate(tqdm(segments, desc=f"{split}/{class_name}", leave=False)):
            img = audio_to_mel_image(seg)
            Image.fromarray(img).save(os.path.join(img_dir, f"{i:06d}.png"))

    save_images(n_train, "train", "class_0")
    save_images(rare_train, "train", "class_1")
    save_images(n_val, "val", "class_0")
    save_images(rare_val, "val", "class_1")

    # 保存数据集统计
    stats = {
        "train_class_0_n": len(n_train),
        "train_class_1_rare": len(rare_train),
        "val_class_0_n": len(n_val),
        "val_class_1_rare": len(rare_val),
        "imbalance_ratio_train": len(n_train) / max(len(rare_train), 1),
        "imbalance_ratio_val": len(n_val) / max(len(rare_val), 1),
        "rare_train_composition": {"g": len(g_train_aug), "h": len(h_train_aug)},
        "rare_val_composition": {"g": len(g_val_aug), "h": len(h_val_aug)},
    }
    with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. 数据保存在 {OUTPUT_DIR}/")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
