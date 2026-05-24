"""
Prepare binary changyin audio data for BEATs training.

Input:
  data_changyin/*.wav
  data_changyin/*.txt

TXT label rule:
  - Each valid row starts with start_time end_time.
  - If the last column is "g" or "h", the segment is rare -> class_1.
  - If the last column is missing or anything else, the segment is normal -> class_0.

Output:
  data_changyin_audio/train/class_0/*.wav
  data_changyin_audio/train/class_1/*.wav
  data_changyin_audio/val/class_0/*.wav
  data_changyin_audio/val/class_1/*.wav
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, get_window, resample_poly
from tqdm import tqdm


DATA_DIR = "data_changyin"
OUTPUT_DIR = "data_changyin_audio"
SR_TARGET = 16000
VAL_RATIO = 0.15
SEED = 42

H_AUG_PER_SAMPLE = 45
G_AUG_PER_SAMPLE = 7
N_MAX_SAMPLES = 4000

N_FFT = 1024
HOP = 256


def load_wav(path):
    sr, data = wavfile.read(path)
    orig_dtype = data.dtype
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        scale = max(abs(info.min), info.max)
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if sr != SR_TARGET:
        data = resample_poly(data, SR_TARGET, sr).astype(np.float32)
    return data


def save_wav(path, y):
    y_int = (y * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(path, SR_TARGET, y_int)


def parse_event_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    try:
        start_time = float(parts[0])
        end_time = float(parts[1])
    except ValueError:
        return None

    label = parts[-1].lower() if len(parts) >= 3 else "n"
    if label not in ("g", "h"):
        label = "n"
    return start_time, end_time, label


def stft(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    n_frames = 1 + max(0, len(y) - N_FFT) // HOP
    win = get_window("hann", N_FFT, fftbins=True)
    spec = np.zeros((N_FFT // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        spec[:, i] = fft(y[i * HOP:i * HOP + N_FFT] * win)[:N_FFT // 2 + 1]
    return spec


def check_h_constraints(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    spec = np.abs(stft(y))
    power = spec ** 2 / N_FFT
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR_TARGET)

    centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    diff = freqs[:, None] - centroid[None, :]
    bandwidth = np.sqrt(np.sum(diff ** 2 * power, axis=0) / (np.sum(power, axis=0) + 1e-10))

    lo_mask = (freqs >= 200) & (freqs <= 500)
    hi_mask = freqs > 4000
    total_en = np.sum(power)
    lo_ratio = np.sum(power[lo_mask]) / (total_en + 1e-10)
    hi_ratio = np.sum(power[hi_mask]) / (total_en + 1e-10)

    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))
    crest = peak / (rms + 1e-10)
    zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))

    return (
        180 <= np.mean(centroid) <= 480
        and 60 <= np.mean(bandwidth) <= 450
        and lo_ratio >= 0.60
        and hi_ratio <= 0.10
        and 0.015 <= zcr <= 0.060
        and 1.5 <= crest <= 20.0
    )


def aug_timestretch(y):
    rate = np.random.uniform(0.88, 1.12)
    old_len = len(y)
    new_len = int(old_len / rate)
    interp = interp1d(
        np.arange(old_len),
        y.astype(np.float64),
        kind="linear",
        assume_sorted=True,
        bounds_error=False,
        fill_value=0,
    )
    return interp(np.linspace(0, old_len - 1, new_len)).astype(np.float32)


def aug_pitchshift(y):
    cents = np.random.uniform(-150, 150)
    rate = 2.0 ** (cents / 1200.0)
    stretched_len = max(4, int(len(y) * rate))
    interp = interp1d(
        np.arange(len(y)),
        y.astype(np.float64),
        kind="linear",
        assume_sorted=True,
        bounds_error=False,
        fill_value=0,
    )
    stretched = interp(np.linspace(0, len(y) - 1, stretched_len)).astype(np.float32)
    return resample_poly(stretched, len(y), stretched_len).astype(np.float32)[:len(y)]


def aug_gain(y):
    return y * (10 ** (np.random.uniform(-6, 6) / 20))


def aug_phase(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    n_fft, hop = N_FFT, N_FFT // 4
    n_frames = 1 + (len(y) - n_fft) // hop
    win = get_window("hann", n_fft, fftbins=True)
    mags, phases = [], []
    for i in range(n_frames):
        spec = fft(y[i * hop:i * hop + n_fft] * win)
        mags.append(np.abs(spec))
        phases.append(np.angle(spec))
    new_phases = [phase + np.random.uniform(0, 2 * np.pi, phase.shape) * 0.3 for phase in phases]
    result = np.zeros(len(y) + n_fft)
    win_sum = np.zeros_like(result)
    for i, (mag, phase) in enumerate(zip(mags, new_phases)):
        frame = np.real(np.fft.ifft(mag * np.exp(1j * phase))) * win
        start = i * hop
        result[start:start + n_fft] += frame
        win_sum[start:start + n_fft] += win ** 2
    return result[:len(y)] / np.maximum(win_sum[:len(y)], 1e-10)


def aug_low_eq(y):
    db = np.random.uniform(-3, 3)
    gain = 10 ** (db / 20)
    nyq = SR_TARGET / 2
    b_lo, a_lo = butter(2, 500 / nyq, btype="low")
    b_hi, a_hi = butter(2, 200 / nyq, btype="high")
    band = filtfilt(b_hi, a_hi, filtfilt(b_lo, a_lo, y))
    return y + band * (gain - 1)


def generate_h_variants(seg, target_n=45):
    valid = []
    methods = [
        (aug_timestretch, 7),
        (aug_pitchshift, 6),
        (aug_gain, 5),
        (aug_phase, 4),
        (aug_low_eq, 3),
        (lambda s: aug_pitchshift(aug_timestretch(s)), 5),
        (lambda s: aug_gain(aug_timestretch(s)), 5),
        (lambda s: aug_gain(aug_pitchshift(s)), 4),
        (lambda s: aug_phase(aug_timestretch(s)), 3),
        (lambda s: aug_gain(aug_low_eq(s)), 3),
    ]
    for fn, target in methods:
        got = 0
        for _ in range(target * 3):
            if len(valid) >= target_n:
                break
            try:
                aug = fn(seg).astype(np.float32)
                if check_h_constraints(aug):
                    valid.append(aug)
                    got += 1
            except Exception:
                continue
            if got >= target:
                break
    return valid[:target_n]


def generate_g_variants(seg, target_n=7):
    valid = []
    methods = [
        (aug_timestretch, 2),
        (aug_pitchshift, 2),
        (aug_gain, 2),
        (lambda s: aug_pitchshift(aug_timestretch(s)), 1),
    ]
    for fn, target in methods:
        got = 0
        for _ in range(target * 2):
            if len(valid) >= target_n:
                break
            try:
                aug = fn(seg).astype(np.float32)
                if len(aug) > 64:
                    valid.append(aug)
                    got += 1
            except Exception:
                continue
            if got >= target:
                break
    return valid[:target_n]


def file_has_rare_label(wav_file):
    txt_file = os.path.splitext(wav_file)[0] + ".txt"
    if not os.path.exists(txt_file):
        return False
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_event_line(line)
            if parsed is not None and parsed[2] in ("g", "h"):
                return True
    return False


def split_files(wav_files, val_ratio):
    rare_files = [wf for wf in wav_files if file_has_rare_label(wf)]
    rare_set = set(rare_files)
    normal_files = [wf for wf in wav_files if wf not in rare_set]
    random.shuffle(rare_files)
    random.shuffle(normal_files)

    def split_group(files):
        if len(files) <= 1:
            return files, []
        n_val = max(1, int(round(len(files) * val_ratio)))
        n_val = min(n_val, len(files) - 1)
        return files[n_val:], files[:n_val]

    rare_train, rare_val = split_group(rare_files)
    normal_train, normal_val = split_group(normal_files)
    train_files = rare_train + normal_train
    val_files = rare_val + normal_val
    random.shuffle(train_files)
    random.shuffle(val_files)
    return train_files, val_files, rare_files, normal_files


def extract_segments(wav_files):
    n_segs, g_segs, h_segs = [], [], []
    skipped_files = 0
    for wav_file in tqdm(wav_files, desc="Extracting"):
        txt_file = os.path.splitext(wav_file)[0] + ".txt"
        if not os.path.exists(txt_file):
            skipped_files += 1
            continue
        try:
            audio = load_wav(wav_file)
        except Exception:
            skipped_files += 1
            continue
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_event_line(line)
                if parsed is None:
                    continue
                start_time, end_time, label = parsed
                start = max(0, int(start_time * SR_TARGET))
                end = min(len(audio), int(end_time * SR_TARGET))
                if end <= start:
                    continue
                seg = audio[start:end]
                if label == "g":
                    g_segs.append(seg)
                elif label == "h":
                    h_segs.append(seg)
                else:
                    n_segs.append(seg)
    return n_segs, g_segs, h_segs, skipped_files


def save_class(segments, output_dir, split, cls_name):
    class_dir = os.path.join(output_dir, split, cls_name)
    os.makedirs(class_dir, exist_ok=True)
    for old_file in glob.glob(os.path.join(class_dir, "*.wav")):
        os.remove(old_file)
    for i, segment in enumerate(tqdm(segments, desc=f"{split}/{cls_name}", leave=False)):
        save_wav(os.path.join(class_dir, f"{i:06d}.wav"), segment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_max_samples", type=int, default=N_MAX_SAMPLES)
    parser.add_argument("--g_aug_per_sample", type=int, default=G_AUG_PER_SAMPLE)
    parser.add_argument("--h_aug_per_sample", type=int, default=H_AUG_PER_SAMPLE)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {args.data_dir}")

    train_files, val_files, rare_files, normal_files = split_files(wav_files, args.val_ratio)
    print(f"Files: total={len(wav_files)}, rare_files={len(rare_files)}, normal_only_files={len(normal_files)}")
    print(f"File-level split: train={len(train_files)}, val={len(val_files)}")

    n_train_all, g_train_orig, h_train_orig, train_skipped = extract_segments(train_files)
    n_val, g_val_orig, h_val_orig, val_skipped = extract_segments(val_files)

    random.shuffle(n_train_all)
    n_train = n_train_all[:args.n_max_samples]

    print(
        f"Original events: train n={len(n_train_all)}, g={len(g_train_orig)}, h={len(h_train_orig)}; "
        f"val n={len(n_val)}, g={len(g_val_orig)}, h={len(h_val_orig)}"
    )

    print("Augmenting h train samples...")
    h_train_aug = []
    for seg in tqdm(h_train_orig):
        h_train_aug.extend(generate_h_variants(seg, args.h_aug_per_sample))

    print("Augmenting g train samples...")
    g_train_aug = []
    for seg in tqdm(g_train_orig):
        g_train_aug.extend(generate_g_variants(seg, args.g_aug_per_sample))

    rare_train = g_train_orig + h_train_orig + g_train_aug + h_train_aug
    rare_val = g_val_orig + h_val_orig
    random.shuffle(rare_train)
    random.shuffle(rare_val)

    print(
        f"Train saved: n={len(n_train)}, rare={len(rare_train)} "
        f"(orig_g={len(g_train_orig)}, orig_h={len(h_train_orig)}, "
        f"aug_g={len(g_train_aug)}, aug_h={len(h_train_aug)})"
    )
    print(
        f"Val saved: n={len(n_val)}, rare={len(rare_val)} "
        f"(orig_g={len(g_val_orig)}, orig_h={len(h_val_orig)}, no augmentation)"
    )

    save_class(n_train, args.output_dir, "train", "class_0")
    save_class(rare_train, args.output_dir, "train", "class_1")
    save_class(n_val, args.output_dir, "val", "class_0")
    save_class(rare_val, args.output_dir, "val", "class_1")

    stats = {
        "label_rule": "last column g/h -> class_1 rare; missing or other label -> class_0 normal",
        "split_unit": "wav_file",
        "seed": args.seed,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "total_wav_files": len(wav_files),
        "rare_wav_files": len(rare_files),
        "normal_only_wav_files": len(normal_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_n_original": len(n_train_all),
        "train_n_saved": len(n_train),
        "train_g_original": len(g_train_orig),
        "train_h_original": len(h_train_orig),
        "train_g_augmented": len(g_train_aug),
        "train_h_augmented": len(h_train_aug),
        "train_rare_saved": len(rare_train),
        "val_n_saved": len(n_val),
        "val_g_original": len(g_val_orig),
        "val_h_original": len(h_val_orig),
        "val_rare_saved": len(rare_val),
        "skipped_train_files": train_skipped,
        "skipped_val_files": val_skipped,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nDone -> {args.output_dir}/")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
