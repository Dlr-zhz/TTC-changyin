"""分析 n/g/h 三类事件的时间分布"""
import os, glob, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "data_changyin"
SEED = 42

all_events = {"n": [], "g": [], "h": []}

for tf in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
    with open(tf) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            try:
                st, et = float(parts[0]), float(parts[1])
                lab = parts[2] if len(parts) >= 3 and parts[2] in ('n','g','h') else 'n'
                dur = et - st
                if dur > 0 and dur < 120:  # 过滤异常值
                    all_events[lab].append(dur)
            except: continue

for lab in ['n','g','h']:
    arr = np.array(all_events[lab])
    print(f"\n{lab}: {len(arr)} events")
    print(f"  mean={arr.mean():.2f}s, std={arr.std():.2f}s")
    print(f"  min={arr.min():.2f}s, max={arr.max():.2f}s")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(arr, p):.2f}s")

# 画图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {'n': '#2196F3', 'g': '#FF9800', 'h': '#F44336'}
bins = np.linspace(0, 30, 60)

for ax, lab in zip(axes, ['n','g','h']):
    arr = np.array(all_events[lab])
    ax.hist(arr, bins=bins, color=colors[lab], alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(arr.mean(), color='black', linestyle='--', linewidth=2, label=f'mean={arr.mean():.1f}s')
    ax.axvline(np.median(arr), color='red', linestyle=':', linewidth=2, label=f'median={np.median(arr):.1f}s')
    ax.set_title(f'{lab} ({len(arr)} events)', fontsize=12)
    ax.set_xlabel('Duration (s)')
    ax.set_xlim(0, 30)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('肠音三类事件时长分布', fontsize=14)
plt.tight_layout()
plt.savefig('duration_distribution.png', dpi=150)
plt.close()
print("\nDone -> duration_distribution.png")
