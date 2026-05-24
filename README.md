# 肠音二分类 — BEATs + SupMin 对比学习

基于 BEATs (AudioSet 预训练) 做音频编码，TTC 的 SupMin 对比损失处理类别不平衡。

## 方案架构

```
WAV音频 → BEATs(frozen, iter3) → [B, T, 768] → Mean Pool → [B, 768]
                                                              ↓
                                              ┌───────────────┴───────────────┐
                                              ↓                               ↓
                                    Projection Head                     Online Evaluator
                                    768→2048→128                        768→256→2
                                              ↓                               ↓
                                       SupMin Loss                    Cross-Entropy
                                    (只聚焦少数类)                    (分类准确率监控)
```

**为什么这样设计**：
- BEATs 在 AudioSet 200 万音频上预训练过，已经学会通用声音模式
- 冻结 BEATs，只训练投影头+分类头 → 5,800 个样本够用
- SupMin (ratio=0.0) 只对 rare 类做对比学习，n 类走标准 NT-Xent
- 输入是原始波形 (16kHz)，不需要 mel→PNG 转换

## 项目结构

```
TTC_changyin/
├── TTC/                               # TTC 代码仓
│   ├── loss.py                        # SupConLoss (复用)
│   ├── data/
│   │   └── audio_dataset.py           # [新] WAV 加载 + 波形增强
│   └── models/
│       └── beats_cont.py              # [新] BEATs + 投影头 + SupMin
├── prepare_audio_data.py              # 数据准备: 提取 WAV 片段 + 约束增强
├── data_changyin_audio/               # 生成的数据
│   ├── train/
│   │   ├── class_0/  (n, ~4000 WAV)
│   │   └── class_1/  (rare=g+h, ~1800 WAV)
│   └── val/
│       ├── class_0/  (n, ~1400 WAV)
│       └── class_1/  (rare, ~100 WAV)
├── train_beats.py                     # [新] 独立训练脚本
├── run_train_beats.bat                # Windows 一键启动
└── README.md
```

## 环境

```bash
# Python 3.8+
pip install torch torchvision lightning timm torchmetrics tqdm pillow scikit-learn
pip install transformers              # BEATs 加载
pip install scipy soundfile
```

## 启动训练

### 第1步：生成数据

```bash
cd d:/code/TTC_changyin
python prepare_audio_data.py
```

这会从 `../MM_class/data_changyin/` 提取所有标注事件，保存为 16kHz WAV，对 h 做约束增强（质心/带宽/RMS/过零率均在 h 物理范围内）。

输出：`data_changyin_audio/train/` 和 `data_changyin_audio/val/`

### 第2步：启动训练

**Windows (双击)**：`run_train_beats.bat`

**命令行**：

```bash
python train_beats.py \
    --train_dir data_changyin_audio/train \
    --val_dir data_changyin_audio/val \
    --beats_path microsoft/beats-iter3 \
    --finetune_layers 0 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --temperature 0.07 \
    --ratio_sup_majority 0.0 \
    --accelerator auto \
    --name changyin_beats_v1
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--beats_path` | `microsoft/beats-iter3` | HuggingFace ID 或本地路径如 `D:/models/BEATs_iter3` |
| `--finetune_layers` | `0` | 0=冻结BEATs, 2=解冻最后2层, -1=全部可训练 |
| `--ratio_sup_majority` | `0.0` | **核心**: 0=SupMin(只对rare对比), -1=标准SupCon |
| `--audio_length` | `48000` | 固定输入长度(samples), 3s@16kHz |
| `--temperature` | `0.07` | 对比学习温度 |
| `--lr` | `0.001` | AdamW 学习率 |

### 三种模式

```bash
# SupMin (推荐, rare类对比学习)
--ratio_sup_majority 0.0

# 标准 SupCon (全量对比)
--ratio_sup_majority -1.0

# 部分监督
--ratio_sup_majority 0.5
```

## 训练监控

- **主指标**: `online_val_acc` — 在线分类器在验证集上的准确率
- **辅助指标**: `val.minority_recall` — rare 类召回率
- **对比损失**: `train.loss` — 对比损失 (SupMin 主要看少数类部分)

日志保存在 `logs/changyin_beats_v1/`，检查点文件会自动保存 best 3 个。

## 推理

训练完成后，推理流程：

```python
import torch
from train_beats import BEATsContrastive

model = BEATsContrastive.load_from_checkpoint('logs/.../best-xxx.ckpt')
model.eval()

# 输入: 1段音频, 16kHz, 任意长度
audio = torch.from_numpy(your_audio).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
encoding, projection = model(audio)

# encoding: [1, 768] — 用于下游任务的通用特征
# 在线分类器预测
logits = model.online_evaluator(encoding)  # [1, 2]
pred = logits.argmax(1).item()  # 0=n, 1=rare
```

## Stage 2: g vs h

TTC/BEATs 做完 n vs rare 后，对判为 rare 的样本提取物理特征（centroid, bandwidth, RMS, ZCR, lo_energy_ratio），用简单的 SVM 或规则分类 g vs h。物理特征分析显示这两类在 RMS 上有 4.4 倍的差异，几乎线性可分。
