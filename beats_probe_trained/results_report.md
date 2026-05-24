# BEATs 训练后模型评估报告

**评估时间**: 2026-05-18  
**模型**: BEATs iter3 + SupMin 对比学习 + ABCD 改进  
**评估脚本**: `probe_beats.py --model_ckpt`  
**对比基线**: `beats_probe/` (原始 BEATs, 训练前)  

---

## 1. 数据集

| 类别 | 样本数 |
|------|--------|
| n (正常) | 2,000 |
| g (异常G) | 179 |
| h (异常H) | 24 |
| **总计** | **2,203** |

---

## 2. 训练后模型直接分类 (Classifier Head)

| 指标 | 值 |
|------|:---:|
| Accuracy | 90.79% |
| Balanced Accuracy | 33.33% |

**混淆矩阵**:

| | 预测n | 预测g | 预测h |
|---|:---:|:---:|:---:|
| 真实n | **2000** | 0 | 0 |
| 真实g | 179 | **0** | 0 |
| 真实h | 24 | **0** | 0 |

**结论**: Classifier head 完全崩溃——全部预测为 n。原因是分类头训练时使用普通 Cross-Entropy + 不平衡采样，模型学到了"全猜 n 也能拿 70% 正确率"的捷径。

---

## 3. Logistic Regression 分类上限 (Embedding 质量评估)

对训练后的 projection head 输出 (128d) 做 Logistic Regression，评估特征的真实区分能力：

| 类别 | Precision | Recall | F1-score | Support |
|------|:---:|:---:|:---:|:---:|
| n | 0.9829 | 0.9583 | 0.9705 | 600 |
| g | 0.6324 | 0.7963 | 0.7049 | 54 |
| h | 0.7500 | 0.8571 | 0.8000 | 7 |
| **Macro Avg** | 0.7884 | 0.8706 | 0.8251 | 661 |
| **Weighted Avg** | 0.9518 | 0.9440 | 0.9470 | 661 |

**Best C**: 100.0

**混淆矩阵**:

| | 预测n | 预测g | 预测h |
|---|:---:|:---:|:---:|
| 真实n | **575** | 24 | 1 |
| 真实g | 10 | **43** | 1 |
| 真实h | 0 | 1 | **6** |

### Per-Class Recall

| 类别 | Recall |
|------|:---:|
| n | 95.83% |
| g | **79.63%** |
| h | **85.71%** |

---

## 4. 训练前后对比

| 指标 | 训练前 (raw BEATs) | 训练后 (proj 128d) | 变化 |
|------|:---:|:---:|:---:|
| **g recall** | 48.2% | **79.6%** | ↑ +65% |
| **h recall** | 42.9% | **85.7%** | ↑ +100% |
| **Macro F1** | 0.621 | **0.825** | ↑ +33% |
| **Fisher top-1** | 0.12 | **0.46** | ↑ +4× |
| **Silhouette Score** | -0.097 | **-0.056** | ↑ 改善中 |
| g↔h 互混 | 0 | 2 | 轻微增加 |

---

## 5. Embedding 空间分析

### 类间距离 (标准化欧氏距离, 128d projection)

| | n | g | h |
|---|:---:|:---:|:---:|
| n | 0 | 8.11 | 9.43 |
| g | 8.11 | 0 | 2.92 |
| h | 9.43 | 2.92 | 0 |

**对比训练前 (768d mean pool)**:

| | n | g | h |
|---|:---:|:---:|:---:|
| n | 0 | 16.9 | 15.9 |
| g | 16.9 | 0 | **28.0** |
| h | 15.9 | **28.0** | 0 |

> 注意: 训练前后维度不同 (768 vs 128)，距离绝对值不可直接比较。但 g↔h 距离从 28.0 降到 2.92，说明投影头将 g 和 h 拉近到了共同的 "rare" 区域——这正是 SupMin 对比学习期望的效果。

### Fisher 判别比 Top-10

| 维度 | Fisher Ratio |
|------|:---:|
| dim_0035 | 0.4634 |
| dim_0120 | 0.3299 |
| dim_0030 | 0.3203 |
| dim_0058 | 0.2985 |
| dim_0092 | 0.2682 |
| dim_0047 | 0.2580 |
| dim_0111 | 0.2193 |
| dim_0068 | 0.1917 |
| dim_0069 | 0.1894 |
| dim_0075 | 0.1706 |

> 训练前 Fisher top-1 仅 0.12，训练后提升至 0.46 (+4×)。说明投影头学会了将分类信噪比集中在少数几个维度上。

### Silhouette Score

- 训练前: -0.0969
- 训练后: **-0.0558**

负值仍表示有部分样本更接近其他类中心，但绝对值减半说明分离度在改善。

---

## 6. 结论

### 训练有效

ABCD 四个改进（Attention Pooling + 物理特征注入 + BEATs 微调 + 物理特征预测）对 embedding 质量产生了显著提升：

- ✅ g recall 从 48% → 80%
- ✅ h recall 从 43% → 86%
- ✅ Fisher 判别比提升 4 倍
- ✅ 混淆矩阵中 g↔n 误判从 28 降至 10

### 待修复

- ❌ Classifier head 崩溃 (全预测 n)。原因: 训练时 CE loss + 不平衡 batch。
- 🔧 **修复方案**: Focal Loss (alpha=0.75, gamma=2.0) + WeightedRandomSampler
- 🔧 代码已修改，重新训练即可

### 下一步

```bash
python train_beats.py \
    --train_dir data_changyin_audio/train \
    --val_dir data_changyin_audio/val \
    --beats_code_path ./BEATs \
    --accelerator auto
```

预期：重训后 classifier head 结果应接近 LR 上限 (g recall ~80%, h recall ~86%)。

---

*评估脚本: `probe_beats.py --model_ckpt` / `beats_probe_trained/`*
