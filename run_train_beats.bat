@echo off
cd /d %~dp0

REM ============================================
REM  BEATs + ABCD 改进 — 肠音二分类
REM  A: Attention Pooling
REM  B: 物理特征 concat
REM  C: BEATs 最后2层微调
REM  D: 物理特征预测 MSE
REM ============================================

set ACCEL=auto
set BEATS_CODE=./BEATs
set BEATS_CKPT=../model/BEATs_iter3.pt

echo ============================================
echo  BEATs + ABCD — 肠音二分类训练
echo  A: Attention Pooling (4 queries)
echo  B: 物理特征注入 (7-dim)
echo  C: BEATs 微调 (最后2层, lr=1e-5)
echo  D: 物理特征预测 (weight=0.1)
echo ============================================

python train_beats.py ^
    --train_dir data_changyin_audio/train ^
    --val_dir data_changyin_audio/val ^
    --beats_code_path %BEATS_CODE% ^
    --beats_ckpt_path %BEATS_CKPT% ^
    --finetune_layers 2 ^
    --beats_lr 1e-5 ^
    --attn_queries 4 ^
    --aux_weight 0.1 ^
    --epochs 100 ^
    --batch_size 64 ^
    --lr 0.001 ^
    --temperature 0.07 ^
    --ratio_sup_majority 0.0 ^
    --accelerator %ACCEL% ^
    --name changyin_beats_abcd_v1

echo.
echo Done. Logs in logs/changyin_beats_abcd_v1/
pause
