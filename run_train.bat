@echo off
cd /d %~dp0TTC

REM ============================================
REM  TTC SupMin 对比学习 — 肠音二分类训练
REM  Stage 1: n (class_0) vs rare=g+h (class_1)
REM ============================================

REM ---- 硬件选择 ----
REM 有GPU则设为 gpu, 无GPU设为 cpu
set ACCEL=cpu

REM ---- 训练参数 ----
set RUN_NAME=changyin_supmin_v1
set BATCH_SIZE=64
set LR=0.01
set EPOCHS=100
set TEMP=0.07

echo ============================================
echo  肠音二分类 SupMin 对比学习
echo  n (class_0, N=4000) vs rare (class_1, N=1831)
echo  加速器: %ACCEL%, Batch=%BATCH_SIZE%, LR=%LR%
echo ============================================

python train.py ^
    data=changyin_binary ^
    experiment=changyin_contrastive ^
    experiment/specs=changyin_binary ^
    module.ratio_supervised_majority=0.0 ^
    module.image_net_init=True ^
    module.lr=%LR% ^
    module.temperature=%TEMP% ^
    module.warmup_epochs=10 ^
    trainer.max_epochs=%EPOCHS% ^
    trainer.accelerator=%ACCEL% ^
    trainer.devices=1 ^
    batch_size=%BATCH_SIZE% ^
    seed=42 ^
    name="%RUN_NAME%" ^
    logger=csv ^
    tags="[changyin,supmin,binary]"

REM 如果上述命令因 logger=csv 报错, 改回默认 logger:
REM python train.py ... logger=wandb  (需要先装 wandb 并登录)

echo.
echo Done. 日志在 logs/%RUN_NAME%/
pause
