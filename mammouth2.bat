@echo off
REM ====================================================
REM  Run Sequential Experiments for seq-tinyimg & seq-cifar100
REM ====================================================

REM ---------- seq-cifar100: bcsr ----------
python main.py ^
    --dataset seq-cifar100 ^
    --model bcsr ^
    --buffer_size 200 ^
    --minibatch_size 32 ^
    --lr 0.001 ^
    --n_epochs 400 ^
    --batch_size 256 ^
    --device cuda:0 ^
    --seed 0 ^
    --backbone resnet18 ^
    --savecheck last ^
    --wandb_on True

REM ---------- seq-tinyimg: bcsr ----------
python main.py ^
    --dataset seq-tinyimg ^
    --model bcsr ^
    --buffer_size 200 ^
    --minibatch_size 32 ^
    --lr 0.001 ^
    --n_epochs 400 ^
    --batch_size 256 ^
    --device cuda:0 ^
    --seed 0 ^
    --backbone resnet18 ^
    --savecheck last ^
    --wandb_on True


@REM REM ---------- seq-tinyimg: DER ----------
@REM python main.py ^
@REM     --dataset seq-tinyimg ^
@REM     --model der ^
@REM     --buffer_size 200 ^
@REM     --minibatch_size 32 ^
@REM     --lr 0.001 ^
@REM     --n_epochs 400 ^
@REM     --batch_size 256 ^
@REM     --device cuda:0 ^
@REM     --seed 0 ^
@REM     --backbone resnet18 ^
@REM     --validation 0 ^
@REM     --alpha 0.5 ^
@REM     --savecheck last ^
@REM     --wandb_on True

@REM REM ---------- seq-cifar100: DER ----------
@REM python main.py ^
@REM     --dataset seq-cifar100 ^
@REM     --model der ^
@REM     --buffer_size 200 ^
@REM     --minibatch_size 32 ^
@REM     --lr 0.001 ^
@REM     --n_epochs 400 ^
@REM     --batch_size 256 ^
@REM     --device cuda:0 ^
@REM     --seed 0 ^
@REM     --backbone resnet18 ^
@REM     --validation 0 ^
@REM     --alpha 0.5 ^
@REM     --savecheck last ^
@REM     --wandb_on True


echo.
echo ====================================================
echo All experiments completed.
echo ====================================================
pause
