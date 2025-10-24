@echo off
REM ====================================================
REM  Run Sequential Experiments for seq-tinyimg & seq-cifar100
REM ====================================================

REM ---------- seq-tinyimg: iCaRL ----------
python main.py ^
    --dataset seq-tinyimg ^
    --model icarl ^
    --buffer_size 200 ^
    --minibatch_size 32 ^
    --lr 0.001 ^
    --n_epochs 400 ^
    --batch_size 256 ^
    --device cuda:0 ^
    --seed 0 ^
    --backbone resnet18 ^
    --validation 0 ^
    --savecheck last ^
    --wandb_on True

REM ---------- seq-cifar100: iCaRL ----------
python main.py ^
    --dataset seq-cifar100 ^
    --model icarl ^
    --buffer_size 200 ^
    --minibatch_size 32 ^
    --lr 0.001 ^
    --n_epochs 400 ^
    --batch_size 256 ^
    --device cuda:0 ^
    --seed 0 ^
    --backbone resnet18 ^
    --validation 0 ^
    --savecheck last ^
    --wandb_on True

echo.
echo ====================================================
echo All experiments completed.
echo ====================================================
pause
