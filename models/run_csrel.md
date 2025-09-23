#!/bin/bash
python3.11 main.py \
    --dataset seq-cifar10 \
    --model csrel \
    --buffer_size 100 \
    --minibatch_size 512 \
    --lr 0.1 \
    --n_epochs 10 \
    --csrel_ref_epochs 40 \
    --csrel_ref_lr 0.1 \
    --csrel_ce_factor 1.0 \
    --csrel_mse_factor 0.0 \
    --csrel_batch_size 256 \
    --csrel_selection_steps 100 \
    --device cuda:0 \
    --seed 0

