#!/bin/bash


# Run OCS with mammoth
python3.11 main.py \
    --dataset seq-cifar100 \
    --model ocs \
    --buffer_size 500 \
    --minibatch_size 32 \
    --lr 0.1 \
    --n_epochs 1 \
    --ocs_tau 1000 \
    --ocs_ref_hyp 0.5 \
    --ocs_batch_size 10 \
    --ocs_r2c_iter 100 \
    --ocs_is_r2c \
    --ocs_select_type ocs_select \
    --ocs_grad_batch_size 32 \
    --device cuda:0 \
   

