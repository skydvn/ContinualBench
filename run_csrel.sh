#!/bin/bash
python3.11 main.py     --dataset seq-cifar10     --model csrel     --buffer_size 200     --minibatch_size 32     --lr 0.001     --n_epochs 100     --batch_size 128     --device cuda:0     --seed 0     --backbone resnet18     --validation 0     --csrel_ref_epochs 15     --csrel_ref_lr 0.003     --csrel_ce_factor 1.0     --csrel_mse_factor 0.0     --csrel_batch_size 128     --csrel_selection_steps 200  --csrel_buffer_path ./csrel_buffer_seq-cifar10_200


python3.11 main.py     --dataset seq-cifar100     --model csrel     --buffer_size 200     --minibatch_size 32     --lr 0.001     --n_epochs 100     --batch_size 128     --device cuda:0     --seed 0     --backbone resnet18     --validation 0     --csrel_ref_epochs 15     --csrel_ref_lr 0.003     --csrel_ce_factor 1.0     --csrel_mse_factor 0.0     --csrel_batch_size 128     --csrel_selection_steps 200  --csrel_buffer_path ./csrel_buffer_seq-cifar10_200


python3.11 main.py     --dataset seq-cifar10     --model csrel     --buffer_size 400     --minibatch_size 32     --lr 0.001     --n_epochs 100     --batch_size 128     --device cuda:0     --seed 0     --backbone resnet18     --validation 0     --csrel_ref_epochs 15     --csrel_ref_lr 0.003     --csrel_ce_factor 1.0     --csrel_mse_factor 0.0     --csrel_batch_size 128     --csrel_selection_steps 400  --csrel_buffer_path ./csrel_buffer_seq-cifar10_200


python3.11 main.py     --dataset seq-cifar100     --model csrel     --buffer_size 400     --minibatch_size 32     --lr 0.001     --n_epochs 100     --batch_size 128     --device cuda:0     --seed 0     --backbone resnet18     --validation 0     --csrel_ref_epochs 15     --csrel_ref_lr 0.003     --csrel_ce_factor 1.0     --csrel_mse_factor 0.0     --csrel_batch_size 128     --csrel_selection_steps 400  --csrel_buffer_path ./csrel_buffer_seq-cifar10_200
