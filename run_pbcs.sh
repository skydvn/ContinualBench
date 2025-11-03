#cifar-100
CUDA_VISIBLE_DEVICES=0 python3.11 main.py     --model pbcs     --dataset seq-cifar100     --buffer_size 200     --lr 0.001     --batch_size 256     --n_epochs 30     --pbcs_K 5     --pbcs_outer_lr 0.001     --pbcs_inner_lr 0.001     --pbcs_max_outer_iter 50     --pbcs_epoch_converge 50     --pbcs_use_vr     --pbcs_clip_grad     --pbcs_coreset_ratio 0.1

#cifar-10
python3.11 main.py     --model pbcs     --dataset seq-cifar10     --buffer_size 200     --lr 0.001     --batch_size 256     --n_epochs 30     --pbcs_K 5     --pbcs_outer_lr 0.001     --pbcs_inner_lr 0.001     --pbcs_max_outer_iter 50     --pbcs_epoch_converge 50     --pbcs_use_vr     --pbcs_clip_grad     --pbcs_coreset_ratio 0.1
