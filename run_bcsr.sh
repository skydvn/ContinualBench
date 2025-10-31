#cifar-100
python3.11 main.py --model bcsr --dataset seq-cifar100 --lr 0.001 --buffer_size 200   --bcsr_lr_proxy 0.001 --bcsr_beta 1.0 --bcsr_outer_it 5 --bcsr_inner_it 1   --bcsr_weight_lr 0.001 --bcsr_candidate_bs 600 --optimizer sgd --n_epochs 200 --batch_size 256

#cifar-10
python3.11 main.py --model bcsr --dataset seq-cifar10 --lr 0.001 --buffer_size 200   --bcsr_lr_proxy 0.001 --bcsr_beta 1.0 --bcsr_outer_it 5 --bcsr_inner_it 1   --bcsr_weight_lr 0.001 --bcsr_candidate_bs 600 --optimizer sgd --n_epochs 200 --batch_size 256

