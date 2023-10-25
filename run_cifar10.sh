#! /bin/bash

## CIFAR10 , these experiments run on 8 2080ti GPUs.

python main.py \
    -cc configs/default_cifar.txt \
    --root './' \
    --mode train \
    --n_gpus_per_node 8 \
    --training_batch_size 16  \
    --testing_batch_size 16 \
    --sampling_batch_size 64 \
    --independent_log_gamma dis \
    --f_learning_times 20 \
    --image_gamma use \
    --skip_type quad \
    --n_discrete_steps 11 \
    --phi_learning_rate 0.000016 \
    --f_learning_rate 0.00008 \
    --n_train_iters 200000 \
    --pretrained_model ./pretrained_model/target_epsilon_cifar10.pt \
    --workdir ./work_dir/cifar10_10steps \
    --master_address 127.0.0.10 \
    --master_port 4372


python main.py \
    -cc configs/default_cifar.txt \
    --root './' \
    --mode train \
    --n_gpus_per_node 8 \
    --training_batch_size 16 \
    --testing_batch_size 32 \
    --sampling_batch_size 64 \
    --independent_log_gamma dis \
    --f_learning_times 20 \
    --image_gamma use \
    --eta 0.2 \
    --skip_type quad \
    --phi_learning_rate 0.000016 \
    --f_learning_rate 0.00008 \
    --n_discrete_steps 16 \
    --n_train_iters 200000 \
    --num_accum 1\
    --pretrained_model ./pretrained_model/target_epsilon_cifar10.pt \
    --workdir ./work_dir/cifar_15steps\
    --master_address 127.0.0.10 \
    --master_port 6987


# #! Note that we adopt the former 10steps weights (best one for 5steps) as the pretrained weights for 5 steps model. 
python main.py \
    -cc configs/default_cifar.txt \
    --root './' \
    --mode train \
    --n_gpus_per_node 8 \
    --training_batch_size 16 \
    --testing_batch_size 32 \
    --sampling_batch_size 64 \
    --independent_log_gamma dis \
    --f_learning_times 20 \
    --image_gamma use \
    --eta 0.2 \
    --skip_type quad \
    --phi_learning_rate 0.0000016 \
    --f_learning_rate 0.0000008 \
    --n_discrete_steps 6 \
    --pretrained_model ./pretrained_model/cifar_10steps.pth\
    --n_train_iters 200000 \
    --num_accum 1\
    --workdir ./work_dir/cifar_5steps\
    --master_address 127.0.0.10 \
    --master_port 6987
