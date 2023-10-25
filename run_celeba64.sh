#! /bin/bash

## Celeba, these experiments run on 8 A100 GPUs.

python main.py \
    -cc configs/default_celeba.txt \
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
    --cosine_lr_decay use \
    --phi_learning_rate 0.00001 \
    --f_learning_rate 0.00005 \
    --n_discrete_steps 11 \
    --n_train_iters 200000 \
    --workdir ./work_dir/celeba_10steps\
    --master_address 127.0.0.10 \
    --master_port 6987


python main.py \
    -cc configs/default_celeba.txt \
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
    --cosine_lr_decay use \
    --phi_learning_rate 0.00001 \
    --f_learning_rate 0.00005 \
    --n_discrete_steps 16 \
    --n_train_iters 200000 \
    --workdir ./work_dir/celeba_15steps\
    --master_address 127.0.0.10 \
    --master_port 6987


#! Note that we adopt the former 10steps weights (best one for 5steps) as the pretrained weights for 5 steps model. 
python main.py \
    -cc configs/default_celeba.txt \
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
    --cosine_lr_decay use \
    --phi_learning_rate 0.000001 \
    --f_learning_rate 0.000005 \
    --n_discrete_steps 6 \
    --pretrained_model ./pretrained_model/checkpoint_40000.pth\
    --n_train_iters 200000 \
    --workdir ./work_dir/celeba_5steps\
    --master_address 127.0.0.10 \
    --master_port 6987
