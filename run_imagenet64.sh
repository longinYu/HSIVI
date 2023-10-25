#! /bin/bash

## ImageNet, running on 8*8 80G A100 GPUs.

#! for 10 steps model training.
python main.py \
   -cc configs/default_imagenet.txt \
   --root './' \
   --mode train \
   --n_gpus_per_node 8 \
   --training_batch_size 32 \
   --sampling_batch_size 64 \
   --independent_log_gamma dis \
   --f_learning_times 20 \
   --image_gamma use \
   --eta 0.2 \
   --cosine_lr_decay use \
   --phi_learning_rate 0.00001 \
   --f_learning_rate 0.00005 \
   --skip_type uniform \
   --n_discrete_steps 11 \
   --n_train_iters 200000 \
   --workdir ./work_dir/imagenet_10steps\
   --master_address 127.0.0.10 \
   --master_port 6987

#! for 15 steps model training.
python main.py \
    -cc configs/default_imagenet.txt \
    --root './' \
    --mode train \
    --n_gpus_per_node 8 \
    --training_batch_size 32 \
    --sampling_batch_size 64 \
    --independent_log_gamma dis \
    --f_learning_times 20 \
    --image_gamma use \
    --eta 0.2 \
    --cosine_lr_decay use \
    --phi_learning_rate 0.00001 \
    --f_learning_rate 0.00005 \
    --skip_type uniform \
    --n_discrete_steps 16 \
    --n_train_iters 200000 \
    --workdir ./work_dir/imagenet_15steps\
    --master_address 127.0.0.10 \
    --master_port 6987

#! for 5 steps model training.
python main.py\
    -cc configs/default_imagenet.txt \
    --root './' \
    --mode train \
    --n_gpus_per_node 8 \
    --training_batch_size 32 \
    --sampling_batch_size 64 \
    --independent_log_gamma dis \
    --f_learning_times 20 \
    --image_gamma use \
    --eta 0.2 \
    --cosine_lr_decay use \
    --phi_learning_rate 0.000001 \
    --f_learning_rate 0.000005 \
    --skip_type uniform \
    --n_discrete_steps 6 \
    --n_train_iters 200000 \
    --pretrained_model ./pretrained_model/init_weights_for_5steps.pth\
    --workdir ./work_dir/imagenet_5steps\
    --master_address 127.0.0.10 \
    --master_port 6987  
