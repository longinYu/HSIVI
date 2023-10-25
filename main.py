import logging
import os
import configargparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_utils.utils import make_dir

def run_main(config):
    
    config.global_size = config.n_nodes * config.n_gpus_per_node
    processes = []
    for rank in range(config.n_gpus_per_node):
        config.local_rank = rank
        config.global_rank = rank + config.node_rank * config.n_gpus_per_node
        print('Node rank %d, local proc %d, global proc %d' %
              (config.node_rank, config.local_rank, config.global_rank))

        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.master_address
    os.environ['MASTER_PORT'] = '%d' % config.master_port
    torch.cuda.set_device(config.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.global_rank,
                            world_size=config.global_size)
    fn(config)
    dist.barrier()
    dist.destroy_process_group()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(config):
    if config.workdir[-1] == '/':
        config.workdir = config.workdir[:-1]
    
    workdir = os.path.join(config.root, config.workdir)

    if config.mode == 'train':
        if config.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)

        import run_lib_speedup
        run_lib_speedup.train(config, workdir)

    elif config.mode == 'eval':
        if config.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
        import run_lib_speedup
        run_lib_speedup.evaluate(config, workdir)

    elif config.mode == 'continue':
        if os.path.exists(workdir):
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')
            set_logger(gfile_stream)

            import run_lib_speedup
            run_lib_speedup.train(config, workdir)

    else:
        raise ValueError('Mode not recognized.')

if __name__ == '__main__':
    p = configargparse.ArgParser()
    p.add('-cc', is_config_file=True)
    p.add('-sc', is_config_file=True)

    p.add('--root')
    p.add('--workdir', required=True)
    p.add('--eval_folder', default=None)
    p.add('--mode', choices=['train', 'eval', 'continue'], required=True)
    p.add('--cont_nbr', type=int, default=None)
    p.add('--checkpoint', default=None)
    p.add('--n_gpus_per_node', type=int, default=1)
    p.add('--n_nodes', type=int, default=1)
    p.add('--node_rank', type=int, default=0)
    p.add('--master_address', default='127.0.0.1')
    p.add('--master_port', type=int, default=6020)
    p.add('--distributed', action='store_false')
    p.add('--overwrite', action='store_true')

    p.add('--seed', type=int, default=0)
    p.add('--num_workers', type=int, default=16)

    # Data
    p.add('--dataset')
    p.add('--image_size', type=int)
    p.add('--center_image', action='store_true')
    p.add('--image_channels', type=int)
    p.add('--data_dim', type=int)  # Dimension of non-image data
    p.add('--data_location', default=None)

    # SDE
    p.add('--gt_steps', type=int, default=1000)
    p.add('--skip_type', type=str, default='uniform')

    # Optimization
    p.add('--optimizer')
    p.add('--phi_learning_rate', type=float)
    p.add('--f_learning_rate', type=float)
    p.add('--weight_decay', type=float)
    p.add('--grad_clip', type=float)
    p.add('--update_phi_step', type=int)

    # Objective
    p.add('--f_learning_times', type=int, default=1)
    p.add('--independent_log_gamma', type=str, default='dis') 
    p.add('--eta', type=float, default=0) 

    # Model
    p.add('--name')
    
    # Training
    p.add('--training_batch_size', type=int)
    p.add('--testing_batch_size', type=int)
    p.add('--sampling_batch_size', type=int)
    p.add('--ema_rate', type=float)
    p.add('--n_train_iters', type=int)
    p.add('--n_warmup_iters', type=int)
    p.add('--snapshot_freq', type=int)
    p.add('--log_freq', type=int)
    p.add('--eval_freq', type=int)
    p.add('--fid_freq', type=int)
    p.add('--eval_threshold', type=int, default=1)
    p.add('--snapshot_threshold', type=int, default=1)
    p.add('--fid_threshold', type=int, default=1)
    p.add('--fid_samples_training', type=int)
    p.add('--n_eval_batches', type=int)
    p.add('--autocast_train', action='store_true')
    p.add('--save_freq', type=int, default=None)
    p.add('--save_threshold', type=int, default=1)
    p.add('--pretrained_model', type=str)
    p.add('--image_gamma', type=str, default='dis')
    p.add('--cosine_lr_decay', type=str, default='dis')
    # Sampling
    p.add('--n_discrete_steps', type=int)
    p.add('--ref_statistics', type=str)
    # Evaluation
    p.add('--ckpt_file')
    p.add('--eval_fid', action='store_true')
    p.add('--eval_fid_samples', type=int, default=50000)
    p.add('--eval_seed', type=int, default=0)
    
    
    config = p.parse_args()
    run_main(config)