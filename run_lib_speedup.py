import os
import time
import logging
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from models import ddpm_cifar, ddpm_celeba, ddpm_imagenet

import tqdm
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from torch_utils.utils import make_dir, get_optimizer_speedup, optimization_manager_speedup, set_seeds
from torch_utils.utils import broadcast_params, reduce_tensor, get_data_inverse_scaler, save_img
from torch_utils.checkpoint import save_checkpoint, restore_checkpoint
from torch_utils.losses import  get_step_fn_speedup
from torch_utils.sampling import get_sampling_fn
from torch_utils.diffusion import Diffusion
import torch_utils.evaluation as evaluation
import PIL.Image

def train(config, workdir):
    ''' Main training script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    global_size = config.global_size
    if config.mode == 'train':
        set_seeds(global_rank, config.seed)
    elif config.mode == 'continue':
        set_seeds(global_rank, config.seed)# + config.cont_nbr
    else:
        raise NotImplementedError('Mode %s is unknown.' % config.mode)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    # Setting up all necessary folders
    sample_dir = os.path.join(workdir, 'samples')
    tb_dir = os.path.join(workdir, 'tensorboard')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    if config.image_channels == 3:
        fid_dir = os.path.join(workdir, 'fid')
    if global_rank == 0:
        logging.info(config)
        if config.mode == 'train':
            make_dir(sample_dir)
            make_dir(tb_dir)
            make_dir(checkpoint_dir)
            if config.image_channels == 3:
                make_dir(fid_dir)
        writer = tensorboard.SummaryWriter(tb_dir)
    dist.barrier()

    diffusion = Diffusion(config)

    # # Creating the score model
    whole_model = mutils.create_model(config).to(config.device)
    if config.n_discrete_steps > 6:
        pretrained_score = torch.load(config.pretrained_model, map_location=torch.device('cpu')) # load target score model
        whole_model.target_model.load_state_dict(pretrained_score, strict=True)
        whole_model.phinet.load_state_dict(pretrained_score, strict=False)
        whole_model.fnet.load_state_dict(pretrained_score, strict=True)
    broadcast_params(whole_model.parameters())  # Sync all parameters
    whole_model = DDP(whole_model, device_ids=[local_rank], find_unused_parameters=True)
    if config.n_discrete_steps ==6:
        whole_model.load_state_dict(torch.load(config.pretrained_model, map_location=torch.device('cpu'))['whole_model'])
    
    ema_phinet = ExponentialMovingAverage(
        whole_model.module.phinet.parameters(), decay=config.ema_rate)
    ema_fnet = ExponentialMovingAverage(
        whole_model.module.fnet.parameters(), decay=config.ema_rate)

    if global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, whole_model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    dist.barrier()
    
    inverse_scaler = get_data_inverse_scaler()

    optim_phinet_params = whole_model.module.phinet.parameters() 
    optimizer_phinet = get_optimizer_speedup(config, optim_phinet_params)
    optim_fnet_params = whole_model.module.fnet.parameters()
    optimizer_fnet = get_optimizer_speedup(config, optim_fnet_params)
    state = dict(optimizer_phinet=optimizer_phinet, optimizer_fnet=optimizer_fnet, 
                  whole_model=whole_model, ema_phinet=ema_phinet, ema_fnet=ema_fnet, step=1)

    if config.mode == 'continue':
        if config.checkpoint is None:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        else:
            ckpt_path = os.path.join(checkpoint_dir, config.checkpoint)
        if global_rank == 0:
            logging.info('Loading model from path: %s' % ckpt_path)
        dist.barrier()
        
        state = restore_checkpoint(ckpt_path, state, device=config.device)

    num_total_iter = config.n_train_iters

    if global_rank == 0:
        logging.info('Number of total iterations: %d' % num_total_iter)
    dist.barrier()
    
    phi_optimize_fn = optimization_manager_speedup(config, phi_net = True)
    f_optimize_fn = optimization_manager_speedup(config, phi_net = False)
    train_step_fn = get_step_fn_speedup(True, phi_optimize_fn, f_optimize_fn, diffusion, config)

    training_shape = (config.training_batch_size,
                    config.image_channels,
                    config.image_size,
                    config.image_size
    )
    sampling_shape = (config.sampling_batch_size,
                    config.image_channels,
                    config.image_size,
                    config.image_size)

    sampling_fn = get_sampling_fn(
        config, diffusion, sampling_shape)
    

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Starting training at step %d' % step)
    dist.barrier()

    if config.mode == 'continue':
        config.eval_threshold = max(step + 1, config.eval_threshold)
        config.snapshot_threshold = max(step + 1, config.snapshot_threshold)
        
        config.save_threshold = max(step + 1, config.save_threshold)

    while step < num_total_iter+1:
        
        start_time = time.time()
        
        # for weight saving.
        if step % config.save_freq == 0:
            if global_rank == 0:
                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % step)
                if not os.path.isfile(checkpoint_file):
                    save_checkpoint(checkpoint_file, state)
            dist.barrier()

        # for visualization.
        if (step % config.snapshot_freq == 0) and global_rank == 0 and step >= config.snapshot_threshold or step==1:
            logging.info('Saving snapshot checkpoint.')
            save_checkpoint(os.path.join(
                checkpoint_dir, 'snapshot_checkpoint.pth'), state)

            ema_phinet.store(whole_model.module.phinet.parameters())
            ema_phinet.copy_to(whole_model.module.phinet.parameters())
            x = sampling_fn(whole_model)
            ema_phinet.restore(whole_model.module.phinet.parameters()) 

            this_sample_dir = os.path.join(sample_dir, 'iter_%d' % step)
            os.makedirs(this_sample_dir, exist_ok=True)
            x = inverse_scaler(x)
            save_img(x, os.path.join(
                this_sample_dir, 'sample.png'))
        dist.barrier()
        
        # for FID evaluation
        if (step % config.fid_freq == 0) and config.eval_fid:
            logging.info('FID evaluating with %d samples.'%config.eval_fid_samples)
            this_sample_dir = os.path.join(fid_dir, 'step_%d' % step)
            if global_rank == 0:
                make_dir(this_sample_dir)
            dist.barrier()

            ema_phinet.store(whole_model.module.phinet.parameters())
            ema_phinet.copy_to(whole_model.module.phinet.parameters())

            seeds = np.arange(0, config.eval_fid_samples, 1)
            num_batches = ((len(seeds) - 1) // (config.sampling_batch_size * evaluation.get_world_size()) + 1) * evaluation.get_world_size()
            all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
            rank_batches = all_batches[evaluation.get_rank() :: evaluation.get_world_size()]

            for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(evaluation.get_rank() != 0)):
                
                dist.barrier()
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                x = sampling_fn(whole_model)
                x = inverse_scaler(x)

                samples = np.clip(x.permute(0, 2, 3, 1).cpu().numpy()
                                * 255., 0, 255).astype(np.uint8)
                
                for seed, image_np in zip(batch_seeds, samples):
                    image_dir = os.path.join(this_sample_dir, f'{seed-seed%1000:06d}') 
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{seed:06d}.png')                        
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np, 'L').save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            dist.barrier()
            ema_phinet.restore(whole_model.module.phinet.parameters())

            if global_rank == 0:
                mu, sigma = evaluation.calculate_inception_stats(this_sample_dir)
                ref = dict(np.load(config.ref_statistics))
                fid = evaluation.calculate_fid_from_inception_stats(
                    mu, sigma, ref['mu'], ref['sigma'])
                logging.info('\tFID: %.6f' % fid)
            torch.cuda.empty_cache()
            dist.barrier()
        
        # for training.          
        loss_phi, loss_fnet = train_step_fn(state,training_shape)
        
        # for logging.
        if step % config.log_freq == 0 and step >= 1:
            loss_phi = reduce_tensor(loss_phi, global_size)
            loss_fnet = reduce_tensor(loss_fnet, global_size)
            if global_rank == 0:
                logging.info('Iter %d/%d Loss_phiNet: %.4f Loss_FNet: %.4f Time: %.3f' % (step + 1,
                            config.n_train_iters, loss_phi.item(), loss_fnet.item(), time.time() - start_time))
                print('Iter %d/%d Loss_phiNet: %.4f Loss_FNet: %.4f Time: %.3f' % (step + 1,
                            config.n_train_iters, loss_phi.item(), loss_fnet.item(), time.time() - start_time))
                writer.add_scalar('training_phinet_loss', loss_phi, step)
                writer.add_scalar('training_fnet_loss', loss_fnet, step)
            dist.barrier()
            
        step += 1
        if step >= num_total_iter+1:
            break

    if global_rank == 0:
        logging.info('Finished after %d iterations.' % config.n_train_iters)
        logging.info('Saving final checkpoint.')
        save_checkpoint(os.path.join(
            checkpoint_dir, 'final_checkpoint.pth'), state)
    dist.barrier()


def evaluate(config, workdir):
    ''' Main evaluation script. '''
    
    local_rank = config.local_rank
    global_rank = config.global_rank
    set_seeds(global_rank, config.seed)
    
    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)


    sample_dir = os.path.join(workdir, 'samples')
    if config.image_channels == 3:
        fid_dir = os.path.join(workdir, 'fid')
    if global_rank == 0:
        logging.info(config)
        make_dir(sample_dir)
        if config.image_channels == 3:
            make_dir(fid_dir)
    dist.barrier()
    
    inverse_scaler = get_data_inverse_scaler()

    if config.diffusion == 'ddpm_alpha':
        diffusion = Diffusion(config)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.diffusion)
    
    if config.is_image:
        sampling_shape = (config.sampling_batch_size,
                        config.image_channels,
                        config.image_size,
                        config.image_size)
    else:
        sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = get_sampling_fn(
        config, diffusion, sampling_shape)

    # whole_model = mutils.create_model(config).to(config.device)
    # # Creating the score model
    whole_model = mutils.create_model(config).to(config.device)
    broadcast_params(whole_model.parameters())  # Sync all parameters
    whole_model = DDP(whole_model, device_ids=[local_rank],find_unused_parameters=False)

    state = torch.load(config.ckpt_file, map_location=config.device)
    if global_rank == 0:
        logging.info('Loading model from path: %s' % config.ckpt_file)
    dist.barrier()

    whole_model.load_state_dict(state['whole_model'], strict=True)

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Evaluating at training step %d' % step)
    dist.barrier()
    
    if config.eval_fid:
        logging.info('FID evaluating with %d samples.'%config.eval_fid_samples)
        this_sample_dir = os.path.join(fid_dir, 'step_%d' % step)
        if global_rank == 0:
            make_dir(this_sample_dir)
        dist.barrier()

        seeds = np.arange(0, config.eval_fid_samples, 1)
        num_batches = ((len(seeds) - 1) // (config.sampling_batch_size * evaluation.get_world_size()) + 1) * evaluation.get_world_size()
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[evaluation.get_rank() :: evaluation.get_world_size()]

        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(evaluation.get_rank() != 0)):
            
            dist.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            x = sampling_fn(whole_model)
            x = inverse_scaler(x)

            samples = np.clip(x.permute(0, 2, 3, 1).cpu().numpy()
                            * 255., 0, 255).astype(np.uint8)
            
            for seed, image_np in zip(batch_seeds, samples):
                image_dir = os.path.join(this_sample_dir, f'{seed-seed%1000:06d}') 
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')                        
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np, 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        dist.barrier()

        if global_rank == 0:
            mu, sigma = evaluation.calculate_inception_stats(this_sample_dir)
            ref = dict(np.load(config.ref_statistics))
            fid = evaluation.calculate_fid_from_inception_stats(
                mu, sigma, ref['mu'], ref['sigma'])
            logging.info('\tFID: %.6f' % fid)
        torch.cuda.empty_cache()
        dist.barrier()
