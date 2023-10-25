import os
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import Adam, Adamax, AdamW

from math import cos,pi

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

def annealing_cos(start, end, factor):
    """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * (start - end) * cos_out

def optimization_manager_speedup(config, phi_net = True):
    if phi_net:
        lr = config.phi_learning_rate
    else:
        lr = config.f_learning_rate
    def optimize_fn(optimizer, 
                    params, 
                    step, 
                    scaler=None,
                    lr=lr,
                    grad_clip=config.grad_clip):

        if config.n_warmup_iters > 0 and step <= config.n_warmup_iters:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / config.n_warmup_iters, 1.0)
        
        if step > config.n_warmup_iters and config.cosine_lr_decay=='use':
            target_lr = lr * 0.9**((config.n_train_iters//3000)) 
            for g in optimizer.param_groups:
                g['lr'] = annealing_cos(lr, target_lr, step / config.n_train_iters)
        elif step > config.n_warmup_iters:
            for g in optimizer.param_groups:
                g['lr'] = lr * 0.9**((step//3000))
        
        if scaler is None:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
        else:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

    return optimize_fn


def get_optimizer_speedup(config, params):
    if config.optimizer == 'Adam':
        optimizer = Adam(params, 
                        lr=config.phi_learning_rate, 
                        weight_decay=config.weight_decay,
                        betas=(.9, .999),
                        amsgrad=True)
    elif config.optimizer == 'Adamax':
        optimizer = Adamax(params, 
                        lr=config.phi_learning_rate, 
                        weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(params, 
                        lr=config.phi_learning_rate, 
                        weight_decay=config.weight_decay)
    else:
        raise NotImplementedError('Optimizer %s is not supported.' % config.optimizer)

    return optimizer



def get_data_inverse_scaler():
    return lambda x: (x + 1.) / 2.  # Rescale from [-1, 1] to [0, 1]
    


def batched_cov(x):
    covars = np.empty((x.shape[0], x.shape[2], x.shape[2]))
    for i in range(x.shape[0]):
        covars[i] = np.cov(x[i], rowvar=False)
    return covars


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def concatenate(tensor, world_size):
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)


def split_tensor(tensor, global_rank, global_size):
    if tensor.shape[0] / global_size - tensor.shape[0] // global_size > 1e-6:
        raise ValueError('Tensor is not divisible by global size.')
    return torch.chunk(tensor, global_size)[global_rank]


def set_seeds(rank, seed):
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    # torch.backends.cudnn.benchmark = True


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def save_img(images, filename, figsize=None):

    figsize = figsize if figsize is not None else (6, 6)

    nrow = int(np.sqrt(images.shape[0]))
    image_grid = make_grid(images, nrow)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).cpu())
    plt.savefig(filename)
    plt.close()


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size
    
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

