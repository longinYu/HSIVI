import torch
import torch.nn as nn
import numpy as np
from torch_utils.utils import extract
import torch.nn.functional as F

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
    
class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_discrete_steps = self.config.n_discrete_steps  
        
        #! for imagenet
        if config.dataset =='imagenet':
            original_betas = betas_for_alpha_bar(
                self.config.gt_steps,
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            )
            original_betas = torch.from_numpy(original_betas).float()
        else:
            min_beta = 1e-4
            max_beta = 0.02 
            original_betas = torch.linspace(min_beta, max_beta, self.config.gt_steps, dtype = torch.float32) # 1000
        
        original_alphas = (1. - original_betas)
        original_alphas_bar = torch.cumprod(original_alphas, dim=0).type(torch.float32)
        original_betas = original_betas.type(torch.float32)
        original_alphas = original_alphas.type(torch.float32)
        if config.skip_type == "uniform":
            skip = self.config.gt_steps // self.n_discrete_steps
            seq = range(0, self.config.gt_steps, skip)
            self.timesteps = torch.from_numpy(np.array(seq))
        elif config.skip_type =='quad':
            self.timesteps = (torch.linspace(
                                0, np.sqrt(self.config.gt_steps * 0.8), self.n_discrete_steps
                                )**2
                            ).round().long() # 21       
        self.alphas_bar = original_alphas_bar[self.timesteps]
        self.one_minus_alpha_bars = 1. - self.alphas_bar
        self.betas = 1.0 - (1.0 - self.one_minus_alpha_bars[1:]) / (1.0 - self.one_minus_alpha_bars[:-1]) 
        self.betas = F.pad(self.betas, (1, 0), value = 1.)
        self.betas = self.betas.type(torch.float32)
    
    def mean_and_var(self, x, t):
        self.one_minus_alpha_bars = self.one_minus_alpha_bars.to(x.device)
        self.alphas_bar = self.alphas_bar.to(x.device)
        mean = x * extract(torch.sqrt(self.alphas_bar), t, x.shape).to(x.device) 
        var = extract(self.one_minus_alpha_bars, t, x.shape).to(x.device)   
        return  mean, var
        
    
