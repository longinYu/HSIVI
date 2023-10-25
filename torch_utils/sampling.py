import torch
import gc
from torch_utils.utils import extract

def get_sampling_fn(config, diffusion, sampling_shape):
    return get_speedup_sampler(config, diffusion, sampling_shape)

@torch.no_grad()           
def get_speedup_sampler(config, diffusion, sampling_shape):
    ''' 
    Sampling using SpeedUp algorithm. 
    '''
    gc.collect()
    def speed_sampler(model):
        model.module.phinet.eval()
        with torch.no_grad():
            x = torch.randn(*sampling_shape, device=config.device)

            one_minus_alpha_bars = diffusion.one_minus_alpha_bars.to(x.device) 
            timesteps = diffusion.timesteps.to(x.device)

            for i in range(1, config.n_discrete_steps):
                positive_t = config.n_discrete_steps-i                         
                index_tensor = torch.ones((x.shape[0],), device=x.device, dtype=torch.long) * positive_t
                sigma_t = extract(one_minus_alpha_bars, index_tensor-1,(x.shape)).sqrt()
                sigma_t_plus_1 = extract(one_minus_alpha_bars, index_tensor,(x.shape)).sqrt()
                sample_t = extract(timesteps, index_tensor, (x.shape[0],))
                x, _ = model(x, time_cond=sample_t, 
                            gamma_index=index_tensor-1,
                            sigma_t=sigma_t,
                            object_='phinet', sigma_t_plus_1=sigma_t_plus_1)

        model.module.phinet.train()
        return x
    return speed_sampler
