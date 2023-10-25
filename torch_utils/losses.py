import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch_utils.utils import extract


def get_loss_fn_speedup(diffusion, config):
    def loss_fn_speedup(model, training_shape, train_phiNet=False, train_FNet=False):
        # Setting up initial means
        device = model.device
        one_minus_alpha_bars = diffusion.one_minus_alpha_bars.to(device)
        timesteps = diffusion.timesteps.to(device)

        rand_index = torch.randint(
                low=1, high=config.n_discrete_steps, size=(training_shape[0] // 2 + 1,)
            ).to(device)
        rand_index = torch.cat([rand_index, config.n_discrete_steps - rand_index], dim=0)[:training_shape[0]]
        t_index_tensor = config.n_discrete_steps - rand_index
        t_index_predict_tensor = t_index_tensor - 1

        with torch.no_grad():
            perturbed_data = torch.randn(*training_shape, dtype=torch.float32, device=device)
            all_perturbed_data = perturbed_data.unsqueeze(0).repeat(config.n_discrete_steps-1, 1,1,1,1)
            for tick in range(1, torch.max(rand_index)):                         
                index_tensor = torch.ones((training_shape[0],), device=device, dtype=torch.long) * tick
                sigma_t = extract(one_minus_alpha_bars, config.n_discrete_steps - index_tensor-1, training_shape).sqrt()
                sigma_t_plus_1 = extract(one_minus_alpha_bars, config.n_discrete_steps - index_tensor, training_shape).sqrt()
                sample_t = extract(timesteps, config.n_discrete_steps - index_tensor, (training_shape[0],))

                perturbed_data, _ = model(perturbed_data, 
                                                time_cond=sample_t, 
                                                gamma_index=config.n_discrete_steps - index_tensor-1,
                                                sigma_t=sigma_t,
                                                object_='phinet', sigma_t_plus_1=sigma_t_plus_1)
                all_perturbed_data[tick, ...] = perturbed_data.clone().detach()

            rand_index_ = rand_index - 1   
            perturbed_data = torch.stack([all_perturbed_data[rand_index_[b], b, ...] for b in range(training_shape[0])])
            del all_perturbed_data

        perturbed_data = perturbed_data.to(torch.float32)
            
        sigma_t = extract(one_minus_alpha_bars, t_index_predict_tensor, training_shape).sqrt()
        sigma_t_plus_1 = extract(one_minus_alpha_bars, t_index_tensor, training_shape).sqrt()
        train_t = extract(timesteps, t_index_tensor, (training_shape[0],))
        train_t_pred = extract(timesteps, t_index_predict_tensor, (training_shape[0],))
        
        if train_phiNet == True and train_FNet==False  :
            # phiNet                
            x_pred_t, eps_div_gamma = model(perturbed_data,                                                 
                                            time_cond=train_t, 
                                            gamma_index=t_index_predict_tensor,
                                            sigma_t = sigma_t, 
                                            object_='phinet',
                                            sigma_t_plus_1=sigma_t_plus_1) 
            target_epsilon = model(x_pred_t, time_cond=train_t_pred, object_='teacher')
            eps, gamma = eps_div_gamma
            f_weight = model(x_pred_t, time_cond=train_t_pred, object_='fnet')
            g_weight = (f_weight - target_epsilon)
            losses = torch.sum(torch.mean((2 * g_weight * (-target_epsilon + eps/gamma) - g_weight**2) * sigma_t**2, 0))
        
        elif train_FNet == True and train_phiNet==False :
            with torch.no_grad():
                # phiNet                
                x_pred_t, eps_div_gamma = model(perturbed_data,                                                 
                                                time_cond=train_t, 
                                                gamma_index=t_index_predict_tensor,
                                                sigma_t = sigma_t,
                                                object_='phinet',
                                                sigma_t_plus_1=sigma_t_plus_1) 
                eps, gamma = eps_div_gamma
            
            f_weight = model(x_pred_t, time_cond=train_t_pred, object_='fnet')
            losses = torch.sum(torch.mean(sigma_t**2 * (f_weight - eps/gamma)**2, 0))
            
        if torch.sum(torch.isnan(losses)) > 0:
            raise ValueError(
                'NaN loss during training!')

        return losses 
    return loss_fn_speedup


def get_step_fn_speedup(train, phi_optimize_fn, f_optimize_fn, diffusion, config):
    loss_fn = get_loss_fn_speedup(diffusion, config)

    scaler = GradScaler() if config.autocast_train else None

    def step_fn_speedup(state, training_shape, optimization=True):
        whole_model = state['whole_model']

        if train:
            torch.cuda.empty_cache()
            ### Optimize the phiNet
            loss_phi = torch.tensor(0.0).to(whole_model.device)
            if state['step'] % config.f_learning_times == 0:
                for param in whole_model.module.phinet.parameters():
                    param.requires_grad = True
                for param in whole_model.module.fnet.parameters():
                    param.requires_grad = False

                with autocast(enabled=config.autocast_train):
                    loss_phi = loss_fn(whole_model, training_shape, train_phiNet=True, train_FNet=False)
                if optimization:
                    if config.autocast_train:
                        scaler.scale(loss_phi).backward()
                    else:
                        loss_phi.backward()
                    
                    optimizer_phi = state['optimizer_phinet']
                
                    phi_optimize_fn(optimizer_phi, whole_model.module.phinet.parameters(), 
                                    step=state['step'])
                    optimizer_phi.zero_grad()
                    state['ema_phinet'].update(whole_model.module.phinet.parameters())

            ### Optimize the FNet
            for param in whole_model.module.phinet.parameters():
                param.requires_grad = False
            for param in whole_model.module.fnet.parameters():
                param.requires_grad = True

            with autocast(enabled=config.autocast_train):
                loss_fnet = loss_fn(whole_model, training_shape, train_phiNet=False, train_FNet=True)
            if optimization:
                if config.autocast_train:
                    scaler.scale(loss_fnet).backward()
                else:
                    loss_fnet.backward()
                optimizer_fnet = state['optimizer_fnet']
        
                f_optimize_fn(optimizer_fnet, whole_model.module.fnet.parameters(), step=state['step'])
                optimizer_fnet.zero_grad()
                state['ema_fnet'].update(whole_model.module.fnet.parameters())
            # update here, and stop lr decay at f_optimize_fn and phi_optimize_fn
            state['step'] += 1  
            return loss_phi, loss_fnet
        else:
            with torch.no_grad():
                with autocast(enabled=config.autocast_eval):
                    loss = loss_fn(whole_model, training_shape, train, step = state["step"],train_FNet=False, train_phiNet=False)
            return loss, loss
    return step_fn_speedup
