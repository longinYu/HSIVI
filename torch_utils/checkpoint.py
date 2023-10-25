import torch
import os
import logging

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        logging.warning(
            'No checkpoint found at %s. Returned the same state as input.' % ckpt_dir)
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer_phinet'].load_state_dict(loaded_state['optimizer_phinet'])
        state['optimizer_fnet'].load_state_dict(loaded_state['optimizer_fnet'])
        state['whole_model'].load_state_dict(loaded_state['whole_model'], strict=False)
        state['ema_phinet'].load_state_dict(loaded_state['ema_phinet'])
        state['ema_fnet'].load_state_dict(loaded_state['ema_fnet'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer_phinet': state['optimizer_phinet'].state_dict(),
        'optimizer_fnet': state['optimizer_fnet'].state_dict(),
        'whole_model': state['whole_model'].state_dict(),
        'ema_phinet': state['ema_phinet'].state_dict(),
        'ema_fnet': state['ema_fnet'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
