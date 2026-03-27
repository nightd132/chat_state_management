import torch
from pathlib import Path
from transformers import MambaCache


def save_state(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.cpu(), path)


def load_state(path, device="cpu"):
    state = torch.load(path, map_location=device)
    return state.to(device)

def feed_synthetic_ssm_state(model, ssm_states):
    cache = MambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
    cache.ssm_states = [s.detach().clone() for s in ssm_states]
    return cache