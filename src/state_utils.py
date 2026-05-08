import torch
from pathlib import Path
from transformers.cache_utils import DynamicCache, LinearAttentionLayer


def save_state(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    

def load_state(path, device="cpu"):
    state = torch.load(path, map_location=device)
    return state

# def feed_synthetic_ssm_state(model, ssm_states):
#     cache = MambaCache(config=model.config, max_batch_size=1, 
#                        device=model.device, dtype=model.dtype)
#     for i in range(len(ssm_states)):
#         cache.update_ssm_state(i, ssm_states[i].to(model.device))
#     return cache

# def feed_synthetic_state(model, ssm_states, conv_states):
#     cache = MambaCache(config=model.config, max_batch_size=1, 
#                        device=model.device, dtype=model.dtype)
#     cache.ssm_states = [s.unsqueeze(0).to(model.device) 
#                         for s in ssm_states]
#     cache.conv_states = [s.unsqueeze(0).to(model.device) 
#                         for s in conv_states]
#     return cache

def extract_state(model_output):
    # recurret_states [batch, heads, head_dim, d_state]
    list_ssm_states = [layer.recurrent_states for layer in model_output.cache_params.layers]
    return list_ssm_states

def create_cache(num_layers=24, batch_size=1, heads=24, head_dim=64, d_state=128, d_inner=1792, d_conv=4, device="cpu"):
    cache = DynamicCache()

    for i in range(num_layers):
        layer = LinearAttentionLayer()
        layer.recurrent_states = torch.zeros(batch_size, heads, head_dim, d_state).to(device)
        layer.conv_states      = torch.zeros(batch_size, d_inner, d_conv).to(device)
        layer.has_previous_state = False
        cache.layers.append(layer)
    return cache