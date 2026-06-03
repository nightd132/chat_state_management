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


def extract_state(model_output):
    # recurret_states [batch, heads, head_dim, d_state]
    list_ssm_states = [layer.recurrent_states for layer in model_output.cache_params.layers]
    return list_ssm_states


def create_cache(num_layers=24, batch_size=1, num_heads=24, head_dim=64, d_state=128, conv_dim=1792, d_conv=4, device="cpu"):
    cache = DynamicCache()

    for i in range(num_layers):
        layer = LinearAttentionLayer()
        layer.recurrent_states = torch.zeros(batch_size, num_heads, head_dim, d_state).to(device)
        layer.conv_states      = torch.zeros(batch_size, conv_dim, d_conv).to(device)
        # layer.is_conv_states_initialized = True
        # layer.is_recurrent_states_initialized = True
        layer.has_previous_state = True # If set to True THe cache would be injected
        cache.layers.append(layer)
    return cache

def save_recurrent_states(cache_params):
    return {
        layer_idx: layer.recurrent_states.clone()
        for layer_idx, layer in enumerate(cache_params.layers)
        if layer.is_recurrent_states_initialized
    }

def save_conv_states(cache_params):
    return {
        layer_idx: layer.conv_states.clone()
        for layer_idx, layer in enumerate(cache_params.layers)
        if layer.is_conv_states_initialized
    }

def load_recurrent_states(saved, model, device=None, dtype=None):
    device = device or next(model.parameters()).device
    dtype  = dtype  or next(model.parameters()).dtype

    dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_ids, use_cache=True, return_dict=True)

    cache = dummy_out.cache_params

    for layer_idx, recurrent_states in saved.items():
        layer = cache.layers[layer_idx]
        layer.lazy_initialization(recurrent_states=recurrent_states.to(device=device, dtype=dtype))
        layer.recurrent_states.copy_(recurrent_states.to(device=device, dtype=dtype))

    return cache