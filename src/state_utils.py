import torch
from pathlib import Path


def save_state(ssm_states: torch.Tensor, conv_states: torch.Tensor, path: str, dtype=torch.float32):
    """
    Save (ssm_states, conv_states) to a .pt file.
    dtype=float16 — half storage vs fp32, ~1e-3 round-trip error (safe).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"Saving state with type {ssm_states.dtype}")
    torch.save(
        {"ssm": ssm_states.to(dtype), "conv": conv_states.to(dtype)},
        path,
    )


def load_state(path: str, device="cpu"):
    """
    Load states saved with save_state().
    Returns (ssm_states, conv_states) on `device`.
    """
    data = torch.load(path, map_location=device)
    return data["ssm"], data["conv"]



def extract_state(model_output):
    layers = model_output.cache_params.layers
    ssm_list  = [layer.recurrent_states.cpu().float() for layer in layers]
    conv_list = [layer.conv_states.cpu().float() for layer in layers]
    # print(f"ssm_list type {ssm_list[0].dtype}")
    return torch.stack(ssm_list, dim=0), torch.stack(conv_list, dim=0)


def get_memory_size_kb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1024
    except FileNotFoundError:
        return 0.0
