import torch
from pathlib import Path
from typing import Tuple

def save_state(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    path: str,
    dtype: torch.dtype = torch.float32
) -> None:
    """Save recurrent SSM and convolution states as CPU tensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ssm": ssm_states.cpu().to(dtype),
            "conv": conv_states.cpu().to(dtype)
        },
        path,
    )

def load_state(path: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load saved recurrent states and map them to the requested device."""
    data = torch.load(path, map_location=device)
    return data["ssm"], data["conv"]

def extract_state(model_output) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and stack recurrent states from a model cache output."""
    layers = model_output.cache_params.layers
    ssm_list = [layer.recurrent_states.cpu().float() for layer in layers]
    conv_list = [layer.conv_states.cpu().float() for layer in layers]
    return torch.stack(ssm_list, dim=0), torch.stack(conv_list, dim=0)

def get_memory_size_kb(path: str) -> float:
    """Return a saved file size in kilobytes, or zero if it is absent."""
    try:
        return Path(path).stat().st_size / 1024
    except FileNotFoundError:
        return 0.0