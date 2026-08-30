import torch
from pathlib import Path


def save_state(ssm_states: torch.Tensor, conv_states: torch.Tensor, path: str, dtype=torch.float32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"Saving state with type {ssm_states.dtype}")
    torch.save(
        {"ssm": ssm_states.cpu().to(dtype), "conv": conv_states.cpu().to(dtype)},
        path,
    )


def load_state(path: str, device="cpu"):
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
