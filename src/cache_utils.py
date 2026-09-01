from pathlib import Path
import torch
import json
from typing import Any

def get_cache_dir(output_dir: str, ensure_exists: bool = True) -> Path:
    """Return the experiment cache directory, optionally creating it."""
    cache_dir = Path(output_dir) / "cache"
    if ensure_exists:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_torch_cache_path(output_dir: str, exp_name: str, label: str) -> Path:
    """Build the path for a named PyTorch experiment cache."""
    return get_cache_dir(output_dir) / f"{exp_name}_{label}.pt"

def get_json_cache_path(
    output_dir: str,
    method_label: str,
    session_id: int
) -> Path:
    """Build the path for a per-session JSON cache."""
    return get_cache_dir(output_dir) / f"{method_label}_session{session_id}.json"

def save_torch_cache(data: dict, output_dir: str, exp_name: str, label: str) -> None:
    """Serialize experiment data to a PyTorch cache file."""
    path = get_torch_cache_path(output_dir, exp_name, label)
    torch.save(data, path)

def load_torch_cache(output_dir: str, exp_name: str, label: str, device: str = "cpu") -> dict | None:
    """Load a PyTorch cache, returning None when it does not exist."""
    path = get_torch_cache_path(output_dir, exp_name, label)
    if not path.exists():
        return None
    return torch.load(path, map_location=device)

def save_json_cache(data: dict, output_dir: str, method_label: str, session_id: int) -> None:
    """Serialize per-session results to a JSON cache file."""
    path = get_json_cache_path(output_dir, method_label, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

def load_json_cache(output_dir: str, method_label: str, session_id: int) -> dict | None:
    """Load a JSON cache and restore integer turn keys."""
    path = get_json_cache_path(output_dir, method_label, session_id)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}