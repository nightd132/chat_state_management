from pathlib import Path
import yaml
import json

from src.state_utils import get_memory_size_kb
from src.cache_utils import get_cache_dir, get_json_cache_path

def read_config(config_path: str) -> dict:
    """Read a YAML experiment configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compress_states(ae, states, device):
    """Encode each state tensor and return CPU-resident latent tensors."""
    compressed = []
    for s in states:
        _, c = ae(s.to(device))
        compressed.append(c.cpu())
    return compressed

def decompress_states(ae, compressed_states, device):
    """Decode latent tensors and return CPU-resident reconstructed states."""
    reconstructed = []
    for c in compressed_states:
        r = ae.decoder(c.to(device))
        reconstructed.append(r.cpu())
    return reconstructed

def save_text(text: str, path: str) -> None:
    """Write text to a file, creating its parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(path: str) -> str:
    """Read and return the complete contents of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def concatenate_texts(texts: list[str]) -> str:
    """Join conversation fragments with newline separators."""
    return "\n".join(texts)

def run_with_cache(
    output_dir: str,
    method_label: str,
    session_id: int,
    force_rerun: bool,
    run_fn
) -> dict:
    """Load cached results or run, save, and return the supplied computation."""
    if not force_rerun:
        cached = load_cached_results(output_dir, method_label, session_id)
        if cached is not None:
            print(f"[cache] {method_label} session {session_id}: using cached results")
            return cached
    output_data = run_fn()
    save_cached_results(output_dir, method_label, session_id, output_data)
    return output_data

def load_cached_results(output_dir: str, method_label: str, session_id: int) -> dict | None:
    """Load per-session JSON results and restore integer turn keys."""
    path = get_json_cache_path(output_dir, method_label, session_id)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def save_cached_results(output_dir: str, method_label: str, session_id: int, output_data: dict) -> None:
    """Save per-session experiment results as JSON."""
    path = get_json_cache_path(output_dir, method_label, session_id)
    with open(path, "w") as f:
        json.dump(output_data, f)
