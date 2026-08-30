from pathlib import Path
import yaml
import json

def get_memory_size_kb(path):
    try:
        size_kb = Path(path).stat().st_size / 1024
    except FileNotFoundError:
        size_kb = 0.0
    return size_kb

def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compress_states(ae, states, device):
    compressed = []
    for s in states:
        _, c = ae(s.to(device))   # [T, D] → [T, hidden_dim]
        compressed.append(c.cpu())
    return compressed

def decompress_states(ae, compressed_states, device):
    reconstructed = []
    for c in compressed_states:
        r = ae.decoder(c.to(device))   # [T, hidden_dim] → [T, D]
        reconstructed.append(r.cpu())
    return reconstructed

def save_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def concatenate_texts(texts):
    return "\n".join(texts)

def _cache_path(output_dir, method_label, session_id):
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{method_label}_session{session_id}.json"

def load_cached_results(output_dir, method_label, session_id):
    path = _cache_path(output_dir, method_label, session_id)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def save_cached_results(output_dir, method_label, session_id, output_data):
    with open(_cache_path(output_dir, method_label, session_id), "w") as f:
        json.dump(output_data, f)

def run_with_cache(output_dir, method_label, session_id, force_rerun, run_fn):
    if not force_rerun:
        cached = load_cached_results(output_dir, method_label, session_id)
        if cached is not None:
            print(f"[cache] {method_label} session {session_id}: using cached results")
            return cached
    output_data = run_fn()
    save_cached_results(output_dir, method_label, session_id, output_data)
    return output_data


