# File-by-File Code Improvement Recommendations

## 1. `src/utils.py`

### Current Issues:
```python
# Line 5-10: Duplicate function (also in state_utils.py)
def get_memory_size_kb(path):
    try:
        size_kb = Path(path).stat().st_size / 1024
    except FileNotFoundError:
        size_kb = 0.0
    return size_kb

# Line 45-49: Helper function that creates cache paths
def _cache_path(output_dir, method_label, session_id):
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{method_label}_session{session_id}.json"
```

### Recommendations:
```python
# ✅ REFACTORED utils.py
from pathlib import Path
import yaml
import json

# Import from proper module instead of duplicating
from src.state_utils import get_memory_size_kb
from src.cache_utils import get_cache_dir, get_json_cache_path

# Keep only utility functions that don't duplicate elsewhere
def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compress_states(ae, states, device):
    """Compress states using autoencoder"""
    compressed = []
    for s in states:
        _, c = ae(s.to(device))
        compressed.append(c.cpu())
    return compressed

def decompress_states(ae, compressed_states, device):
    """Decompress states using autoencoder decoder"""
    reconstructed = []
    for c in compressed_states:
        r = ae.decoder(c.to(device))
        reconstructed.append(r.cpu())
    return reconstructed

def save_text(text: str, path: str) -> None:
    """Save text to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(path: str) -> str:
    """Load text from file"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def concatenate_texts(texts: list[str]) -> str:
    """Concatenate multiple text strings with newline separator"""
    return "\n".join(texts)

def run_with_cache(
    output_dir: str,
    method_label: str,
    session_id: int,
    force_rerun: bool,
    run_fn
) -> dict:
    """Run function with caching support"""
    if not force_rerun:
        cached = load_cached_results(output_dir, method_label, session_id)
        if cached is not None:
            print(f"[cache] {method_label} session {session_id}: using cached results")
            return cached
    output_data = run_fn()
    save_cached_results(output_dir, method_label, session_id, output_data)
    return output_data

def load_cached_results(output_dir: str, method_label: str, session_id: int) -> dict | None:
    """Load cached results from JSON"""
    path = get_json_cache_path(output_dir, method_label, session_id)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def save_cached_results(output_dir: str, method_label: str, session_id: int, output_data: dict) -> None:
    """Save results to JSON cache"""
    path = get_json_cache_path(output_dir, method_label, session_id)
    with open(path, "w") as f:
        json.dump(output_data, f)
```

---

## 2. `src/state_utils.py`

### Current Issues:
```python
# Duplicate get_memory_size_kb function
# Commented-out debug prints
```

### Recommendations:
```python
# ✅ REFACTORED state_utils.py
import torch
from pathlib import Path
from typing import Tuple

def save_state(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    path: str,
    dtype: torch.dtype = torch.float32
) -> None:
    """Save SSM and Conv states to disk"""
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
    """Load SSM and Conv states from disk"""
    data = torch.load(path, map_location=device)
    return data["ssm"], data["conv"]

def extract_state(model_output) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract SSM and Conv states from model output"""
    layers = model_output.cache_params.layers
    ssm_list = [layer.recurrent_states.cpu().float() for layer in layers]
    conv_list = [layer.conv_states.cpu().float() for layer in layers]
    return torch.stack(ssm_list, dim=0), torch.stack(conv_list, dim=0)

def get_memory_size_kb(path: str) -> float:
    """Get file size in kilobytes"""
    try:
        return Path(path).stat().st_size / 1024
    except FileNotFoundError:
        return 0.0
```

---

## 3. `src/cache_utils.py` (NEW FILE)

### Purpose:
Centralize all cache-related operations

```python
# ✅ NEW FILE: cache_utils.py
from pathlib import Path
import torch
import json
from typing import Any

def get_cache_dir(output_dir: str, ensure_exists: bool = True) -> Path:
    """Get or create cache directory"""
    cache_dir = Path(output_dir) / "cache"
    if ensure_exists:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_torch_cache_path(output_dir: str, exp_name: str, label: str) -> Path:
    """Get path for PyTorch tensor cache"""
    return get_cache_dir(output_dir) / f"{exp_name}_{label}.pt"

def get_json_cache_path(
    output_dir: str,
    method_label: str,
    session_id: int
) -> Path:
    """Get path for JSON cache"""
    return get_cache_dir(output_dir) / f"{method_label}_session{session_id}.json"

def save_torch_cache(data: dict, output_dir: str, exp_name: str, label: str) -> None:
    """Save PyTorch data to cache"""
    path = get_torch_cache_path(output_dir, exp_name, label)
    torch.save(data, path)

def load_torch_cache(output_dir: str, exp_name: str, label: str, device: str = "cpu") -> dict | None:
    """Load PyTorch data from cache"""
    path = get_torch_cache_path(output_dir, exp_name, label)
    if not path.exists():
        return None
    return torch.load(path, map_location=device)

def save_json_cache(data: dict, output_dir: str, method_label: str, session_id: int) -> None:
    """Save JSON data to cache"""
    path = get_json_cache_path(output_dir, method_label, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

def load_json_cache(output_dir: str, method_label: str, session_id: int) -> dict | None:
    """Load JSON data from cache"""
    path = get_json_cache_path(output_dir, method_label, session_id)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
```

---

## 4. `src/aggregation_utils.py` (NEW FILE)

### Purpose:
Consolidate aggregation functions used by experiments 3 and 4

```python
# ✅ NEW FILE: aggregation_utils.py
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def aggregate_by_boundary_offset(
    output_data: Dict,
    exclude_first_session: bool = True
) -> Dict[int, Dict[str, float]]:
    """Aggregate metrics by turns since session boundary"""
    bucket = defaultdict(lambda: defaultdict(list))
    
    for (_session_id, _turn_id), metrics in output_data.items():
        if exclude_first_session and metrics["is_first_session"]:
            continue
        
        offset = metrics["turns_since_boundary"]
        for k in ("state_ppl", "state_latency", "state_size_kb"):
            bucket[offset][k].append(metrics[k])
    
    agg = {}
    for offset, metric_lists in bucket.items():
        agg[offset] = {}
        n = None
        for k, values in metric_lists.items():
            agg[offset][f"{k}_mean"] = float(np.mean(values))
            agg[offset][f"{k}_std"] = float(np.std(values))
            n = len(values)
        agg[offset]["n"] = n
    
    return agg

def aggregate_boundary_only_by_session(
    output_data: Dict
) -> List[Tuple[int, float]]:
    """Extract boundary turn perplexity by session"""
    rows = []
    for (session_id, turn_id), metrics in sorted(output_data.items()):
        if turn_id == 0 and not metrics["is_first_session"]:
            rows.append((session_id, metrics["state_ppl"]))
    return rows

def summarize_boundary_health(
    df,
    threshold: float = 1.5
) -> None:
    """Print health summary of boundary turns"""
    baseline = df[(df["label"] == "cold_start") & (df["offset"] == 0)]
    if baseline.empty:
        print("No cold_start baseline found at offset=0 -- skipping summary.")
        return
    
    baseline_ppl = baseline["state_ppl_mean"].iloc[0]
    
    print_section("Boundary-turn perplexity vs cold_start baseline")
    print(f"Baseline PPL: {baseline_ppl:.3f}\n")
    
    for label in sorted(df["label"].unique()):
        if label == "cold_start":
            continue
        row = df[(df["label"] == label) & (df["offset"] == 0)]
        if row.empty:
            continue
        
        ppl = row["state_ppl_mean"].iloc[0]
        ratio = ppl / baseline_ppl
        status = "✅ OK" if ratio <= threshold else "⚠️ DEGRADED"
        print(f"  [{status}] {label:20s}  ppl={ppl:8.3f}  ratio={ratio:5.2f}x")

def print_section(title: str, width: int = 60) -> None:
    """Print formatted section header"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
```

---

## 5. `src/log_utils.py` (NEW FILE)

### Purpose:
Centralize all logging and printing utilities

```python
# ✅ NEW FILE: log_utils.py
import torch
from typing import Optional

def print_section(title: str, width: int = 50, char: str = "=") -> None:
    """Print a formatted section header"""
    print(f"\n{char*width}")
    print(f"{title}")
    print(f"{char*width}")

def print_subsection(title: str, width: int = 50, char: str = "-") -> None:
    """Print a formatted subsection header"""
    print(f"\n{char*width}")
    print(f"  {title}")
    print(f"{char*width}")

def print_memory_stats(label: str = "", device: str = "cuda") -> None:
    """Print GPU memory statistics"""
    if not torch.cuda.is_available():
        print(f"{label} GPU not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    if label:
        print(f"{label}")
    print(f"  Allocated: {allocated:8.2f} GB")
    print(f"  Reserved:  {reserved:8.2f} GB")

def print_progress(message: str, current: int, total: int) -> None:
    """Print progress message"""
    percentage = (current / total) * 100
    print(f"[{current:3d}/{total:3d}] ({percentage:5.1f}%) {message}")

def print_experiment_header(exp_num: int, method: str = "", n_sessions: Optional[int] = None) -> None:
    """Print experiment header"""
    title = f"Experiment {exp_num}"
    if method:
        title += f": {method}"
    if n_sessions:
        title += f" ({n_sessions} sessions)"
    print_section(title, width=60)

def log_cache_load(label: str, cache_path: str) -> None:
    """Log cache loading message"""
    print(f"[cache] {label}: loading from {cache_path}")

def log_cache_save(label: str, cache_path: str) -> None:
    """Log cache saving message"""
    print(f"[cache] {label}: saved to {cache_path}")

def log_session_complete(session_id: int, total_sessions: int) -> None:
    """Log session completion"""
    print(f"✓ Session {session_id + 1}/{total_sessions} complete")
```

---

## 6. `src/experiment1.py` - Key Changes

### Current Issues:
```python
# Line 110-114: String concatenation for paths
text_history_dir = paths["text_history_dir"]+"/history.txt"
output_dir = str(root) + "/" + paths["output_dir"]

# Line 51-56: Inconsistent print formatting
print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### Recommendations:
```python
# ✅ REFACTORED experiment1.py (key sections)
from pathlib import Path
from src import model_loader, state_utils, data, plot, utils
from src.evaluate import evaluate_baseline, evaluate_injected
from src.mamba2_stateful import patch_model
from src.log_utils import print_section, print_memory_stats

# ... (keep run_baseline_session and run_injected_session mostly the same)

def main():
    torch.manual_seed(0)
    config = utils.read_config("configs/config1.yaml")
    
    # ✅ Use pathlib for all paths
    paths = config["paths"]
    root = Path(__file__).parent.parent
    
    text_history_dir = root / paths["text_history_dir"] / "history.txt"
    output_dir = root / paths["output_dir"] / "experiment1"
    state_dir = root / paths["state_dir"] / "state.pt"
    plot_dir = root / paths["plot_dir"] / "experiment1"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    max_seq_len = config["data"]["max_length"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ... load model and dataset ...
    
    # ✅ Use logging utilities
    print_section(f"Running Experiment 1 on {len(test_sessions)} test sessions")
    
    # In run_baseline_session, replace print statements:
    # OLD:
    # print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # NEW:
    print_memory_stats()  # Uses consistent formatting
    
    # ... rest of main ...
```

---

## 7. `src/experiment2.py` - Key Changes

### Current Issues:
```python
# Line 169, 193, 314: Hardcoded tensor dimensions
reconstructed = reconstructed.view(1, 24, 64, 128)

# Line 1: Module alias import
from src import evaluate as evaluate_module
```

### Recommendations:
```python
# ✅ REFACTORED experiment2.py (key sections)
from pathlib import Path
from src import model_loader, state_utils, evaluate, data, autoencoder, plot, utils
from src.cache_utils import get_torch_cache_path, save_torch_cache, load_torch_cache
from src.log_utils import print_section, print_memory_stats

# ✅ Define constants at module level
NUM_HEADS = 24
HEAD_DIM = 64
D_STATE = 128

def run_autoencoder(model, tokenizer, snapshots, device, state_dir, ae_list):
    """Run experiment with autoencoder compression"""
    output_data = {}
    
    for ae in ae_list:
        ae.eval()
        ae.to(device)
    
    num_layers = len(ae_list)
    
    for snap in snapshots:
        turn_id = snap["turn_id"]
        
        if turn_id == 0:
            # ... turn 0 logic ...
            pass
        else:
            # ✅ Use constants instead of magic numbers
            payload = load_torch_cache(state_dir, device=device)
            latents = payload["ssm_latents"]
            conv_states = payload["conv_states"].to(device)
            
            decompressed_layers = []
            for layer_idx in range(num_layers):
                latent = latents[layer_idx].to(device).unsqueeze(0)
                reconstructed = ae_list[layer_idx].decoder(latent)
                
                # ✅ Clear what dimensions are
                reconstructed = reconstructed.view(1, NUM_HEADS, HEAD_DIM, D_STATE)
                decompressed_layers.append(reconstructed)
            
            ssm_states = torch.stack(decompressed_layers, dim=0)
            # ... rest of logic ...

        # ✅ Use cache utilities
        latents = {}
        for layer_idx in range(num_layers):
            state = ssm_states[layer_idx].to(device)
            latent = ae_list[layer_idx].encoder(
                state.view(1, NUM_HEADS, -1)  # ✅ Clear what -1 expands to
            )
            latents[layer_idx] = latent.squeeze(0).cpu()
        
        payload = {
            "ssm_latents": latents,
            "conv_states": conv_states.cpu()
        }
        save_torch_cache(payload, state_dir, "exp2", "ae")
        # ... rest ...
```

---

## 8. `src/experiment3.py` - Key Changes

### Current Issues:
```python
# Line 70-96: Duplicate aggregation functions
# Line 50-54: Silent error handling
```

### Recommendations:
```python
# ✅ REFACTORED experiment3.py
from src import model_loader, state_utils, evaluate, data, utils, plot
from src.mamba2_stateful import patch_model
from src.aggregation_utils import (
    aggregate_by_boundary_offset,
    aggregate_boundary_only_by_session,
    summarize_boundary_health
)
from src.cache_utils import get_torch_cache_path, load_torch_cache, save_torch_cache
from src.log_utils import print_section, log_cache_load

# ✅ Remove duplicate aggregation functions
# They're now imported from aggregation_utils.py

def apply_forgetting_factor(ssm_states: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply exponential forgetting factor to states"""
    return ssm_states * alpha

def run_state_chain(
    model,
    tokenizer,
    sessions,
    device,
    state_dir: str,
    alpha: float = None
) -> dict:
    output_data = {}
    carryover_ssm, carryover_conv = None, None
    
    for session_id, session in enumerate(sessions):
        snapshots = data.build_turn_snapshots(session)
        
        for snap in snapshots:
            turn_id = snap["turn_id"]
            turns_since_boundary = turn_id
            
            if turn_id == 0 and carryover_ssm is not None and alpha is not None:
                seed_ssm = apply_forgetting_factor(carryover_ssm, alpha)
                seed_conv = carryover_conv
                ssm_states, conv_states, state_latency, state_ppl = evaluate.evaluate_injected(
                    model, tokenizer, snap["new_text"], seed_ssm, seed_conv, device=device
                )
            elif turn_id == 0:
                state_output, state_latency, state_ppl = evaluate.evaluate_baseline(
                    model, tokenizer, snap["history_text"], snap["new_text"], device=device
                )
                ssm_states, conv_states = state_utils.extract_state(state_output)
            else:
                prev_ssm, prev_conv = state_utils.load_state(state_dir, device="cpu")
                ssm_states, conv_states, state_latency, state_ppl = evaluate.evaluate_injected(
                    model, tokenizer, snap["new_text"], prev_ssm, prev_conv, device=device
                )
            
            state_utils.save_state(ssm_states, conv_states, state_dir)
            carryover_ssm, carryover_conv = ssm_states.cpu(), conv_states.cpu()
            
            # ✅ Better error handling
            if ssm_states is not None:
                del ssm_states
            if conv_states is not None:
                del conv_states
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if snap["role"] == "assistant":
                output_data[(session_id, turn_id)] = {
                    "state_latency": state_latency,
                    "state_size_kb": utils.get_memory_size_kb(state_dir),
                    "state_ppl": state_ppl,
                    "turns_since_boundary": turns_since_boundary,
                    "is_first_session": session_id == 0,
                }
    
    return output_data
```

---

## 9. `src/experiment4.py` - Key Changes

### Similar to experiment3.py but with EMA updates
```python
# ✅ REFACTORED experiment4.py (key changes)
from src.aggregation_utils import (
    aggregate_by_boundary_offset,
    aggregate_boundary_only_by_session,
    summarize_boundary_health
)
from src.cache_utils import get_torch_cache_path, load_torch_cache, save_torch_cache
from src.log_utils import print_section

# Remove duplicate functions - import instead
# Functions aggregate_by_boundary_offset, aggregate_boundary_only_by_session, 
# and summarize_boundary_health are now imported

# Rest of file remains mostly the same, just update import statements
```

---

## Summary of Changes

| File | Change Type | Impact | Priority |
|------|------------|--------|----------|
| utils.py | Remove duplicate function | High | 🔴 |
| state_utils.py | Keep as-is, document | Low | ✅ |
| cache_utils.py | CREATE NEW | High | 🔴 |
| aggregation_utils.py | CREATE NEW | High | 🔴 |
| log_utils.py | CREATE NEW | Medium | 🟡 |
| experiment1.py | Path handling, logging | Medium | 🟡 |
| experiment2.py | Constants, imports | Medium | 🟡 |
| experiment3.py | Remove duplication, logging | Medium | 🟡 |
| experiment4.py | Remove duplication, logging | Medium | 🟡 |

---

## Implementation Steps

1. **Create new utility modules** (5 min)
   - cache_utils.py
   - aggregation_utils.py
   - log_utils.py

2. **Update imports in experiment files** (10 min)
   - Remove duplicate functions
   - Import from new modules

3. **Standardize path handling** (30 min)
   - Convert all string concatenation to pathlib

4. **Replace print statements** (20 min)
   - Use log_utils functions for consistency

5. **Add type hints** (1-2 hours)
   - Gradually across all files

6. **Test everything** (30 min)
   - Ensure all experiments still work
