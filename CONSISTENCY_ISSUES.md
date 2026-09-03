# Code Consistency Issues - Quick Reference

## 🔴 CRITICAL INCONSISTENCIES

### Issue 1: Duplicate Function Definition
**Files**: `src/utils.py` vs `src/state_utils.py`

```python
# ❌ PROBLEM: Same function defined twice

# In utils.py (line 5-10)
def get_memory_size_kb(path):
    try:
        size_kb = Path(path).stat().st_size / 1024
    except FileNotFoundError:
        size_kb = 0.0
    return size_kb

# In state_utils.py (line 29-33) - EXACT DUPLICATE
def get_memory_size_kb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1024
    except FileNotFoundError:
        return 0.0

# ✅ SOLUTION:
# Keep the better typed version (state_utils.py)
# Update imports in utils.py to:
# from src.state_utils import get_memory_size_kb
```

---

### Issue 2: Identical Helper Functions Repeated
**Files**: `src/experiment3.py:70-96` vs `src/experiment4.py:90-116`

```python
# ❌ PROBLEM: Same functions in different files

# experiment3.py
def aggregate_by_boundary_offset(output_data, exclude_first_session=True):
    bucket = defaultdict(lambda: defaultdict(list))
    for (_session_id, _turn_id), metrics in output_data.items():
        if exclude_first_session and metrics["is_first_session"]:
            continue
        offset = metrics["turns_since_boundary"]
        for k in ("state_ppl", "state_latency", "state_size_kb"):
            bucket[offset][k].append(metrics[k])
    # ... 10 more lines

# experiment4.py - IDENTICAL IMPLEMENTATION
def aggregate_by_boundary_offset(output_data, exclude_first_session=True):
    bucket = defaultdict(lambda: defaultdict(list))
    for (_session_id, _turn_id), metrics in output_data.items():
        if exclude_first_session and metrics["is_first_session"]:
            continue
        offset = metrics["turns_since_boundary"]
        for k in ("state_ppl", "state_latency", "state_size_kb"):
            bucket[offset][k].append(metrics[k])
    # ... same 10 lines

# ✅ SOLUTION: Create src/aggregation_utils.py
from src.aggregation_utils import aggregate_by_boundary_offset
```

---

### Issue 3: Path Handling Inconsistency
**Files**: All experiment files have different patterns

```python
# ❌ INCONSISTENT PATTERNS:

# Pattern 1: String concatenation (experiment1.py:110-114)
text_history_dir = paths["text_history_dir"]+"/history.txt"
output_dir = str(root) + "/" + paths["output_dir"]
state_dir = str(root) + "/" + paths["state_dir"] + "/state.pt"

# Pattern 2: pathlib.Path (state_utils.py:6)
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)

# Pattern 3: Mixed (experiment2.py:46-49)
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, path)

# ✅ SOLUTION: Use pathlib everywhere
text_history_dir = Path(root) / paths["text_history_dir"] / "history.txt"
output_dir = Path(root) / paths["output_dir"]
state_dir = Path(root) / paths["state_dir"] / "state.pt"
plot_dir = Path(root) / paths["plot_dir"]
```

---

## 🟡 READABILITY INCONSISTENCIES

### Issue 4: Import Style Mismatch
**Files**: Different experiments use different patterns

```python
# ❌ Pattern 1: Direct imports (experiment1.py:1-2)
from src.evaluate import evaluate_baseline, evaluate_injected

# ❌ Pattern 2: Module alias (experiment2.py:1)
from src import evaluate as evaluate_module

# ❌ Pattern 3: Full module import (experiment3.py:1)
from src import evaluate as evaluate_module

# Usage inconsistency:
# Pattern 1: evaluate_baseline(...)
# Pattern 2: evaluate_module.evaluate_baseline(...)
# Pattern 3: evaluate_module.evaluate_baseline(...)

# ✅ SOLUTION: Choose ONE and apply everywhere
# Recommended: Direct imports for clarity
from src.evaluate import (
    evaluate_baseline,
    evaluate_injected
)
```

---

### Issue 5: Print Formatting Inconsistency
**Files**: Multiple experiments

```python
# ❌ DIFFERENT STYLES:

# experiment1.py:51-56
print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# experiment2.py:28-30
print(f"\n{'='*50}")
print(f"Experiment 4: running chain with {label}")
print(f"{'='*50}")

# experiment3.py:132-134
print(f"\n{'='*50}")
print(f"Experiment 3: running chain with {label}")
print(f"{'='*50}")

# ✅ SOLUTION: Create log_utils.py with helpers
def print_section(title: str, width: int = 50):
    print(f"\n{'='*width}")
    print(title)
    print(f"{'='*width}")

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")

# Usage:
print_section(f"Experiment 4: running chain with {label}")
print_memory_stats()
```

---

### Issue 6: Magic Numbers in Code
**Files**: experiment2.py:169, 193, 314

```python
# ❌ SCATTERED MAGIC NUMBERS:
reconstructed = reconstructed.view(1, 24, 64, 128)  # Line 169
latent = ae_list[layer_idx].encoder(
    state.view(1, 24, -1)  # Line 193
)
reconstructed = reconstructed.view(1, 24, 64, 128).cpu().float()  # Line 314

# What do 24, 64, 128 mean? No context!

# ✅ SOLUTION: Define constants
NUM_HEADS = 24      # Number of attention heads
HEAD_DIM = 64       # Dimension per head
D_STATE = 128       # State dimension

# Usage:
reconstructed = reconstructed.view(1, NUM_HEADS, HEAD_DIM, D_STATE)
latent = ae_list[layer_idx].encoder(
    state.view(1, NUM_HEADS, -1)
)
```

---

### Issue 7: Repeated State Management Logic
**Files**: experiment1.py, experiment2.py, experiment3.py, experiment4.py

```python
# ❌ PATTERN REPEATED IN ALL EXPERIMENTS:

# experiment1.py:71-81 (simplified)
if turn_id == 0:
    output, latency, ppl = evaluate_baseline(...)
    ssm_states, conv_states = state_utils.extract_state(output)
else:
    new_ssm, new_conv, latency, ppl = evaluate_injected(...)
    ssm_states = new_ssm
    conv_states = new_conv

# experiment3.py:26-41 (similar structure, different details)
if turn_id == 0 and carryover_ssm is not None and alpha is not None:
    ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(...)
elif turn_id == 0:
    state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(...)
    ssm_states, conv_states = state_utils.extract_state(state_output)
else:
    prev_ssm, prev_conv = state_utils.load_state(state_dir, device="cpu")
    ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(...)

# ✅ SOLUTION: Extract to base class
class StateRunner:
    def run_turn(self, model, tokenizer, snap, device, state_dir):
        """Common state management pattern"""
        turn_id = snap["turn_id"]
        
        if turn_id == 0:
            output, latency, ppl = self.evaluate_first_turn(
                model, tokenizer, snap, device
            )
            ssm_states, conv_states = state_utils.extract_state(output)
        else:
            ssm_states, conv_states = state_utils.load_state(
                state_dir, device=device
            )
            ssm_states, conv_states, latency, ppl = evaluate_injected(
                model, tokenizer, snap["new_text"],
                ssm_states, conv_states, device=device
            )
        
        state_utils.save_state(ssm_states, conv_states, state_dir)
        return ssm_states, conv_states, latency, ppl
```

---

### Issue 8: Inconsistent Error Handling
**Files**: experiment3.py:50-54

```python
# ❌ SILENT ERROR HANDLING:
try:
    del ssm_states
    del conv_states
except Exception:
    pass  # Problems hidden!

# ✅ SOLUTIONS:

# Option 1: Only catch expected errors
try:
    del ssm_states
    del conv_states
except (NameError, AttributeError):
    # These variables might not exist, which is fine
    pass

# Option 2: Remove try-except if not needed
del ssm_states
del conv_states

# Option 3: Add logging
try:
    del ssm_states
    del conv_states
except Exception as e:
    logger.warning(f"Failed to clean up variables: {e}")
```

---

### Issue 9: Type Hints Missing
**Files**: All experiment files

```python
# ❌ NO TYPE HINTS:
def run_baseline_session(model, tokenizer, snapshots, device, text_history_dir, max_seq_len):
    output_data = {}
    history_text = ""
    # ... unclear what types are expected

# ✅ WITH TYPE HINTS:
from typing import Dict, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def run_baseline_session(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    snapshots: List[Dict],
    device: str,
    text_history_dir: str,
    max_seq_len: int
) -> Dict[int, Dict[str, float]]:
    output_data: Dict[int, Dict[str, float]] = {}
    history_text: str = ""
    # ... clear types!
```

---

## 📊 Consistency Matrix

| Aspect | Consistency | Examples |
|--------|-------------|----------|
| **Function naming** | ✅ Good | `run_baseline_session`, `evaluate_baseline` |
| **Variable naming** | ⚠️ Mixed | `ssm_states` vs `output_data`, `turn_id` vs `_turn_id` |
| **Path handling** | ❌ Poor | Mix of string concat and pathlib |
| **Imports** | ❌ Poor | Direct vs aliased imports vary |
| **Print formatting** | ❌ Poor | Different separator styles |
| **Error handling** | ⚠️ Inconsistent | Silent vs explicit handling |
| **Type hints** | ❌ Missing | Rarely used |
| **Documentation** | ⚠️ Sparse | Few docstrings |
| **Magic numbers** | ❌ Poor | Hardcoded dimensions scattered |
| **Comment style** | ⚠️ Inconsistent | Various formats |

---

## Quick Fixes (Ranked by Impact)

### 1. Remove duplicate `get_memory_size_kb()` (5 minutes)
```python
# In utils.py, replace with:
from src.state_utils import get_memory_size_kb
```

### 2. Standardize path handling (30 minutes)
```python
# Replace all string concatenation with:
from pathlib import Path
output_dir = Path(root) / paths["output_dir"]
```

### 3. Consolidate cache logic (20 minutes)
```python
# Create src/cache_utils.py with unified functions
```

### 4. Extract aggregation functions (10 minutes)
```python
# Move to src/aggregation_utils.py
# Import in exp3 and exp4
```

### 5. Add basic type hints (1-2 hours)
```python
# Add to all public functions
```

---

## Recommended Reading Order

1. **Start here**: Focus on "CRITICAL INCONSISTENCIES" (Issues 1-3)
2. **Then**: Address "READABILITY INCONSISTENCIES" (Issues 4-8)  
3. **Finally**: Polish with "Type hints" and formatting

This order maximizes code quality improvement with minimum effort!
