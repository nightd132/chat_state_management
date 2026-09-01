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
    print(f"Session {session_id + 1}/{total_sessions} complete")