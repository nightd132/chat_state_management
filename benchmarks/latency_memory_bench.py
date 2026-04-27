import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from storage import state_io, history_text_io

def measure_baseline_latency(model, tokenizer, input_text: str, history: str, device: str = "cpu"):
    
    input_text = history_text_io.concatenate_texts([history, input_text])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    return output, latency

def measure_state_management_latency(model, tokenizer, input_text: str, saved_state: torch.Tensor, device: str = "cpu"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        cache_position = torch.arange(inputs["input_ids"].size(1), device=device)
        output = model(**inputs, cache_params=state_io.feed_synthetic_ssm_state(model, saved_state), cache_position=cache_position)
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time


    return output,latency


def get_memory_size_kb(path):
    try:
        size_kb = Path(path).stat().st_size / 1024
    except FileNotFoundError:
        size_kb = 0.0
    return size_kb



def print_benchmark_results(baseline_latency: float, state_management_latency: float):
    print(f"Baseline Latency: {baseline_latency*1000:.4f} milliseconds")
    print(f"State Management Latency: {state_management_latency*1000:.4f} milliseconds")

def plot_memory_growth(results, save_path):
    df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))

    plt.plot(df["turn"], df["txt_size_kb"], label="Text history (.txt)")
    plt.plot(df["turn"], df["pt_size_kb"], label="Mamba state (.pt)")

    plt.xlabel("Conversation Turn")
    plt.ylabel("Memory Size (KB)")
    plt.title("Memory Growth: Text History vs Saved State")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_latency_comparison(results, save_path):
    df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))

    plt.plot(df["turn"], df["baseline_latency"], label="Baseline Latency")
    plt.plot(df["turn"], df["state_latency"], label="State Management Latency")

    plt.xlabel("Conversation Turn")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Comparison: Baseline vs State Management")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")