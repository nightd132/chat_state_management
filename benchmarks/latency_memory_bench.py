import time
import torch
from state import state_io, history_text_io

def measure_baseline_latency(model, tokenizer, input_text: str, history: str, device: str = "cpu"):
    
    input_text = history_text_io.concatenate_texts([history, input_text])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        model(**inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = end_time - start_time
    return latency

def measure_state_management_latency(model, tokenizer, input_text: str, saved_state: torch.Tensor, device: str = "cpu"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    torch.cuda.synchronize() 
    start_time = time.time()
    with torch.no_grad():
        cache_position = torch.arange(inputs["input_ids"].size(1), device=device)
        model(**inputs, catch_param=state_io.feed_synthetic_ssm_state(model, saved_state), cache_position=cache_position)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = end_time - start_time
    return latency

def print_benchmark_results(baseline_latency: float, state_management_latency: float):
    print(f"Baseline Latency: {baseline_latency*1000:.4f} milliseconds")
    print(f"State Management Latency: {state_management_latency*1000:.4f} milliseconds")