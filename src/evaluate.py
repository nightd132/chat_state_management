import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src import state_utils
def evaluate_baseline(model, tokenizer, input_text: str, device: str = "cpu"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs, labels=inputs["input_ids"].to(device))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    loss = output.loss.item() if output.loss is not None else 0.0
    perplexity = torch.exp(torch.tensor(loss)).item() if loss > 0 else 0.0

    return output, latency, perplexity

def evaluate(model, tokenizer, input_text: str, saved_state, device: str = "cpu"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    cache = state_utils.create_cache(device=device)
    for i, ssm_state in enumerate(saved_state):
        cache.layers[i].recurrent_states = ssm_state.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs, cache_params=cache, labels=inputs["input_ids"].to(device))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    loss = output.loss.item() if output.loss is not None else 0.0
    perplexity = torch.exp(torch.tensor(loss)).item() if loss > 0 else 0.0

    return output, latency, perplexity

# def evaluate_conv(model, tokenizer, input_text: str, saved_ssm_state, saved_conv_state, device: str = "cpu"):
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)
#     cache = state_utils.feed_synthetic_state(model, saved_ssm_state, saved_conv_state)
#     if torch.cuda.is_available():
#         torch.cuda.synchronize() 
#     start_time = time.perf_counter()
#     with torch.no_grad():
#         cache_position = torch.arange(len(inputs["input_ids"][0]), device=model.device)
#         output = model(**inputs, cache_params=cache, cache_position=cache_position, labels=inputs["input_ids"].to(device))
    
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     end_time = time.perf_counter()
    
#     latency = end_time - start_time
#     loss = output.loss.item() if output.loss is not None else 0.0
#     perplexity = torch.exp(torch.tensor(loss)).item() if loss > 0 else 0.0

#     return output,latency, perplexity