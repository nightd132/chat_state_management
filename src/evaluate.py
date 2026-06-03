import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from pathlib import Path
from torch.nn import functional as F

from src import state_utils
def evaluate_baseline(model, tokenizer, history_text: str, input_text: str, device: str = "cpu"):
    inputs = tokenizer(history_text, return_tensors="pt").to(device)
    prefix = tokenizer(history_text[:-len(input_text)], return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    prefix_len = prefix["input_ids"].shape[1]
    labels[0,:prefix_len] = -100
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs, labels=labels)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    loss = output.loss.item() if output.loss is not None else 0.0
    perplexity = torch.exp(torch.tensor(loss)).item() if loss > 0 else 0.0

    return output, latency, perplexity

def evaluate(model, tokenizer, input_text: str, saved_state, device: str = "cpu"):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    cache = None
    cache = state_utils.create_cache(device=device)
    total_loss = 0
    for i, ssm_state in enumerate(saved_state):
        cache.layers[i].recurrent_states = ssm_state.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        
        # output = model(**inputs, cache_params=cache, labels=inputs["input_ids"].to(device))
        for i in range(len(input_ids[0])-1):
            token = input_ids[:, i:i+1]
            output = model(token, cache_params=cache, use_cache=True)
            logits = output.logits[:, -1]
            loss = F.cross_entropy(logits, input_ids[:, :i+1])
            total_loss += loss
            cache = output.cache_params
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency = end_time - start_time
    # loss = output.loss.item() if output.loss is not None else 0.0
    loss = total_loss/(len(input_ids[0])-1) if len(input_ids[0])-1>0 else total_loss
    print(loss)
    perplexity = torch.exp(loss.detach().clone()).item() if loss > 0 else 0.0

    return output, latency, perplexity

def evaluate_injected_mode(model, tokenizer, input_text,saved_states, device=None, dtype=None):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    curr_ids = inputs["input_ids"]
    fresh_cache = state_utils.load_recurrent_states(saved_states, model, device, dtype)
    total_loss  = 0.0
    num_tokens  = 0
    total_inference_time = 0.0
    out = None

    
    # Feed token by token so the recurrent single-step path is valid
    for i in range(curr_ids.shape[1] - 1):
        token_in  = curr_ids[:, i:i+1]       # [1, 1]
        token_tgt = curr_ids[:, i+1:i+2]     # [1, 1] — next token is the label
        if torch.cuda.is_available():
            torch.cuda.synchronize() 
        start_time = time.perf_counter()

        with torch.no_grad():
            out = model(token_in, cache_params=fresh_cache, use_cache=True, return_dict=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Loss for this single step
        logits = out.logits[:, -1, :]         # [1, vocab]
        loss   = F.cross_entropy(logits, token_tgt.squeeze(1)).item()
        total_loss += loss
        num_tokens += 1
        total_inference_time += end_time-start_time
        # Cache is updated in-place by the model, carry it forward
        fresh_cache = out.cache_params

    avg_loss = total_loss / num_tokens if num_tokens > 0 else 0.0
    ppl      = math.exp(avg_loss)
    return out, total_inference_time, ppl
