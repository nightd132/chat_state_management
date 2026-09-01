import time
import math
import torch
import torch.nn.functional as F

from src import state_utils
from src.mamba2_stateful import run_forward_with_states


def evaluate_baseline(model, tokenizer, history_text, input_text, device="cpu"):
    """Evaluate new tokens by running the model on the full text history."""
    full_ids    = tokenizer(history_text, return_tensors="pt").input_ids.to(device)
    new_ids     = tokenizer(input_text,   return_tensors="pt").input_ids.to(device)
    prefix_len  = full_ids.shape[1] - new_ids.shape[1]

    labels = full_ids.clone()
    labels[0, :prefix_len] = -100   # mask everything except new tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model(input_ids=full_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    shift_logits = output.logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    ).item()
    ppl  = math.exp(loss) if loss > 0 else 0.0
    torch.cuda.empty_cache()
    return output, latency, ppl
    


def evaluate_injected(model, tokenizer, new_text: str, ssm_states, conv_states, device: str = "cpu"):
    """Evaluate new tokens using previously captured recurrent states."""

    inputs = tokenizer(new_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # (1, L)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        logits, new_ssm, new_conv = run_forward_with_states(
            model, input_ids, ssm_states, conv_states
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start_time

    # PPL: shift logits/labels by 1 (standard LM loss)
    shift_logits = logits[:, :-1].contiguous()           # (1, L-1, vocab)
    shift_labels = input_ids[:, 1:].contiguous()         # (1, L-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    perplexity = math.exp(loss.item())
    torch.cuda.empty_cache()
    return new_ssm, new_conv, latency, perplexity
