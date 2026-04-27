import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from storage import state_io, history_text_io

def evaluate(model, tokenizer, input_text: str, saved_state, device: str = "cpu"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    cache = state_io.feed_synthetic_ssm_state(model, saved_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    start_time = time.perf_counter()
    with torch.no_grad():
        cache_position = torch.arange(inputs["input_ids"].size(1), device=device)
        output = model(**inputs, cache_params=cache, 
                       cache_position=cache_position, labels=inputs["input_ids"])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    loss = output.loss
    perplexity = torch.exp(loss)

    return output,latency, perplexity.item()

def plot_perplexity_comparison(perplexity_data, plot_dir):
    df = pd.DataFrame(perplexity_data)
    plt.figure(figsize=(10, 6))

    original = df[df["autoencoder_latent_dim"] == df["autoencoder_latent_dim"].iloc[0]]
    plt.plot(original["turn"], original["state_ppl"],
             marker='o', linestyle='--', color='black', label="Original State")

    for dim, group in df.groupby("autoencoder_latent_dim"):
        plt.plot(group["turn"], group["compressed_ppl"],
                 marker='x', label=f"Compressed (dim={dim})")

    plt.title("Perplexity Comparison: Original vs Compressed States")
    plt.xlabel("Turn")
    plt.ylabel("Perplexity")
    plt.xticks(df["turn"].unique())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/perplexity_comparison_exp2.png")
    plt.close()

def plot_latency_comparison_exp2(latency_data, plot_dir):
    df = pd.DataFrame(latency_data)
    plt.figure(figsize=(10, 6))
    
    original = df[df["autoencoder_latent_dim"] == df["autoencoder_latent_dim"].iloc[0]]
    plt.plot(original["turn"], original["state_latency"],
             marker='o', linestyle='--', color='black', label="Baseline Latency")

    for dim, group in df.groupby("autoencoder_latent_dim"):
        plt.plot(group["turn"], group["compressed_latency"],
                 marker='x', label=f"Compressed (dim={dim}) Latency")
    
    plt.title("Latency Comparison Across Autoencoder Latent Dimensions")
    plt.xlabel("Turn")
    plt.ylabel("Latency (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/latency_comparison_exp2.png")
    plt.close()

def plot_memory_growth_exp2(results, plot_dir):
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    original = df[df["autoencoder_latent_dim"] == df["autoencoder_latent_dim"].iloc[0]]
    plt.plot(original["turn"], original["original_size_kb"],
             marker='o', linestyle='--', color='black', label="Original State")

    for dim, group in df.groupby("autoencoder_latent_dim"):
        plt.plot(group["turn"], group["compressed_size_kb"],
                 marker='x', label=f"Compressed (dim={dim}) state")

    plt.xlabel("Turn")
    plt.ylabel("Memory Size (KB)")
    plt.title("Memory Growth: Original vs Compressed State")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/memory_growth_exp2.png")
    plt.close()