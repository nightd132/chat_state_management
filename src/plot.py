import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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

def plot_memory_growth(size_data, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(size_data["turn"], size_data["txt_size_kb"], marker='o', label="Text History Size (KB)")
    plt.plot(size_data["turn"], size_data["pt_size_kb"], marker='x', label="State Size (KB)")
    plt.xlabel("Turn")
    plt.ylabel("Memory Size (KB)")
    plt.title("Memory Growth: Text History vs State")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def plot_latency_comparison(latency_data, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(latency_data["turn"], latency_data["baseline_latency"], marker='o', label="Baseline Latency")
    plt.plot(latency_data["turn"], latency_data["state_latency"], marker='x', label="State Management Latency")
    plt.xlabel("Turn")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Comparison: Baseline vs State Management")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def plot_ppl_comparison(perplexity_data, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(perplexity_data["turn"], perplexity_data["baseline_ppl"], marker='o', label="Baseline Perplexity")
    plt.plot(perplexity_data["turn"], perplexity_data["state_ppl"], marker='x', label="State Management Perplexity")
    plt.xlabel("Turn")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Comparison: Baseline vs State Management")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()