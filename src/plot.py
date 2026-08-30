import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


#Experiment 1 plots

def plot_ppl_comparison(df: pd.DataFrame, plot_path: str):
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    turns = df["turn"].values

    ax.plot(turns, df["baseline_ppl_mean"], marker="o", label="Baseline (full history)")
    ax.fill_between(turns,
                    df["baseline_ppl_mean"] - df["baseline_ppl_std"],
                    df["baseline_ppl_mean"] + df["baseline_ppl_std"],
                    alpha=0.2)

    ax.plot(turns, df["state_ppl_mean"], marker="x", label="Injected state")
    ax.fill_between(turns,
                    df["state_ppl_mean"] - df["state_ppl_std"],
                    df["state_ppl_mean"] + df["state_ppl_std"],
                    alpha=0.2)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity: Baseline vs Injected State (mean ± std)")
    ax.legend()
    ax.grid(True)

    # Annotate n_sessions at last turn
    last = df.iloc[-1]
    ax.annotate(f"n={int(last['n_sessions'])} sessions",
                xy=(last["turn"], last["state_ppl_mean"]),
                xytext=(8, 8), textcoords="offset points", fontsize=8, color="grey")

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_latency_comparison(df: pd.DataFrame, plot_path: str):
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    turns = df["turn"].values

    ax.plot(turns, df["baseline_latency_mean"], marker="o", label="Baseline (full history)")
    ax.fill_between(turns,
                    df["baseline_latency_mean"] - df["baseline_latency_std"],
                    df["baseline_latency_mean"] + df["baseline_latency_std"],
                    alpha=0.2)

    ax.plot(turns, df["state_latency_mean"], marker="x", label="Injected state")
    ax.fill_between(turns,
                    df["state_latency_mean"] - df["state_latency_std"],
                    df["state_latency_mean"] + df["state_latency_std"],
                    alpha=0.2)

    # Mark crossover point if it exists
    bl = df["baseline_latency_mean"].values
    st = df["state_latency_mean"].values
    for i in range(len(turns) - 1):
        if (bl[i] - st[i]) * (bl[i+1] - st[i+1]) < 0:  # sign change = crossover
            cross_turn = (turns[i] + turns[i+1]) / 2
            ax.axvline(cross_turn, color="red", linestyle="--", alpha=0.6,
                       label=f"Crossover ~turn {cross_turn:.0f}")
            break

    ax.set_xlabel("Turn")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency: Baseline vs Injected State (mean ± std)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_memory_growth(df: pd.DataFrame, plot_path: str):
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    turns = df["turn"].values

    ax.plot(turns, df["txt_size_kb_mean"], marker="o", label="Text history (.txt)")
    ax.fill_between(turns,
                    df["txt_size_kb_mean"] - df["txt_size_kb_std"],
                    df["txt_size_kb_mean"] + df["txt_size_kb_std"],
                    alpha=0.2)

    ax.plot(turns, df["pt_size_kb_mean"], marker="x", label="Recurrent state (.pt)")
    ax.fill_between(turns,
                    df["pt_size_kb_mean"] - df["pt_size_kb_std"],
                    df["pt_size_kb_mean"] + df["pt_size_kb_std"],
                    alpha=0.2)

    # Mark storage crossover
    txt = df["txt_size_kb_mean"].values
    pt  = df["pt_size_kb_mean"].values
    for i in range(len(turns) - 1):
        if (txt[i] - pt[i]) * (txt[i+1] - pt[i+1]) < 0:
            cross_turn = (turns[i] + turns[i+1]) / 2
            ax.axvline(cross_turn, color="red", linestyle="--", alpha=0.6,
                       label=f"Crossover ~turn {cross_turn:.0f}")
            break

    ax.set_xlabel("Turn")
    ax.set_ylabel("Size (KB)")
    ax.set_title("Storage: Text History vs Recurrent State (mean ± std)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_speedup(df: pd.DataFrame, plot_path: str):
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    turns   = df["turn"].values
    speedup = df["baseline_latency_mean"].values / df["state_latency_mean"].values

    ax.plot(turns, speedup, marker="o", color="green", label="Speedup (baseline / state)")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Break-even (1×)")
    ax.fill_between(turns, 1.0, speedup,
                    where=(speedup >= 1.0), alpha=0.15, color="green",
                    label="State faster")
    ax.fill_between(turns, 1.0, speedup,
                    where=(speedup < 1.0),  alpha=0.15, color="red",
                    label="State slower")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Latency Speedup of Injected State over Baseline")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


# Experiment2 plot

def _sorted_by_turn(df, method):
    return df[df["method"] == method].sort_values("turn")
 
 
def plot_perplexity_comparison(df: pd.DataFrame, plot_dir: str):
    plot_path = f"{plot_dir}/perplexity_comparison_exp2.png"
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = sorted(df["method"].unique())
    baseline = _sorted_by_turn(df, methods[0])

    ax.plot(baseline["turn"], baseline["state_ppl_mean"],
            marker="o", linestyle="--", color="black", linewidth=2,
            label="Original (no compression)")

    for method in methods:
        group = _sorted_by_turn(df, method)
        ax.plot(group["turn"], group["compressed_ppl_mean"], marker="x", label=method)

    ax.set_yscale("log")
    ax.set_title("Perplexity Comparison: Original vs Compressed States")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Perplexity (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

 
 
def plot_latency_comparison_exp2(df: pd.DataFrame, plot_dir: str):
    plot_path = f"{plot_dir}/latency_comparison_exp2.png"
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))
 
    methods = sorted(df["method"].unique())
    baseline = _sorted_by_turn(df, methods[0])
 
    ax.plot(baseline["turn"], baseline["state_latency_mean"],
            marker="o", linestyle="--", color="black", label="Baseline (no compression)")
    ax.fill_between(baseline["turn"],
                    baseline["state_latency_mean"] - baseline["state_latency_std"],
                    baseline["state_latency_mean"] + baseline["state_latency_std"],
                    alpha=0.15, color="black")
 
    for method in methods:
        group = _sorted_by_turn(df, method)
        ax.plot(group["turn"], group["compressed_latency_mean"], marker="x", label=method)
        ax.fill_between(group["turn"],
                        group["compressed_latency_mean"] - group["compressed_latency_std"],
                        group["compressed_latency_mean"] + group["compressed_latency_std"],
                        alpha=0.15)
 
    ax.set_title("Latency Comparison Across Compression Methods (mean ± std)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Latency (seconds)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
 
 
def plot_memory_growth_exp2(df: pd.DataFrame, plot_dir: str):
    plot_path = f"{plot_dir}/memory_growth_exp2.png"
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))
 
    methods = sorted(df["method"].unique())
    baseline = _sorted_by_turn(df, methods[0])
 
    ax.plot(baseline["turn"], baseline["original_size_kb_mean"],
            marker="o", linestyle="--", color="black", label="Original State")
    ax.fill_between(baseline["turn"],
                    baseline["original_size_kb_mean"] - baseline["original_size_kb_std"],
                    baseline["original_size_kb_mean"] + baseline["original_size_kb_std"],
                    alpha=0.15, color="black")
 
    for method in methods:
        group = _sorted_by_turn(df, method)
        ax.plot(group["turn"], group["compressed_size_kb_mean"], marker="x", label=method)
        ax.fill_between(group["turn"],
                        group["compressed_size_kb_mean"] - group["compressed_size_kb_std"],
                        group["compressed_size_kb_mean"] + group["compressed_size_kb_std"],
                        alpha=0.15)
 
    ax.set_xlabel("Turn")
    ax.set_ylabel("Memory Size (KB)")
    ax.set_title("Memory Growth: Original vs Compressed State (mean ± std)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)



def plot_recovery_curve(df: pd.DataFrame, plot_dir: str):
    plot_path = f"{plot_dir}/experiment3_recovery_curve.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label in sorted(df["label"].unique()):
        group = df[df["label"] == label].sort_values("offset")
        ax.plot(group["offset"], group["state_ppl_mean"], marker="o", label=label)
        ax.fill_between(
            group["offset"],
            group["state_ppl_mean"] - group["state_ppl_std"],
            group["state_ppl_mean"] + group["state_ppl_std"],
            alpha=0.15,
        )

    ax.set_xlabel("Turns since session boundary")
    ax.set_ylabel("Perplexity")
    ax.set_title("Recovery after session-boundary state carryover (mean ± std)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_boundary_drift(df: pd.DataFrame, plot_dir: str):
    plot_path = f"{plot_dir}/experiment3_boundary_drift.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label in sorted(df["label"].unique()):
        group = df[df["label"] == label].sort_values("session_id")
        ax.plot(group["session_id"], group["boundary_ppl"], marker="x", label=label, alpha=0.8)

    ax.set_xlabel("Session index in chain")
    ax.set_ylabel("Perplexity at first turn of session (session boundary)")
    ax.set_title("Does boundary perplexity drift as decayed carryover compounds?")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)