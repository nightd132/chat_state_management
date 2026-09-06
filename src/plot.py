import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path




def _ensure_dir(path):
    """Create the parent directory for a plot path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)




def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to numeric while tolerating empty values from saved CSV exports."""
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned




def _normalise_comparison_data(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Convert a wide experiment result table to plotting rows."""
    rows = []
    if experiment == "experiment1":
        df = _coerce_numeric_columns(
            df,
            [
                "turn",
                "baseline_ppl_mean", "baseline_ppl_std",
                "state_ppl_mean", "state_ppl_std",
                "baseline_latency_mean", "baseline_latency_std",
                "state_latency_mean", "state_latency_std",
                "txt_size_kb_mean", "txt_size_kb_std",
                "pt_size_kb_mean", "pt_size_kb_std",
            ],
        )
        rows = []
        for _, row in df.iterrows():
            for condition, prefix in (("baseline", "baseline"), ("state", "state")):
                rows.append({
                    "condition": condition,
                    "step": row["turn"],
                    "ppl_mean": row[f"{prefix}_ppl_mean"],
                    "ppl_std": row[f"{prefix}_ppl_std"],
                    "latency_mean": row[f"{prefix}_latency_mean"],
                    "latency_std": row[f"{prefix}_latency_std"],
                    "size_kb_mean": row[
                        "txt_size_kb_mean" if condition == "baseline" else "pt_size_kb_mean"
                    ],
                    "size_kb_std": row[
                        "txt_size_kb_std" if condition == "baseline" else "pt_size_kb_std"
                    ],
                })
    elif experiment == "experiment2":
        for _, row in df.iterrows():
            condition = "baseline" if row["method"] == "baseline" else str(row["method"])
            source = "state" if condition == "baseline" else "compressed"
            rows.append({
                "condition": condition,
                "step": row["turn"],
                "ppl_mean": row[f"{source}_ppl_mean"],
                "ppl_std": row[f"{source}_ppl_std"],
                "latency_mean": row[f"{source}_latency_mean"],
                "latency_std": row[f"{source}_latency_std"],
                "size_kb_mean": row[
                    "original_size_kb_mean" if condition == "baseline"
                    else "compressed_size_kb_mean"
                ],
                "size_kb_std": row[
                    "original_size_kb_std" if condition == "baseline"
                    else "compressed_size_kb_std"
                ],
            })
    else:
        raise ValueError(f"Unsupported comparison experiment: {experiment}")
    return pd.DataFrame(rows)




def _plot_normalised_metric(
    data: pd.DataFrame,
    metric: str,
    plot_path: str,
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> None:
    """Plot one normalized metric with mean and standard-deviation bands."""
    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False
    for condition in sorted(data["condition"].unique()):
        group = data[data["condition"] == condition].sort_values("step")
        # Keep the x/mean/std arrays aligned while dropping incomplete rows.
        if f"{metric}_mean" not in group.columns:
            continue
        plot_data = pd.DataFrame({
            "step": pd.to_numeric(group["step"], errors="coerce"),
            "mean": pd.to_numeric(group[f"{metric}_mean"], errors="coerce"),
            "std": pd.to_numeric(
                group.get(f"{metric}_std", pd.Series(index=group.index)),
                errors="coerce",
            ),
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["step", "mean"])
        if log_scale:
            # Zero and negative values cannot be represented on a log axis.
            plot_data = plot_data[plot_data["mean"] > 0]
        if plot_data.empty:
            continue


        plot_data["std"] = plot_data["std"].fillna(0).clip(lower=0)
        x = plot_data["step"].to_numpy(dtype=float)
        mean = plot_data["mean"].to_numpy(dtype=float)
        std = plot_data["std"].to_numpy(dtype=float)
        # Compute symmetric bands in linear space, or multiplicative bands
        # when plotting on a log scale. For log-scale plots we approximate
        # the standard deviation on the log scale via the delta method:
        # Var(log X) ≈ Var(X) / E[X]^2  =>  std_log ≈ std / mean
        # and then form bands as mean * exp(±std_log).
        if log_scale:
            with np.errstate(divide='ignore', invalid='ignore'):
                std_log = np.where(mean > 0, std / mean, 0.0)
            lower = mean * np.exp(-std_log)
            upper = mean * np.exp(std_log)
            # Keep the band strictly positive for log plots.
            lower = np.maximum(lower, np.finfo(float).tiny)
        else:
            lower = mean - std
            upper = mean + std


        ax.plot(x, mean, marker="o", label=condition)
        ax.fill_between(x, lower, upper, alpha=0.15)
        any_plotted = True
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any_plotted:
        ax.legend(fontsize=8)
    ax.grid(True, which="both" if log_scale else "major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)




#Experiment 1 plots


def plot_ppl_comparison(df: pd.DataFrame, plot_path: str):
    """Plot Experiment 1 perplexity for baseline and injected state."""
    data = _normalise_comparison_data(df, "experiment1")
    _plot_normalised_metric(
        data, "ppl", plot_path,
        "Perplexity Comparison: Baseline vs Injected State",
        "Perplexity",
        log_scale=True,
    )




def plot_latency_comparison(df: pd.DataFrame, plot_path: str):
    """Plot Experiment 1 latency for baseline and injected state."""
    data = _normalise_comparison_data(df, "experiment1")
    _plot_normalised_metric(
        data, "latency", plot_path,
        "Latency Comparison: Baseline vs Injected State",
        "Latency (seconds)",
    )




def plot_memory_growth(df: pd.DataFrame, plot_path: str):
    """Plot Experiment 1 storage growth for text and recurrent state."""
    data = _normalise_comparison_data(df, "experiment1")
    _plot_normalised_metric(
        data, "size_kb", plot_path,
        "Storage Comparison: Text History vs Recurrent State",
        "State size (KB)",
    )




def plot_speedup(df: pd.DataFrame, plot_path: str):
    """Plot baseline-to-state latency speedup by turn."""
    df = _coerce_numeric_columns(
        df,
        [
            "turn",
            "baseline_latency_mean",
            "state_latency_mean",
        ],
    )
    df = df.dropna(subset=["turn", "baseline_latency_mean", "state_latency_mean"]).copy()
    df = df[df["state_latency_mean"] != 0]


    _ensure_dir(plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))


    turns = df["turn"].values
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
    """Select and sort one Experiment 2 method by turn."""
    return df[df["method"] == method].sort_values("turn")
 
 
def plot_perplexity_comparison(df: pd.DataFrame, plot_dir: str):
    """Plot Experiment 2 perplexity for all compression conditions."""
    plot_path = f"{plot_dir}/perplexity_comparison_exp2.png"
    data = _normalise_comparison_data(df, "experiment2")
    _plot_normalised_metric(
        data, "ppl", plot_path,
        "Perplexity Comparison: Baseline vs Compression Methods",
        "Perplexity (log scale)", log_scale=True,
    )


 
 
def plot_latency_comparison_exp2(df: pd.DataFrame, plot_dir: str):
    """Plot Experiment 2 latency for all compression conditions."""
    plot_path = f"{plot_dir}/latency_comparison_exp2.png"
    data = _normalise_comparison_data(df, "experiment2")
    _plot_normalised_metric(
        data, "latency", plot_path,
        "Latency Comparison: Baseline vs Compression Methods",
        "Latency (seconds)",
    )
 
 
def plot_memory_growth_exp2(df: pd.DataFrame, plot_dir: str):
    """Plot Experiment 2 storage size for all compression conditions."""
    plot_path = f"{plot_dir}/memory_growth_exp2.png"
    data = _normalise_comparison_data(df, "experiment2")
    _plot_normalised_metric(
        data, "size_kb", plot_path,
        "Storage Comparison: Baseline vs Compression Methods",
        "State size (KB)",
    )






def plot_recovery_curve(df: pd.DataFrame, plot_dir: str):
    """Plot perplexity recovery as distance from a session boundary grows."""
    plot_path = f"{plot_dir}/experiment3_recovery_curve.png"
    data = df.rename(
        columns={
            "label": "condition",
            "offset": "step",
            "state_ppl_mean": "ppl_mean",
            "state_ppl_std": "ppl_std",
        }
    )
    _plot_normalised_metric(
        data, "ppl", plot_path,
        "Perplexity Recovery after Session Boundary",
        "Perplexity",
        log_scale=True,
    )




def plot_boundary_drift(
    df: pd.DataFrame,
    plot_dir: str,
    experiment_name: str = "experiment3",
):
    """Plot first-turn perplexity drift across chained sessions."""
    plot_path = f"{plot_dir}/{experiment_name}_boundary_drift.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))


    for label in sorted(df["label"].unique()):
        group = df[df["label"] == label].sort_values("session_id")
        ax.plot(group["session_id"], group["boundary_ppl"], marker="x", label=label, alpha=0.8)


    ax.set_xlabel("Session index in chain")
    ax.set_ylabel("Perplexity at first turn of session (session boundary)")
    ax.set_title(
        f"Boundary perplexity drift ({experiment_name})"
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)