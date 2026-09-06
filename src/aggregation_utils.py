import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def write_experiment_summary_csv(path, agg_map):
    """Write aggregated PPL, latency, and size metrics for each condition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "offset", "n",
            "state_ppl_mean", "state_ppl_std",
            "state_latency_mean", "state_latency_std",
            "state_size_kb_mean", "state_size_kb_std",
        ])
        for label, agg in agg_map.items():
            for offset in sorted(agg):
                a = agg[offset]
                writer.writerow([
                    label, offset, a.get("n", 0),
                    a.get("state_ppl_mean", ""), a.get("state_ppl_std", ""),
                    a.get("state_latency_mean", ""), a.get("state_latency_std", ""),
                    a.get("state_size_kb_mean", ""), a.get("state_size_kb_std", ""),
                ])


def write_boundary_sequence_csv(path, boundary_map):
    """Write boundary perplexity rows for each condition and session."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "session_id", "boundary_ppl"])
        for label, rows in boundary_map.items():
            for session_id, ppl in rows:
                writer.writerow([label, session_id, ppl])


def aggregate_by_boundary_offset(
    output_data: Dict,
    exclude_first_session: bool = True
) -> Dict[int, Dict[str, float]]:
    """Group state metrics by distance from a session boundary."""
    bucket = defaultdict(lambda: defaultdict(list))
    
    for (_session_id, _turn_id), metrics in output_data.items():
        if exclude_first_session and metrics["is_first_session"]:
            continue
        
        offset = metrics["turns_since_boundary"]
        for k in ("state_ppl", "state_latency", "state_size_kb"):
            bucket[offset][k].append(metrics[k])
    
    agg = {}
    for offset, metric_lists in bucket.items():
        agg[offset] = {}
        n = None
        for k, values in metric_lists.items():
            agg[offset][f"{k}_mean"] = float(np.mean(values))
            agg[offset][f"{k}_std"] = float(np.std(values))
            n = len(values)
        agg[offset]["n"] = n
    
    return agg

def aggregate_boundary_only_by_session(
    output_data: Dict
) -> List[Tuple[int, float]]:
    """Extract the first assistant-turn perplexity for each non-initial session."""
    by_session = defaultdict(list)
    for (session_id, turn_id), metrics in sorted(output_data.items()):
        if metrics.get("is_first_session"):
            continue
        by_session[session_id].append((turn_id, metrics["state_ppl"]))

    rows = []
    for session_id, values in sorted(by_session.items()):
        if not values:
            continue
        _, boundary_ppl = min(values, key=lambda item: item[0])
        rows.append((session_id, boundary_ppl))
    return rows

def summarize_boundary_health(
    df,
    threshold: float = 1.5
) -> None:
    """Print degradation ratios relative to the baseline boundary metric."""
    baseline = df[(df["label"] == "baseline") & (df["offset"] == 0)]
    if baseline.empty:
        print("No cold_start baseline found at offset=0 -- skipping summary.")
        return
    
    baseline_ppl = baseline["state_ppl_mean"].iloc[0]
    
    print_section("Boundary-turn perplexity vs cold_start baseline")
    print(f"Baseline PPL: {baseline_ppl:.3f}\n")
    
    for label in sorted(df["label"].unique()):
        if label == "baseline":
            continue
        row = df[(df["label"] == label) & (df["offset"] == 0)]
        if row.empty:
            continue
        
        ppl = row["state_ppl_mean"].iloc[0]
        ratio = ppl / baseline_ppl
        status = "OK" if ratio <= threshold else "DEGRADED"
        print(f"  [{status}] {label:20s}  ppl={ppl:8.3f}  ratio={ratio:5.2f}x")

def print_section(title: str, width: int = 60) -> None:
    """Print formatted section header"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")