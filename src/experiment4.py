from src import model_loader, state_utils, evaluate as evaluate_module, data, plot, utils
from src.mamba2_stateful import patch_model
import torch
import pandas as pd
import csv
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np


def _label_for_beta(beta):
    return "cold_start" if beta is None else f"ema_beta_{beta}"


def _cache_path(output_dir, label):
    return Path(output_dir) / "cache" / f"exp4_{label}.pt"


def load_or_run_chain(model, tokenizer, sessions, device, output_dir, beta, force_rerun=False):
    label = _label_for_beta(beta)
    cache_path = _cache_path(output_dir, label)

    if cache_path.exists() and not force_rerun:
        print(f"Experiment 4: loading cached chain for {label} from {cache_path}")
        output_data = torch.load(cache_path, map_location=device)
    else:
        print(f"\n{'='*50}")
        print(f"Experiment 4: running chain with {label}")
        print(f"{'='*50}")
        state_dir = f"{output_dir}/state_exp4_{label}.pt"
        output_data = run_state_chain_ema(model, tokenizer, sessions, device, state_dir, beta=beta)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_data, cache_path)

    return output_data


def ema_update(running_mean: torch.Tensor, new_state: torch.Tensor, beta: float):
    if running_mean is None:
        return new_state.clone()
    return beta * running_mean + (1 - beta) * new_state


def run_state_chain_ema(model, tokenizer, sessions, device, state_dir, beta=None):
    output_data = {}
    running_mean_ssm, running_mean_conv = None, None

    for session_id, session in enumerate(sessions):
        snapshots = data.build_turn_snapshots(session)

        for snap in snapshots:
            turn_id = snap["turn_id"]
            turns_since_boundary = turn_id

            if turn_id == 0 and running_mean_ssm is not None and beta is not None:
                ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                    model, tokenizer, snap["new_text"], running_mean_ssm, running_mean_conv, device=device
                )
            elif turn_id == 0:
                state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                    model, tokenizer, snap["history_text"], snap["new_text"], device=device
                )
                ssm_states, conv_states = state_utils.extract_state(state_output)
            else:
                prev_ssm, prev_conv = state_utils.load_state(state_dir, device=device)
                ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                    model, tokenizer, snap["new_text"], prev_ssm, prev_conv, device=device
                )

            state_utils.save_state(ssm_states, conv_states, state_dir)

            if snap["role"] == "assistant":
                output_data[(session_id, turn_id)] = {
                    "state_latency": state_latency,
                    "state_size_kb": utils.get_memory_size_kb(state_dir),
                    "state_ppl": state_ppl,
                    "turns_since_boundary": turns_since_boundary,
                    "is_first_session": session_id == 0,
                }

        if beta is not None:
            running_mean_ssm = ema_update(running_mean_ssm, ssm_states, beta)
            running_mean_conv = conv_states

    return output_data


def aggregate_by_boundary_offset(output_data, exclude_first_session=True):
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


def aggregate_boundary_only_by_session(output_data):
    rows = []
    for (session_id, turn_id), metrics in sorted(output_data.items()):
        if turn_id == 0 and not metrics["is_first_session"]:
            rows.append((session_id, metrics["state_ppl"]))
    return rows


def summarize_boundary_health(df: pd.DataFrame, threshold=1.5):
    baseline = df[(df["label"] == "cold_start") & (df["offset"] == 0)]
    if baseline.empty:
        print("No cold_start baseline found at offset=0 -- skipping summary.")
        return
    baseline_ppl = baseline["state_ppl_mean"].iloc[0]

    print(f"\n{'='*50}")
    print(f"Boundary-turn perplexity vs cold_start baseline ({baseline_ppl:.3f})")
    print(f"{'='*50}")
    for label in sorted(df["label"].unique()):
        if label == "cold_start":
            continue
        row = df[(df["label"] == label) & (df["offset"] == 0)]
        if row.empty:
            continue
        ppl = row["state_ppl_mean"].iloc[0]
        ratio = ppl / baseline_ppl
        status = "OK" if ratio <= threshold else "DEGRADED"
        print(f"  [{status}] {label:12s}  ppl={ppl:8.3f}  ratio={ratio:5.2f}x")


def run_experiment_4(
    model, tokenizer, sessions,
    output_dir, experiment_4_benchmark_path, plot_dir,
    device, beta_list=(None, 0.0, 0.3, 0.5, 0.7, 0.9),
    force_rerun_labels=None,
):
    valid_labels = {_label_for_beta(b) for b in beta_list}
    force_rerun_labels = set(force_rerun_labels or [])
    unknown = force_rerun_labels - valid_labels
    if unknown:
        raise ValueError(
            f"force_rerun_labels contains labels not in beta_list: {unknown}. "
            f"Valid labels for this sweep: {sorted(valid_labels)}"
        )

    all_agg = {}
    all_boundary_seq = {}

    for beta in beta_list:
        label = _label_for_beta(beta)
        output_data = load_or_run_chain(
            model, tokenizer, sessions, device, output_dir, beta,
            force_rerun=label in force_rerun_labels,
        )

        all_agg[label] = aggregate_by_boundary_offset(output_data)
        all_boundary_seq[label] = aggregate_boundary_only_by_session(output_data)

        torch.cuda.empty_cache()

    with open(experiment_4_benchmark_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "offset", "n",
            "state_ppl_mean", "state_ppl_std",
            "state_latency_mean", "state_latency_std",
            "state_size_kb_mean", "state_size_kb_std",
        ])
        for label, agg in all_agg.items():
            for offset in sorted(agg):
                a = agg[offset]
                writer.writerow([
                    label, offset, a.get("n", 0),
                    a.get("state_ppl_mean", ""), a.get("state_ppl_std", ""),
                    a.get("state_latency_mean", ""), a.get("state_latency_std", ""),
                    a.get("state_size_kb_mean", ""), a.get("state_size_kb_std", ""),
                ])

    boundary_csv_path = str(Path(experiment_4_benchmark_path).with_name("experiment4_boundary_sequence.csv"))
    with open(boundary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "session_id", "boundary_ppl"])
        for label, rows in all_boundary_seq.items():
            for session_id, ppl in rows:
                writer.writerow([label, session_id, ppl])

    df = pd.read_csv(experiment_4_benchmark_path)
    boundary_df = pd.read_csv(boundary_csv_path)

    plot.plot_recovery_curve(df, plot_dir)
    plot.plot_boundary_drift(boundary_df, plot_dir)
    summarize_boundary_health(df)

    return df, boundary_df


def main():
    torch.manual_seed(0)
    config = utils.read_config("configs/config1.yaml")

    paths = config["paths"]
    root = Path(__file__).parent.parent
    output_dir = str(root) + "/" + paths["output_dir"]
    plot_dir = str(root) + "/" + paths["plot_dir"]
    experiment4_path = output_dir + "/experiment4/experiment4.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )
    patch_model(model)

    dataset = data.load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = data.extract_sessions(dataset)

    _, _, test_sessions = data.split_sessions(sessions, train=0.70, val=0.15, test=0.15)
    print(f"Running Experiment 4 on a chain of {len(test_sessions)} test sessions")

    beta_list = [None, 0.0, 0.3, 0.5, 0.7, 0.9]
    # Add labels here (e.g. ["ema_beta_0.5"]) to force-recompute specific
    # arms even if a cached chain exists for them; leave empty to reuse
    # everything cached and only compute newly-added beta values.
    force_rerun_labels = []

    df, boundary_df = run_experiment_4(
        model, tokenizer, test_sessions,
        output_dir + "/experiment4",
        experiment4_path,
        plot_dir + "/experiment4",
        device,
        beta_list=beta_list,
        force_rerun_labels=force_rerun_labels,
    )


if __name__ == "__main__":
    main()


