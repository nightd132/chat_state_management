from src import model_loader, state_utils, evaluate as evaluate_module, data, utils, plot
from src.mamba2_stateful import patch_model
import torch
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np


def apply_forgetting_factor(ssm_states: torch.Tensor, alpha: float):
    return ssm_states * alpha


def run_state_chain(model, tokenizer, sessions, device, state_dir, alpha=None):
    output_data = {}
    carryover_ssm, carryover_conv = None, None

    for session_id, session in enumerate(sessions):
        snapshots = data.build_turn_snapshots(session)

        for snap in snapshots:
            turn_id = snap["turn_id"]
            turns_since_boundary = turn_id 

            if turn_id == 0 and carryover_ssm is not None and alpha is not None:
                seed_ssm = apply_forgetting_factor(carryover_ssm, alpha)
                seed_conv = carryover_conv
                ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                    model, tokenizer, snap["new_text"], seed_ssm, seed_conv, device=device
                )
            elif turn_id == 0:
                state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                    model, tokenizer, snap["history_text"], snap["new_text"], device=device
                )
                ssm_states, conv_states = state_utils.extract_state(state_output)
            else:
                prev_ssm, prev_conv = state_utils.load_state(state_dir, device="cpu")
                ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                    model, tokenizer, snap["new_text"], prev_ssm, prev_conv, device=device
                )

            # Save state to disk as CPU tensors to avoid keeping GPU copies
            state_utils.save_state(ssm_states, conv_states, state_dir)

            # Ensure we keep only CPU copies for carryover between sessions
            carryover_ssm, carryover_conv = ssm_states.cpu(), conv_states.cpu()

            # Free local references to large tensors and clear CUDA cache
            try:
                del ssm_states
                del conv_states
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if snap["role"] == "assistant":
                output_data[(session_id, turn_id)] = {
                    "state_latency": state_latency,
                    "state_size_kb": utils.get_memory_size_kb(state_dir),
                    "state_ppl": state_ppl,
                    "turns_since_boundary": turns_since_boundary,
                    "is_first_session": session_id == 0,
                }

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



def run_experiment_3(
    model, tokenizer, sessions,
    output_dir, experiment_3_benchmark_path, plot_dir,
    device, alpha_list=(None, 1.0, 0.999, 0.99, 0.9),
):
    all_agg = {}           # label -> {offset: agg metrics}
    all_boundary_seq = {}  # label -> [(session_id, ppl), ...]

    for alpha in alpha_list:
        label = "cold_start" if alpha is None else f"alpha_{alpha}"
        print(f"\n{'='*50}")
        print(f"Experiment 3: running chain with {label}")
        print(f"{'='*50}")

        state_dir = f"{output_dir}/state_exp3_{label}.pt"

        # Cache chain results to avoid rerunning expensive cold-start sessions
        cache_dir = Path(output_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"exp3_{label}_chain.pt"
        if cache_path.exists():
            print(f"[cache] loading cached results for {label} from {cache_path}")
            output_data = torch.load(cache_path)
        else:
            output_data = run_state_chain(model, tokenizer, sessions, device, state_dir, alpha=alpha)
            torch.save(output_data, cache_path)

        all_agg[label] = aggregate_by_boundary_offset(output_data)
        all_boundary_seq[label] = aggregate_boundary_only_by_session(output_data)

        print(f"Experiment 3: finished {label}")
        torch.cuda.empty_cache()

    with open(experiment_3_benchmark_path, "w", newline="") as f:
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

    boundary_csv_path = str(Path(experiment_3_benchmark_path).with_name("experiment3_boundary_sequence.csv"))
    with open(boundary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "session_id", "boundary_ppl"])
        for label, rows in all_boundary_seq.items():
            for session_id, ppl in rows:
                writer.writerow([label, session_id, ppl])

    df = pd.read_csv(experiment_3_benchmark_path)
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
    experiment3_path = output_dir + "/experiment3/experiment3.csv"

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
    print(f"Running Experiment 3 on a chain of {len(test_sessions)} test sessions")

    alpha_list = [None,1.0, 0.9, 0.5, 0.1]

    df, boundary_df = run_experiment_3(
        model, tokenizer, test_sessions,
        output_dir + "/experiment3",
        experiment3_path,
        plot_dir + "/experiment3",
        device,
        alpha_list=alpha_list,
    )


if __name__ == "__main__":
    main()


