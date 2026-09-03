import csv
from pathlib import Path

import pandas as pd
import torch

from src import model_loader, state_utils, evaluate, data, utils, plot
from src.mamba2_stateful import patch_model
from src.aggregation_utils import (
    aggregate_by_boundary_offset,
    aggregate_boundary_only_by_session,
    summarize_boundary_health,
    write_boundary_sequence_csv,
    write_experiment_summary_csv,
)
from src.cache_utils import get_torch_cache_path, load_torch_cache, save_torch_cache
from src.log_utils import print_section, log_cache_load
from src.state_runner import StateRunner


def apply_forgetting_factor(ssm_states: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale recurrent states to reduce retained information."""
    return ssm_states * alpha


def run_state_chain(
    model,
    tokenizer,
    sessions,
    device,
    state_dir: str,
    alpha: float = None,
) -> dict:
    """Run the shared state chain with optional alpha carry-over."""
    runner = StateRunner(
        state_dir,
        state_seed=(
            lambda prev_ssm, prev_conv: (
                apply_forgetting_factor(prev_ssm, alpha),
                prev_conv,
            )
        )
        if alpha is not None
        else None,
        use_carryover=alpha is not None,
    )
    return runner.run_chain(model, tokenizer, sessions, device)

def run_experiment_3(
    model, tokenizer, sessions,
    output_dir, experiment_3_benchmark_path, plot_dir,
    device, alpha_list=(None, 1.0, 0.999, 0.99, 0.9),
):
    """Run Experiment 3 across forgetting-factor configurations."""
    all_agg = {}           # label -> {offset: agg metrics}
    all_boundary_seq = {}  # label -> [(session_id, ppl), ...]

    for alpha in alpha_list:
        label = "baseline" if alpha is None else f"carryover_alpha_{alpha}"
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

    write_experiment_summary_csv(experiment_3_benchmark_path, all_agg)

    boundary_csv_path = str(Path(experiment_3_benchmark_path).with_name("experiment3_boundary_sequence.csv"))
    write_boundary_sequence_csv(boundary_csv_path, all_boundary_seq)

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
    output_dir = Path(root) / paths["output_dir"]
    plot_dir = Path(root) / paths["plot_dir"]
    experiment3_path = output_dir / "experiment3" / "experiment3.csv"

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
        str(output_dir / "experiment3"),
        str(experiment3_path),
        str(plot_dir / "experiment3"),
        device,
        alpha_list=alpha_list,
    )


if __name__ == "__main__":
    main()
