from pathlib import Path

import pandas as pd
import torch

from src import model_loader, data, plot, utils
from src.mamba2_stateful import patch_model
from src.aggregation_utils import (
    aggregate_by_boundary_offset,
    aggregate_boundary_only_by_session,
    summarize_boundary_health,
    write_boundary_sequence_csv,
    write_experiment_summary_csv,
)
from src.cache_utils import get_torch_cache_path, load_torch_cache, save_torch_cache
from src.log_utils import print_section
from src.state_runner import StateRunner


def _label_for_beta(beta):
    """Return the stable output label for an EMA configuration."""
    return "baseline" if beta is None else f"carryover_beta_{beta}"


def _cache_path(output_dir, label):
    """Return the cache path for one Experiment 4 condition."""
    return Path(output_dir) / "cache" / f"exp4_{label}.pt"


def load_or_run_chain(model, tokenizer, sessions, device, output_dir, beta, force_rerun=False):
    """Load a cached EMA chain or compute it when missing/invalidated."""
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
    """Blend a previous state with the latest session state."""
    if running_mean is None:
        return new_state.clone()
    return beta * running_mean + (1 - beta) * new_state


def run_state_chain_ema(model, tokenizer, sessions, device, state_dir, beta=None):
    """Run the shared state runner with optional EMA boundary carry-over."""
    def update_carryover(previous_ssm, previous_conv, session_ssm, session_conv):
        if beta is None:
            return session_ssm, session_conv
        return (
            ema_update(previous_ssm, session_ssm, beta),
            session_conv,
        )

    runner = StateRunner(
        state_dir,
        carryover_update=update_carryover,
        use_carryover=beta is not None,
    )
    return runner.run_chain(model, tokenizer, sessions, device)

def run_experiment_4(
    model, tokenizer, sessions,
    output_dir, experiment_4_benchmark_path, plot_dir,
    device, beta_list=(None, 0.0, 0.3, 0.5, 0.7, 0.9),
    force_rerun_labels=None,
):
    """Run Experiment 4 across EMA beta configurations."""
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

    write_experiment_summary_csv(experiment_4_benchmark_path, all_agg)

    boundary_csv_path = str(Path(experiment_4_benchmark_path).with_name("experiment4_boundary_sequence.csv"))
    write_boundary_sequence_csv(boundary_csv_path, all_boundary_seq)

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
    output_dir = Path(root) / paths["output_dir"]
    plot_dir = Path(root) / paths["plot_dir"]
    experiment4_path = output_dir / "experiment4" / "experiment4.csv"

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
        str(output_dir / "experiment4"),
        str(experiment4_path),
        str(plot_dir / "experiment4"),
        device,
        beta_list=beta_list,
        force_rerun_labels=force_rerun_labels,
    )


if __name__ == "__main__":
    main()
