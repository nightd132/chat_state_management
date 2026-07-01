from src import model_loader, state_utils, data, plot, utils
from src.evaluate import evaluate_baseline, evaluate_injected
from src.mamba2_stateful import patch_model, extract_states

import torch
import pandas as pd
import csv
from pathlib import Path
from tqdm import tqdm

NUM_LATENCY_RUNS = 1


def run_baseline_session(model, tokenizer, snapshots, device,
                         text_history_dir, max_seq_len):
    output_data = {}
    history_text = ""

    for snap in snapshots:
        turn_id  = snap["turn_id"]
        new_text = snap["new_text"]

        combined_text = utils.concatenate_texts([history_text, new_text]) \
                        if history_text else new_text

        tokenizer.truncation_side = "left"
        encoded = tokenizer(combined_text, max_length=max_seq_len,
                            truncation=True, return_tensors="pt")
        truncated_ids  = encoded["input_ids"]
        truncated_text = tokenizer.decode(truncated_ids[0], skip_special_tokens=True)

        history_len = tokenizer(history_text, return_tensors="pt").input_ids.shape[1] \
                      if history_text else 0

        _, latency, ppl = evaluate_baseline(
            model, tokenizer,
            history_text=truncated_text,
            input_text=new_text,
            device=device,
        )

        if snap["role"] == "assistant":
            output_data[turn_id] = {
                "baseline_latency": latency,
                "baseline_ppl": ppl,
                "txt_size_kb": utils.get_memory_size_kb(text_history_dir)
            }
        
        history_text = truncated_text 
    return output_data


def run_injected_session(model, tokenizer, snapshots, device, state_dir):
    output_data = {}
    ssm_states  = None
    conv_states = None

    for snap in snapshots:
        turn_id  = snap["turn_id"]
        new_text = snap["new_text"]

        if turn_id == 0:
            # Turn 0: no prior state => run normally and extract states
            history_ids = tokenizer(snap["history_text"],
                                    return_tensors="pt").input_ids.to(device)
            _, latency, ppl = evaluate_baseline(
                model, tokenizer,
                history_text=snap["history_text"],
                input_text=new_text,
                device=device,
            )
            ssm_states, conv_states = state_utils.extract_states(model)

        else:
            # Turn N: inject saved states => run only new_text tokens
            new_ssm, new_conv, latency, ppl = evaluate_injected(
                model, tokenizer, new_text,
                ssm_states, conv_states,
                device=device,
            )
            ssm_states  = new_ssm
            conv_states = new_conv

        if snap["role"] == "assistant":
            state_utils.save_state(ssm_states, conv_states, state_dir)
            state_size_kb = utils.get_memory_size_kb(state_dir)

            output_data[turn_id] = {
                "state_latency": latency,
                "state_ppl": ppl,
                "pt_size_kb": state_size_kb,
            }

    return output_data


def main():
    config = utils.read_config("configs/config1.yaml")
    paths = config["paths"]
    root = Path(__file__).parent.parent

    output_dir = str(root) + "/" + paths["output_dir"]
    state_dir = str(root) + "/" + paths["state_dir"] + "/state.pt"
    plot_dir = str(root) + "/" + paths["plot_dir"]
    max_seq_len = config["data"]["max_length"]

    Path(output_dir + "/experiment1").mkdir(parents=True, exist_ok=True)
    Path(plot_dir + "/experiment1").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"]),
    )
    patch_model(model)

    dataset  = data.load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = data.extract_sessions(dataset)

    # 70/15/15 split — test set only for Experiment 1
    _, _, test_sessions = data.split_sessions(sessions, train=0.70, val=0.15, test=0.15)
    print(f"Running Experiment 1 on {len(test_sessions)} test sessions")

    all_baseline_results = []
    all_injected_results = []

    for session_idx, session in enumerate(tqdm(test_sessions, desc="Sessions")):
        snapshots = data.build_turn_snapshots(session)
        torch.cuda.empty_cache()

        baseline_result = run_baseline_session(
            model, tokenizer, snapshots, device,
            text_history_dir=None,   # we manage history inside the function now
            max_seq_len=max_seq_len,
        )

        # For latency averaging: re-run baseline NUM_LATENCY_RUNS-1 more times
        for _ in range(NUM_LATENCY_RUNS - 1):
            extra = run_baseline_session(
                model, tokenizer, snapshots, device,
                text_history_dir=None,
                max_seq_len=max_seq_len,
            )
            for turn_id in baseline_result:
                baseline_result[turn_id]["baseline_latency"] += \
                    extra[turn_id]["baseline_latency"]

        for turn_id in baseline_result:
            baseline_result[turn_id]["baseline_latency"] /= NUM_LATENCY_RUNS

        all_baseline_results.append(baseline_result)

        #Injected
        injected_result = run_injected_session(
            model, tokenizer, snapshots, device, state_dir
        )

        for _ in range(NUM_LATENCY_RUNS - 1):
            extra = run_injected_session(
                model, tokenizer, snapshots, device, state_dir
            )
            for turn_id in injected_result:
                injected_result[turn_id]["state_latency"] += \
                    extra[turn_id]["state_latency"]

        for turn_id in injected_result:
            injected_result[turn_id]["state_latency"] /= NUM_LATENCY_RUNS

        all_injected_results.append(injected_result)

    baseline_agg = data.aggregate_turn_results(all_baseline_results)
    injected_agg = data.aggregate_turn_results(all_injected_results)

    csv_path = output_dir + "/experiment1/experiment1.csv"
    all_turns = sorted(set(baseline_agg) | set(injected_agg))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn",
            "n_sessions",
            "baseline_ppl_mean", "baseline_ppl_std",
            "state_ppl_mean", "state_ppl_std",
            "baseline_latency_mean", "baseline_latency_std",
            "state_latency_mean", "state_latency_std",
            "txt_size_kb_mean", "txt_size_kb_std",
            "pt_size_kb_mean", "pt_size_kb_std",
        ])
        for turn_id in all_turns:
            b = baseline_agg.get(turn_id, {})
            s = injected_agg.get(turn_id, {})
            writer.writerow([
                turn_id,
                b.get("n_sessions", 0),
                b.get("baseline_ppl_mean", ""),
                b.get("baseline_ppl_std", ""),
                s.get("state_ppl_mean", ""),
                s.get("state_ppl_std", ""),
                b.get("baseline_latency_mean", ""),
                b.get("baseline_latency_std", ""),
                s.get("state_latency_mean", ""),
                s.get("state_latency_std", ""),
                b.get("txt_size_kb_mean", ""),
                b.get("txt_size_kb_std", ""),
                s.get("pt_size_kb_mean", ""),
                s.get("pt_size_kb_std", ""),
            ])

    print(f"Results saved to {csv_path}")

    df = pd.read_csv(csv_path)

    plot.plot_ppl_comparison(df, plot_dir + "/experiment1/perplexity_comparison.png")

    plot.plot_latency_comparison(df, plot_dir + "/experiment1/latency_comparison.png")

    plot.plot_memory_growth(df, plot_dir + "/experiment1/memory_growth.png")

    plot.plot_speedup(df, plot_dir + "/experiment1/speedup.png")

    print("Experiment 1 complete.")
    print(f"Plots saved to {plot_dir}/experiment1/")


if __name__ == "__main__":
    main()


