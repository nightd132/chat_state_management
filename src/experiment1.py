from src import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils
import copy
import yaml
import torch
import pandas as pd
import csv
import time
import numpy as np
from pathlib import Path
NUM_RUNS = 5

def run_baseline(model, tokenizer, snapshots, device, text_history_dir, max_seq_len):
    output_data = {}
    for snap in snapshots:
        turn_id = snap["turn_id"]
        history_text = ""

        if turn_id > 0:
            history_text = utils.load_text(text_history_dir)

        combined_text = utils.concatenate_texts([history_text, snap["new_text"]])

        encoded_input = tokenizer(combined_text, max_length=max_seq_len, truncation=True, return_tensors="pt")
        truncated_input_ids = encoded_input["input_ids"]
        truncated_combined_text = tokenizer.decode(truncated_input_ids[0], skip_special_tokens=True)

        _, baseline_latency, baseline_ppl = evaluate_module.evaluate_baseline(
            model,
            tokenizer,
            truncated_combined_text,
            device=device
        )
        if snap["role"] == "assistant":
            baseline_size_kb = utils.get_memory_size_kb(text_history_dir)
            output_data[turn_id] = {
                "baseline_latency": baseline_latency,
                "baseline_size_kb": baseline_size_kb,
                "baseline_ppl": baseline_ppl
            }
        utils.save_text(truncated_combined_text, text_history_dir)
    return output_data

def run_state_management(model, tokenizer, snapshots, device, state_dir):
    output_data = {}
    for snap in snapshots:
        turn_id = snap["turn_id"]

        if turn_id == 0:
            state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                model,
                tokenizer,
                snap["new_text"],
                device=device
            )
        else:
            previous_state = state_utils.load_state(state_dir)

            state_output, state_latency, state_ppl = evaluate_module.evaluate_injected_mode(
                model,
                tokenizer,
                snap["new_text"],
                previous_state,
                device=device
            )

        new_state = [layer.recurrent_states for layer in state_output.cache_params.layers]
        state_utils.save_state(new_state, state_dir)
        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }
    return output_data


def main():
    config = utils.read_config("configs/config1.yaml")
    paths = config["paths"]
    root = Path(__file__).parent.parent
    output_dir = str(root) + "/" + paths["output_dir"]
    text_history_dir = str(root) + "/" + paths["text_history_dir"]+"/history.txt"
    state_dir = str(root) + "/" + paths["state_dir"]+"/state.pt"
    plot_dir = str(root) + "/" + paths["plot_dir"]
    max_seq_len = config["data"]["max_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )

    dataset = data.load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = data.extract_sessions(dataset)

    session = sessions[0]
    snapshots = data.build_turn_snapshots(session)

    experiment1_path = output_dir + "/experiment1/experiment1.csv"
    baseline_output = {}
    state_output = {}
    for _ in range(NUM_RUNS):
        torch.cuda.empty_cache()
        print(f"Run {_+1}/{NUM_RUNS}...")
        print("starting baseline run...")
        baseline_data = run_baseline(model, tokenizer, snapshots, device, text_history_dir, max_seq_len=max_seq_len)
        print("starting state management run...")
        state_data = run_state_management(model, tokenizer, snapshots, device, state_dir)
        print("aggregating results...")
        for turn_id in baseline_data:
            if turn_id not in baseline_output:
                baseline_output[turn_id] = {}
                baseline_output[turn_id]["baseline_latency"] = 0
                baseline_output[turn_id]["baseline_size_kb"] = 0
                baseline_output[turn_id]["baseline_ppl"] = 0
            baseline_output[turn_id]["baseline_latency"] += baseline_data[turn_id]["baseline_latency"]
            baseline_output[turn_id]["baseline_size_kb"] += baseline_data[turn_id]["baseline_size_kb"]
            baseline_output[turn_id]["baseline_ppl"] += baseline_data[turn_id]["baseline_ppl"]
            
            if turn_id not in state_output:
                state_output[turn_id] = {}
                state_output[turn_id]["state_latency"] = 0
                state_output[turn_id]["state_size_kb"] = 0
                state_output[turn_id]["state_ppl"] = 0
            state_output[turn_id]["state_latency"] += state_data[turn_id]["state_latency"]
            state_output[turn_id]["state_size_kb"] += state_data[turn_id]["state_size_kb"]
            state_output[turn_id]["state_ppl"] += state_data[turn_id]["state_ppl"]

        print(f"Completed run {_+1}/{NUM_RUNS}")

    for turn_id in baseline_output:
        baseline_output[turn_id]["baseline_latency"] /= NUM_RUNS
        baseline_output[turn_id]["baseline_size_kb"] /= NUM_RUNS
        baseline_output[turn_id]["baseline_ppl"] /= NUM_RUNS

        state_output[turn_id]["state_latency"] /= NUM_RUNS
        state_output[turn_id]["state_size_kb"] /= NUM_RUNS
        state_output[turn_id]["state_ppl"] /= NUM_RUNS

    with open(experiment1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["turn", "baseline_latency", "state_latency", "txt_size_kb", "pt_size_kb", "baseline_ppl", "state_ppl"])
        for turn_id in baseline_output:
            writer.writerow([
                turn_id,
                baseline_output[turn_id]["baseline_latency"],
                state_output[turn_id]["state_latency"],
                baseline_output[turn_id]["baseline_size_kb"],
                state_output[turn_id]["state_size_kb"],
                baseline_output[turn_id]["baseline_ppl"],
                state_output[turn_id]["state_ppl"]
            ])

    df = pd.read_csv(experiment1_path)
    latency_df = df[["turn", "baseline_latency", "state_latency"]]
    size_df = df[["turn", "txt_size_kb", "pt_size_kb"]]
    plot.plot_memory_growth(size_df, plot_dir + "/experiment1/memory_growth.png")
    plot.plot_latency_comparison(latency_df, plot_dir + "/experiment1/latency_comparison.png")
    plot.plot_ppl_comparison(df[["turn", "baseline_ppl", "state_ppl"]], plot_dir + "/experiment1/perplexity_comparison.png")

if __name__ == "__main__":
    main()