import yaml
import torch
import pandas as pd
import numpy as np
import csv

from data.data_loader import load_data, extract_sessions, build_turn_snapshots
from models.model_loader import load_model
from storage.state_io import save_state, load_state, feed_synthetic_ssm_state
from storage.history_text_io import save_text, load_text, concatenate_texts
from benchmarks.latency_memory_bench import measure_baseline_latency, measure_state_management_latency, print_benchmark_results, plot_memory_growth, plot_latency_comparison, get_memory_size_kb

def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():

    config = read_config("configs/config1.yaml")
    
    paths = config["paths"]
    output_dir = paths["output_dir"]
    text_history_dir = paths["text_history_dir"]+"/history.txt"
    state_dir = paths["state_dir"]+"/state.pt"
    plot_dir = paths["plot_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(config["model"]["name"], 
                                  device=config["model"]["device"], 
                                  dtype=getattr(torch, config["model"]["dtype"]))
    
    dataset = load_data(config["data"]["name"], 
                        split=config["data"]["split"])

    sessions = extract_sessions(dataset)

    session = sessions[0]
    snapshots = build_turn_snapshots(session)
    turn = 0
    latency_size_comparison_path = output_dir + "/latency_size_comparison.csv"
    NUM_RUNS = 5
    with open(latency_size_comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["turn", "baseline_latency", "state_latency", "txt_size_kb", "pt_size_kb"])
        
        for snap in snapshots:
            turn_id = snap["turn_id"]
            history_text = ""

            if turn_id > 0:
                history_text = load_text(text_history_dir)
            
            baseline_output, baseline_latency = measure_baseline_latency(
                model,
                tokenizer,
                snap["new_text"],
                history_text,
                device=device
            )

            if turn_id == 0:
                state_output = baseline_output
                state_latency = baseline_latency
            else:
                previous_state = load_state(state_dir)
                state_latencies = []
                for _ in range(NUM_RUNS-1):
                    _, latency = measure_state_management_latency(
                        model, tokenizer,
                        snap["new_text"], previous_state,
                        device=device
                    )
                    state_latencies.append(latency)


                state_output, state_latency = measure_state_management_latency(
                    model,
                    tokenizer,
                    snap["new_text"],
                    previous_state,
                    device=device
                )
                state_latencies.append(latency)
                state_latency = float(np.mean(state_latencies))

            new_state = state_output.cache_params.ssm_states
            save_state(new_state, state_dir)
            save_text(concatenate_texts([history_text, snap["new_text"]]), text_history_dir)
            baseline_size_kb = get_memory_size_kb(text_history_dir)
            state_size_kb = get_memory_size_kb(state_dir)
            turn += 1
            writer.writerow([turn_id, baseline_latency, state_latency, baseline_size_kb, state_size_kb])

    df = pd.read_csv(latency_size_comparison_path)
    latency_df = df[["turn", "baseline_latency", "state_latency"]]
    size_df = df[["turn", "txt_size_kb", "pt_size_kb"]]
    plot_memory_growth(size_df, plot_dir + "/memory_growth.png")
    plot_latency_comparison(latency_df, plot_dir + "/latency_comparison.png")
    

if __name__ == "__main__":
    main()
    
    
    