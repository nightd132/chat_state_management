from time import time

import yaml
import torch
import pandas as pd
import numpy as np
import csv
from data.data_loader import load_data, extract_sessions, build_turn_snapshots
from models.model_loader import load_model
from storage.state_io import save_state, load_state, feed_synthetic_ssm_state
from storage.history_text_io import save_text, load_text, concatenate_texts
from benchmarks.perplexity_benchmark import evaluate, plot_perplexity_comparison, plot_latency_comparison_exp2, plot_memory_growth_exp2
from benchmarks.latency_memory_bench import measure_baseline_latency, measure_state_management_latency, print_benchmark_results, plot_memory_growth, plot_latency_comparison, get_memory_size_kb
from models.autoencoder import Autoencoder
import time

def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = read_config("configs/config1.yaml")
    
    paths = config["paths"]
    output_dir = paths["output_dir"]
    plot_dir = paths["plot_dir"]
    experiment_2_benchmark_path = output_dir + "/experiment_2_benchmark.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )

    dataset = load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = extract_sessions(dataset)

    sample_states = []
    training_sessions = sessions[1]

    train_snapshots = build_turn_snapshots(training_sessions)
    for snap in train_snapshots:
        with torch.no_grad():
            output = model(**tokenizer(snap["new_text"], return_tensors="pt").to(device))
            state_list = output.cache_params.ssm_states
            for layer_state in state_list:
                sample_states.append(layer_state[0].detach().cpu())
        torch.cuda.empty_cache()

    flatten_dim = sample_states[0].size(-1)
    latent_dims = [1, 2, 4, 8]
    AEs = []

    for latent_dim in latent_dims:
        print(f"Training AE with latent dim {latent_dim}...")
        AE = Autoencoder(
            input_dim=flatten_dim,
            hidden_dim=latent_dim
        ).to(device)
        AE.fit(sample_states, num_epochs=10, batch_size=256,
               learning_rate=1e-3, device=device)
        AEs.append(AE)
        torch.cuda.empty_cache()
        print(f"Finished training AE with latent dim {latent_dim}.")

    session = sessions[0]
    snapshots = build_turn_snapshots(session)

    with open(experiment_2_benchmark_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn", "state_latency", "compressed_latency",
            "state_ppl", "compressed_ppl",
            "original_size_kb", "compressed_size_kb",
            "autoencoder_latent_dim"
        ])

        for snap in snapshots:
            turn_id = snap["turn_id"]
            role    = snap["role"]

            for AE in AEs:
                AE = AE.to(device)

                if turn_id == 0:
                    with torch.no_grad():
                        output = model(**tokenizer(snap["new_text"], return_tensors="pt").to(device))
                        current_states = [s[0].detach().cpu() for s in output.cache_params.ssm_states]

                    compressed_states = [AE.encoder(s.to(device)).cpu() for s in current_states]

                    save_state(current_states, f"{output_dir}/state_original_{AE.hidden_dim}.pt")
                    save_state(compressed_states, f"{output_dir}/state_compressed_{AE.hidden_dim}.pt")
                    continue

                prev_original = load_state(f"{output_dir}/state_original_{AE.hidden_dim}.pt")
                prev_compressed  = load_state(f"{output_dir}/state_compressed_{AE.hidden_dim}.pt")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_start = time.perf_counter()
                with torch.no_grad():
                    reconstructed_states = [AE.decoder(c.to(device)).cpu() for c in prev_compressed]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - decode_start
                output, state_latency, state_ppl = evaluate(
                    model, tokenizer, snap["new_text"],
                    prev_original, device=device
                )
                compressed_output, compressed_latency, compressed_ppl = evaluate(
                    model, tokenizer, snap["new_text"],
                    reconstructed_states, device=device
                )

                if role == "assistant":
                    compressed_latency += decode_time

                    original_size_kb = get_memory_size_kb(f"{output_dir}/state_original_{AE.hidden_dim}.pt")
                    compressed_size_kb = get_memory_size_kb(f"{output_dir}/state_compressed_{AE.hidden_dim}.pt")

                    writer.writerow([
                        turn_id, state_latency, compressed_latency,
                        state_ppl, compressed_ppl,
                        original_size_kb, compressed_size_kb,
                        AE.hidden_dim
                    ])

                
                current_compressed_states = [s[0].detach().cpu() for s in compressed_output.cache_params.ssm_states]

                current_states = [s[0].detach().cpu() for s in output.cache_params.ssm_states]
                compressed_states = [AE.encoder(s.to(device)).cpu() for s in current_compressed_states]

                save_state(current_states, f"{output_dir}/state_original_{AE.hidden_dim}.pt")
                save_state(compressed_states, f"{output_dir}/state_compressed_{AE.hidden_dim}.pt")

            torch.cuda.empty_cache()

    df = pd.read_csv(experiment_2_benchmark_path)
    plot_perplexity_comparison(df, plot_dir)
    plot_latency_comparison_exp2(df, plot_dir)
    plot_memory_growth_exp2(df, plot_dir)

if __name__ == "__main__":
    main()