from src import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils
import copy
import yaml
import torch
from torch import nn
import pandas as pd
import csv
import time
import os
from pathlib import Path

def run_autoencoder(model, tokenizer, snapshots, device, state_dir, ae_list):
    output_data = {}
    for ae in ae_list:
        ae.eval()
        ae.to(device)
    for snap in snapshots:
        turn_id = snap["turn_id"]

        if turn_id == 0:
            state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                model,
                tokenizer,
                snap["history_text"],
                snap["new_text"],
                device=device
            )
        else:
            compressed_state = state_utils.load_state(state_dir)

            decompressed_state = {}
            for layer_idx, latent in compressed_state.items():
                latent = latent.to(device)                          # [heads, latent_dim]
                latent = latent.unsqueeze(0)                        # [1, heads, latent_dim]
                reconstructed = ae_list[layer_idx].decoder(latent)  # [1, heads, head_dim, d_state]
                # print(f"layer: {layer_idx}, reconstructed shape: {reconstructed.shape}")
                reconstructed = reconstructed.view(1, 24, 64, 128)
                decompressed_state[layer_idx] = reconstructed
                
            state_output, state_latency, state_ppl = evaluate_module.evaluate_injected_mode(
                model,
                tokenizer,
                snap["new_text"],
                decompressed_state,
                device=device
            )

        raw_states = state_utils.save_recurrent_states(state_output.cache_params)
        compressed = {}
        for layer_idx, state in raw_states.items():
            state = state.to(device)                                # [1, heads, head_dim, d_state]
            latent = ae_list[layer_idx].encoder(
                state.view(1, 24, -1)                               # [1, heads, head_dim*d_state]
            )                                                       # [1, heads, latent_dim]
            compressed[layer_idx]=(latent.squeeze(0).cpu())              # [heads, latent_dim]

        state_utils.save_state(compressed, state_dir)

        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }

    return output_data

def run_state_management(model, tokenizer, snapshots, device, state_dir):
    output_data = {}
    for snap in snapshots:
        turn_id = snap["turn_id"]

        if turn_id %2!= 0:
            history_text = snap["history_text"]
            history_ids = tokenizer(history_text, return_tensors="pt").to(device)
            state_output = model(history_ids["input_ids"], use_cache=True)
        else:
            previous_state = state_utils.load_state(state_dir)
            state = {}
            if type(previous_state) == list:
                for layer_idx, states in enumerate(previous_state):
                    state[layer_idx] = states
            else:
                state = previous_state
            state_output, state_latency, state_ppl = evaluate_module.evaluate_injected_mode(
                model,
                tokenizer,
                snap["new_text"],
                state,
                device=device
            )

        new_state = state_utils.save_recurrent_states(state_output.cache_params)
        state_utils.save_state(new_state, state_dir)
        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }
    return output_data

def run_experiment_2(
    model, tokenizer, snapshots, ae_experiments,  # ae_experiments: {latent_dim: ae_list}
    output_dir, experiment_2_benchmark_path, plot_dir,
    device
):
    # Write header once
    with open(experiment_2_benchmark_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn", "state_latency", "compressed_latency",
            "state_ppl", "compressed_ppl",
            "original_size_kb", "compressed_size_kb",
            "autoencoder_latent_dim"
        ])

    print("Running original state management...")
    original_dir     = f"{output_dir}/state_original.pt"
    
    original_results = run_state_management(
        model, tokenizer, snapshots, device, original_dir
    )

    for latent_dim, ae_list in ae_experiments.items():
        print(f"\n{'='*50}")
        print(f"Running AE latent_dim={latent_dim}")
        print(f"{'='*50}")

        compress_dir = f"{output_dir}/state_compressed_{latent_dim}.pt"

        ae_results = run_autoencoder(
            model, tokenizer, snapshots, device, compress_dir, ae_list
        )

        with open(experiment_2_benchmark_path, "a", newline="") as f:
            writer = csv.writer(f)
            for turn_id in original_results:
                if turn_id not in ae_results:
                    continue
                orig = original_results[turn_id]
                comp = ae_results[turn_id]
                writer.writerow([
                    turn_id,
                    orig["state_latency"], comp["state_latency"],
                    orig["state_ppl"],     comp["state_ppl"],
                    orig["state_size_kb"], comp["state_size_kb"],
                    latent_dim
                ])
        print(f"Finish running AE latent_dim={latent_dim}")
        print(f"{'='*50}")
        torch.cuda.empty_cache()

    df = pd.read_csv(experiment_2_benchmark_path)
    plot.plot_perplexity_comparison(df, plot_dir)
    plot.plot_latency_comparison_exp2(df, plot_dir)
    plot.plot_memory_growth_exp2(df, plot_dir)
    return df

def main():
    config = utils.read_config("configs/config1.yaml")
    
    paths = config["paths"]
    root = Path(__file__).parent.parent
    output_dir = str(root) + "/" + paths["output_dir"]
    text_history_dir = str(root) + "/" + paths["text_history_dir"]+"/history.txt"
    state_dir = str(root) + "/" + paths["state_dir"]+"/state.pt"
    plot_dir = str(root) + "/" + paths["plot_dir"]
    experiment2_path = output_dir + "/experiment2/experiment2.csv"

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


    latent_dims = [256, 512, 1024]
    num_layers = 24

    ae_experiments = {
        ld: nn.ModuleList([
            autoencoder.Autoencoder(head_dim=64, d_state=128, hidden_dim=ld)
            for _ in range(num_layers)
        ])
        for ld in latent_dims
    }
    save_dir = "autoencoders"
    for latent_dim, ae_list in ae_experiments.items():
        for layer_idx in range(num_layers):
            path =  os.path.join(save_dir, f"latent_dim_{latent_dim}/layer_{layer_idx}/autoencoder.pt")
            ae_list[layer_idx].load_state_dict(
                torch.load(
                    path,
                    map_location="cpu")
            )
            ae_list[layer_idx].eval()

    df = run_experiment_2(
        model, tokenizer, snapshots, ae_experiments,
        output_dir + "/experiment2",
        experiment2_path,
        plot_dir + "/experiment2",
        device
    )

if __name__ == "__main__":
    main()