from src import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils
import copy
import yaml
import torch
from torch import nn
import pandas as pd
import csv
import time

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
                snap["new_text"],
                device=device
            )
        else:
            compressed_state = state_utils.load_state(state_dir)

            decompressed_state = []
            for layer_idx, latent in enumerate(compressed_state):
                latent = latent.to(device)                          # [heads, latent_dim]
                latent = latent.unsqueeze(0)                        # [1, heads, latent_dim]
                reconstructed = ae_list[layer_idx].decoder(latent)  # [1, heads, head_dim, d_state]
                decompressed_state.append(reconstructed)

            state_output, state_latency, state_ppl = evaluate_module.evaluate(
                model,
                tokenizer,
                snap["new_text"],
                decompressed_state,
                device=device
            )

        raw_states = [layer.recurrent_states for layer in state_output.cache_params.layers]
        compressed = []
        for layer_idx, state in enumerate(raw_states):
            state = state.to(device)                                # [1, heads, head_dim, d_state]
            latent = ae_list[layer_idx].encoder(
                state.view(1, 24, -1)                               # [1, heads, head_dim*d_state]
            )                                                        # [1, heads, latent_dim]
            compressed.append(latent.squeeze(0).cpu())              # [heads, latent_dim]

        state_utils.save_state(compressed, state_dir)

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

        with open(experiment_2_benchmark_path, "a", newline="") as f:  # ← "a" not "w"
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

        torch.cuda.empty_cache()

    df = pd.read_csv(experiment_2_benchmark_path)
    plot.plot_perplexity_comparison(df, plot_dir)
    plot.plot_latency_comparison_exp2(df, plot_dir)
    plot.plot_memory_growth_exp2(df, plot_dir)
    return df
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

            state_output, state_latency, state_ppl = evaluate_module.evaluate(
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
    output_dir = paths["output_dir"]
    plot_dir = paths["plot_dir"]
    experiment_2_benchmark_path = output_dir + "/experiment_2_benchmark.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )

    dataset = data.load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = data.extract_sessions(dataset)

    training_sessions = sessions[1]
    train_snapshots = data.build_turn_snapshots(training_sessions)
    num_layers = 24
    sample_states = {i: [] for i in range(num_layers)}

    for snap in train_snapshots:
        inputs = tokenizer(snap["new_text"], return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(
                **inputs,
                use_cache=True
            )

        cache = output.cache_params

        for i, layer in enumerate(cache.layers):
            state = layer.recurrent_states          # [1, 24, 64, 128]
            sample_states[i].append(
                state.squeeze(0).cpu()              # [24, 64, 128] — remove batch dim
            )

        torch.cuda.empty_cache()

    latent_dims = [16, 32, 64, 128]

    ae_experiments = {
        ld: nn.ModuleList([
            autoencoder.Autoencoder(head_dim=64, d_state=128, hidden_dim=ld)
            for _ in range(num_layers)
        ])
        for ld in latent_dims
    }


    stacked_states = {
        i: torch.stack(sample_states[i], dim=0)   # [num_samples, heads, head_dim, d_state]
        for i in range(num_layers)
    }

    for layer_idx in range(num_layers):
        print(f"\n{'='*40}")
        print(f"Training AE for Layer {layer_idx}")
        print(f"{'='*40}")

        states = stacked_states[layer_idx]
        print(f"  States shape: {states.shape}")   # [num_samples, 24, 64, 128]

        loss_history = ae_list[layer_idx].fit(
            states,
            num_epochs=20,
            batch_size=256,
            learning_rate=1e-3,
            device=device
        )
    
    experiment2_path = output_dir + "/experiment2/experiment2.csv"
    session = sessions[0]
    snapshots = data.build_turn_snapshots(session)

    df = run_experiment_2(
        model, tokenizer, snapshots, ae_experiments,
        output_dir + "/experiment2",
        experiment2_path,
        plot_dir + "/experiment2",
        device
    )

if __name__ == "__main__":
    main()