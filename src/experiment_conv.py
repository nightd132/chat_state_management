from Test import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils

import yaml
import torch
import pandas as pd
import csv
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
    experiment_2_benchmark_path = output_dir + "/experiment_2_benchmark_conv.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )

    dataset = data.load_data(config["data"]["name"], split=config["data"]["split"])
    sessions = data.extract_sessions(dataset)

    sample_states = []
    training_sessions = sessions[1]

    train_snapshots = data.build_turn_snapshots(training_sessions)
    for snap in train_snapshots:
        with torch.no_grad():
            output = model(**tokenizer(snap["new_text"], return_tensors="pt").to(device))
            state_list = output.cache_params.ssm_states
            for layer_state in state_list:            # all layers
                sample_states.append(layer_state[0].detach().cpu())  # [T,D]
        torch.cuda.empty_cache()

    flatten_dim = sample_states[0].size(-1)
    latent_dims = [1, 2, 4, 8]
    AEs = []

    for latent_dim in latent_dims:
        print(f"Training AE with latent dim {latent_dim}...")
        AE = autoencoder.Autoencoder(
            input_dim=flatten_dim,
            hidden_dim=latent_dim
        ).to(device)
        AE.fit(sample_states, num_epochs=10, batch_size=256,
               learning_rate=1e-3, device=device)
        AEs.append(AE)
        torch.cuda.empty_cache()
        print(f"Finished training AE with latent dim {latent_dim}.")

    session = sessions[0]
    snapshots = data.build_turn_snapshots(session)

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
                        current_ssm_states = [s[0].detach().cpu() for s in output.cache_params.ssm_states]
                        current_conv_states = [s[0].detach().cpu() for s in output.cache_params.conv_states]
                    
                    current_states = {"ssm": current_ssm_states, "conv": current_conv_states}
                    compressed_states = [AE.encoder(s.to(device)).cpu() for s in current_ssm_states]
                    current_compressed_states = {"ssm":compressed_states, "conv": current_conv_states}

                    state_utils.save_state(current_states,    f"{output_dir}/state_original_conv_{AE.hidden_dim}.pt")
                    state_utils.save_state(current_compressed_states, f"{output_dir}/state_compressed_conv_{AE.hidden_dim}.pt")
                    
                    continue

                prev_original    = state_utils.load_state(f"{output_dir}/state_original_conv_{AE.hidden_dim}.pt")
                prev_compressed  = state_utils.load_state(f"{output_dir}/state_compressed_conv_{AE.hidden_dim}.pt")
                prev_compressed_ssm = prev_compressed["ssm"]
                prev_compressed_conv = prev_compressed["conv"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_start = time.perf_counter()
                with torch.no_grad():
                    reconstructed_states = [AE.decoder(c.to(device)).cpu() for c in prev_compressed_ssm]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - decode_start
                output, state_latency, state_ppl = evaluate_module.evaluate_conv(
                    model, tokenizer, snap["new_text"],
                    prev_original["ssm"], prev_original["conv"], device=device
                )
                compressed_output, compressed_latency, compressed_ppl = evaluate_module.evaluate_conv(
                    model, tokenizer, snap["new_text"],
                    reconstructed_states, prev_compressed_conv, device=device
                )

                if role == "assistant":
                    compressed_latency += decode_time

                    original_size_kb   = utils.get_memory_size_kb(f"{output_dir}/state_original_conv_{AE.hidden_dim}.pt")
                    compressed_size_kb = utils.get_memory_size_kb(f"{output_dir}/state_compressed_conv_{AE.hidden_dim}.pt")

                    writer.writerow([
                        turn_id, state_latency, compressed_latency,
                        state_ppl, compressed_ppl,
                        original_size_kb, compressed_size_kb,
                        AE.hidden_dim
                    ])

                
                current_compressed_ssm_states = [s[0].detach().cpu() 
                                                 for s in compressed_output.cache_params.ssm_states]
                current_compressed_conv_states = [s[0].detach().cpu() 
                                                  for s in compressed_output.cache_params.conv_states]

                current_ssm_states = [s[0].detach().cpu() for s in output.cache_params.ssm_states]
                current_conv_states = [s[0].detach().cpu() for s in output.cache_params.conv_states]
                compressed_ssm_states = [AE.encoder(s.to(device)).cpu() 
                                         for s in current_compressed_ssm_states]

                current_states = {"ssm": current_ssm_states, "conv": current_conv_states}
                compressed_states = {"ssm": compressed_ssm_states, "conv": current_compressed_conv_states}

                state_utils.save_state(current_states,    f"{output_dir}/state_original_conv_{AE.hidden_dim}.pt")
                state_utils.save_state(compressed_states, f"{output_dir}/state_compressed_conv_{AE.hidden_dim}.pt")

            torch.cuda.empty_cache()

    df = pd.read_csv(experiment_2_benchmark_path)
    plot.plot_perplexity_comparison(df, plot_dir+"/conv")
    plot.plot_latency_comparison_exp2(df, plot_dir+"/conv")
    plot.plot_memory_growth_exp2(df, plot_dir+"/conv")

if __name__ == "__main__":
    main()