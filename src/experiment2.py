import csv
import time
from pathlib import Path

import torch
from torch import nn

from src import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils
from src.cache_utils import get_torch_cache_path, save_torch_cache, load_torch_cache
from src.log_utils import print_section, print_memory_stats

NUM_HEADS = 24
HEAD_DIM = 64
D_STATE = 128

# Simple uniform affine quantization utilities (used by run_quantization)

def quantize_tensor(state: torch.Tensor, num_bits: int = 8):
    """Quantize a tensor and return values plus its scale metadata."""
    state = state.float()
    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val = state.min()
    max_val = state.max()

    # Avoid division by zero for constant tensors
    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)

    zero_point = qmin - min_val / scale
    zero_point = zero_point.round().clamp(qmin, qmax)

    q_state = ((state / scale) + zero_point).round().clamp(qmin, qmax)

    dtype = torch.uint8 if num_bits <= 8 else torch.uint16
    q_state = q_state.to(dtype)

    return q_state, scale.float(), zero_point.float()


def dequantize_tensor(q_state: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """Reconstruct floating-point values from quantized tensor metadata."""
    return (q_state.float() - zero_point) * scale

def run_state_management(model, tokenizer, snapshots, device, state_dir):
    """Run the uncompressed recurrent-state baseline for one session."""
    output_data = {}
    for snap in snapshots:
        turn_id = snap["turn_id"]
        decomp_latency = 0.0

        if turn_id == 0:
            # Turn 0: no prior state -> run fresh and get real latency/ppl
            state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                model,
                tokenizer,
                snap["history_text"],
                snap["new_text"],
                device=device
            )
            ssm_states, conv_states = state_utils.extract_state(state_output)
        else:
            ssm_states, conv_states = state_utils.load_state(state_dir, device=device)
            ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                model,
                tokenizer,
                snap["new_text"],
                ssm_states,
                conv_states,
                device=device
            )

        state_utils.save_state(ssm_states, conv_states, state_dir)
        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }
    return output_data


# Autoencoder-based compression
def run_autoencoder(model, tokenizer, snapshots, device, state_dir, ae_list):
    """Run state inference while encoding states with autoencoders."""
    output_data = {}
    for ae in ae_list:
        ae.eval()
        ae.to(device)

    num_layers = len(ae_list)

    for snap in snapshots:
        turn_id = snap["turn_id"]
        dequantize_latency = 0.0

        if turn_id == 0:
            state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                model,
                tokenizer,
                snap["history_text"],
                snap["new_text"],
                device=device
            )
            ssm_states, conv_states = state_utils.extract_state(state_output)
        else:
            payload = load_torch_cache(state_dir, device=device)
            latents = payload["ssm_latents"]      # {layer_idx: [heads, latent_dim]}
            conv_states = payload["conv_states"].to(device)

            decompressed_layers = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for layer_idx in range(num_layers):
                latent = latents[layer_idx].to(device).unsqueeze(0)   # [1, heads, latent_dim]
                reconstructed = ae_list[layer_idx].decoder(latent)    # [1, heads, head_dim, d_state]
                reconstructed = reconstructed.view(1, NUM_HEADS, HEAD_DIM, D_STATE)
                decompressed_layers.append(reconstructed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decomp_latency = time.perf_counter() - start_time
            ssm_states = torch.stack(decompressed_layers, dim=0)      # [num_layers, 1, heads, head_dim, d_state]

            ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                model,
                tokenizer,
                snap["new_text"],
                ssm_states,
                conv_states,
                device=device
            )

        # Compress ssm_states per layer for the next turn; conv_states pass through untouched.
        latents = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for layer_idx in range(num_layers):
            state = ssm_states[layer_idx].to(device)               # [1, heads, head_dim, d_state]
            latent = ae_list[layer_idx].encoder(
                state.view(1, NUM_HEADS, -1)                               # [1, heads, head_dim*d_state]
            )                                                       # [1, heads, latent_dim]
            latents[layer_idx] = latent.squeeze(0).cpu()            # [heads, latent_dim]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_latency = time.perf_counter() - start_time

        payload = {
            "ssm_latents": latents,
            "conv_states": conv_states.cpu()
        }
        save_torch_cache(payload, state_dir, "exp2", "ae")

        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency + encode_latency + decomp_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }

    return output_data


# Quantization-based compression
def run_quantization(model, tokenizer, snapshots, device, state_dir, num_bits=8):
    """Run state inference with direct SSM-state quantization."""
    output_data = {}
    for snap in snapshots:
        turn_id = snap["turn_id"]
        decomp_latency = 0.0

        if turn_id == 0:
            state_output, state_latency, state_ppl = evaluate_module.evaluate_baseline(
                model,
                tokenizer,
                snap["history_text"],
                snap["new_text"],
                device=device
            )
            ssm_states, conv_states = state_utils.extract_state(state_output)
        else:
            
            payload = load_compressed_payload(state_dir, device="cpu")
            q_ssm = payload["q_ssm"]
            scale = payload["scale"]
            zero_point = payload["zero_point"]
            conv_states = payload["conv_states"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            ssm_states = dequantize_tensor(q_ssm, scale, zero_point)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dequantize_latency = time.perf_counter() - start_time

            ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                model,
                tokenizer,
                snap["new_text"],
                ssm_states,
                conv_states,
                device=device
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        q_ssm, scale, zero_point = quantize_tensor(ssm_states.cpu(), num_bits=num_bits)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        quantize_latency = time.perf_counter() - start_time
        save_compressed_payload(
            {
                "q_ssm": q_ssm, "scale": scale, "zero_point": zero_point,
                "conv_states": conv_states.cpu(),
            },
            state_dir
        )

        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency + quantize_latency + dequantize_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }

    return output_data

def run_autoencoder_quantization(model, tokenizer, snapshots, device, state_dir, ae_list, num_bits=8):
    """Run state inference with autoencoder encoding and quantization."""
    output_data = {}
    for ae in ae_list:
        ae.eval()
        ae.to(device)
 
    num_layers = len(ae_list)
 
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
            ssm_states, conv_states = state_utils.extract_state(state_output)
        else:
            payload = load_compressed_payload(state_dir, device="cpu")
            q_latents = payload["q_latents"]      # [num_layers, heads, latent_dim]
            scale = payload["scale"]
            zero_point = payload["zero_point"]
            conv_states = payload["conv_states"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            latents = dequantize_tensor(q_latents, scale, zero_point)  # [num_layers, heads, latent_dim]

            decompressed_layers = []
            for layer_idx in range(num_layers):
                latent = latents[layer_idx].unsqueeze(0).to(device)    # [1, heads, latent_dim]
                reconstructed = ae_list[layer_idx].decoder(latent)    # [1, heads, head_dim, d_state]
                reconstructed = reconstructed.view(1, 24, 64, 128).cpu().float()
                decompressed_layers.append(reconstructed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decomp_latency = time.perf_counter() - start_time
            ssm_states = torch.stack(decompressed_layers, dim=0)      # [num_layers, 1, heads, head_dim, d_state]

            ssm_states, conv_states, state_latency, state_ppl = evaluate_module.evaluate_injected(
                model,
                tokenizer,
                snap["new_text"],
                ssm_states,
                conv_states,
                device=device
            )
 
        # Encode ssm_states per layer, stack into one tensor, then quantize
        # the whole latent stack at once (single scale/zero_point).
        latents = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for layer_idx in range(num_layers):
            state = ssm_states[layer_idx].to(device)                  # [1, heads, head_dim, d_state]
            latent = ae_list[layer_idx].encoder(
                state.view(1, 24, -1)                                  # [1, heads, head_dim*d_state]
            )                                                          # [1, heads, latent_dim]
            latents.append(latent.squeeze(0).cpu())                    # [heads, latent_dim]
        latents = torch.stack(latents, dim=0)                          # [num_layers, heads, latent_dim]
 
        q_latents, scale, zero_point = quantize_tensor(latents, num_bits=num_bits)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_latency = time.perf_counter() - start_time

        save_compressed_payload(
            {
                "q_latents": q_latents, "scale": scale, "zero_point": zero_point,
                "conv_states": conv_states.cpu(),
            },
            state_dir
        )
 
        if snap["role"] == "assistant":
            state_size_kb = utils.get_memory_size_kb(state_dir)
            output_data[turn_id] = {
                "state_latency": state_latency + encode_latency + decomp_latency,
                "state_size_kb": state_size_kb,
                "state_ppl": state_ppl
            }
 
    return output_data

# Experiment runner

def run_experiment_2_session(
    model, tokenizer, snapshots, session_id,
    output_dir,
    device, quant_bits_list=(8, 4),
    ae_experiments=None,           
    ae_untrained_experiments=None,
    force_rerun=None
):
    """Run all configured compression methods for one conversation."""
    force_rerun = force_rerun or set()
    print(f"Running original (no compression) state management for session {session_id}...")
    original_dir = f"{output_dir}/state_original_session.pt"

    original_results = utils.run_with_cache(
        output_dir, "baseline", session_id, "baseline" in force_rerun,
        lambda: run_state_management(model, tokenizer, snapshots, device, original_dir),
    )

    session_results = {}

    session_results["baseline"] = {
        turn_id: {
            "state_latency": metrics["state_latency"],
            "compressed_latency": metrics["state_latency"],
            "state_ppl": metrics["state_ppl"],
            "compressed_ppl": metrics["state_ppl"],
            "original_size_kb": metrics["state_size_kb"],
            "compressed_size_kb": metrics["state_size_kb"],
        }
        for turn_id, metrics in original_results.items()
    }

    def build_turn_dict(results):
        """Align one method's metrics with the original turn results."""
        turn_dict = {}
        for turn_id in original_results:
            if turn_id not in results:
                continue
            orig = original_results[turn_id]
            comp = results[turn_id]
            turn_dict[turn_id] = {
                "state_latency": orig["state_latency"],
                "compressed_latency": comp["state_latency"],
                "state_ppl": orig["state_ppl"],
                "compressed_ppl": comp["state_ppl"],
                "original_size_kb": orig["state_size_kb"],
                "compressed_size_kb": comp["state_size_kb"],
            }
        return turn_dict
    # Trained Autoencoders
    if ae_experiments:
        for latent_dim, ae_list in ae_experiments.items():
            print(f"\n{'='*50}")
            print(f"Session {session_id}: running AE latent_dim={latent_dim}")
            print(f"{'='*50}")

            compress_dir = f"{output_dir}/state_compressed_ae_{latent_dim}_session{session_id}.pt"

            ae_results = utils.run_with_cache(
                output_dir, f"ae_{latent_dim}", session_id, f"ae_{latent_dim}" in force_rerun,
                lambda: run_autoencoder(model, tokenizer, snapshots, device, compress_dir, ae_list)
            )

            session_results[f"ae_{latent_dim}"] = build_turn_dict(ae_results)

            print(f"Session {session_id}: finished AE latent_dim={latent_dim}")
            print(f"{'='*50}")
            torch.cuda.empty_cache()

    # Untrained autoencoder
    if ae_untrained_experiments:
        for latent_dim, ae_list in ae_untrained_experiments.items():
            print(f"\n{'='*50}")
            print(f"Session {session_id}: running UNTRAINED AE latent_dim={latent_dim}")
            print(f"{'='*50}")

            compress_dir = f"{output_dir}/state_compressed_ae_untrained_{latent_dim}.pt"

            ae_results = utils.run_with_cache(
                output_dir, f"ae_untrained_{latent_dim}", session_id, f"ae_untrained_{latent_dim}" in force_rerun,
                lambda: run_autoencoder(model, tokenizer, snapshots, device, compress_dir, ae_list)
            )

            session_results[f"ae_untrained_{latent_dim}"] = build_turn_dict(ae_results)

            print(f"Session {session_id}: finished UNTRAINED AE latent_dim={latent_dim}")
            print(f"{'='*50}")
            torch.cuda.empty_cache()

    # Quantization
    for num_bits in quant_bits_list:
        print(f"\n{'='*50}")
        print(f"Session {session_id}: running quantization num_bits={num_bits}")
        print(f"{'='*50}")

        quant_dir = f"{output_dir}/state_compressed_quant_{num_bits}bit.pt"

        quant_results = utils.run_with_cache(
            output_dir, f"quant_{num_bits}bit", session_id, f"quant_{num_bits}bit" in force_rerun,
            lambda: run_quantization(
                model, tokenizer, snapshots, device, quant_dir, num_bits=num_bits
            )
        )

        session_results[f"quant_{num_bits}bit"] = build_turn_dict(quant_results)

        print(f"Session {session_id}: finished quantization num_bits={num_bits}")
        print(f"{'='*50}")
        torch.cuda.empty_cache()

    # untrained + quantization int 8 bit
    if ae_untrained_experiments:
        for latent_dim, ae_list in ae_untrained_experiments.items():
            print(f"\n{'='*50}")
            print(f"Session {session_id}: running UNTRAINED AE latent_dim={latent_dim}")
            print(f"{'='*50}")

            compress_dir = f"{output_dir}/state_compressed_ae_untrained_{latent_dim}_quan_8.pt"

            ae_results = utils.run_with_cache(
                output_dir, f"ae_untrained_{latent_dim}_quan_8", session_id, f"ae_untrained_{latent_dim}_quan_8" in force_rerun,
                lambda: run_autoencoder_quantization(
                    model, tokenizer, snapshots, device, compress_dir, ae_list, 8
                )
            )

            session_results[f"ae_untrained_{latent_dim}_quan_8"] = build_turn_dict(ae_results)

            print(f"Session {session_id}: finished UNTRAINED AE latent_dim={latent_dim} + quantization 8 bit")
            print(f"{'='*50}")
            torch.cuda.empty_cache()
    return session_results


def run_experiment_2(
    model, tokenizer, sessions,
    output_dir, experiment_2_benchmark_path, plot_dir,
    device, quant_bits_list=(8, 4),
    ae_experiments=None,
    ae_untrained_experiments=None,
    force_rerun=None
):
    """Run Experiment 2 and aggregate compression results by turn."""
    all_results = {}

    for session_id, session in enumerate(sessions):
        print(f"\n{'#'*60}")
        print(f"# Session {session_id + 1}/{len(sessions)}")
        print(f"{'#'*60}")

        snapshots = data.build_turn_snapshots(session)

        session_results = run_experiment_2_session(
            model, tokenizer, snapshots, session_id,
            output_dir,
            device, quant_bits_list=quant_bits_list,
            ae_experiments=ae_experiments,
            ae_untrained_experiments=ae_untrained_experiments,
            force_rerun=force_rerun
        )

        for method_label, turn_dict in session_results.items():
            all_results.setdefault(method_label, []).append(turn_dict)

    with open(experiment_2_benchmark_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "turn", "n_sessions",
            "state_latency_mean", "state_latency_std",
            "compressed_latency_mean", "compressed_latency_std",
            "state_ppl_mean", "state_ppl_std",
            "compressed_ppl_mean", "compressed_ppl_std",
            "original_size_kb_mean", "original_size_kb_std",
            "compressed_size_kb_mean", "compressed_size_kb_std",
            "ppl_delta", "latency_delta", "latency_speedup",
            "compression_ratio",
        ])

        for method_label, per_session_turn_dicts in all_results.items():
            agg = data.aggregate_turn_results(per_session_turn_dicts)
            for turn_id in sorted(agg):
                a = agg[turn_id]
                writer.writerow([
                    method_label,
                    turn_id,
                    a.get("n_sessions", 0),
                    a.get("state_latency_mean", ""), a.get("state_latency_std", ""),
                    a.get("compressed_latency_mean", ""), a.get("compressed_latency_std", ""),
                    a.get("state_ppl_mean", ""), a.get("state_ppl_std", ""),
                    a.get("compressed_ppl_mean", ""), a.get("compressed_ppl_std", ""),
                    a.get("original_size_kb_mean", ""), a.get("original_size_kb_std", ""),
                    a.get("compressed_size_kb_mean", ""), a.get("compressed_size_kb_std", ""),
                    (
                        a["compressed_ppl_mean"] - a["state_ppl_mean"]
                        if "compressed_ppl_mean" in a and "state_ppl_mean" in a else ""
                    ),
                    (
                        a["compressed_latency_mean"] - a["state_latency_mean"]
                        if "compressed_latency_mean" in a and "state_latency_mean" in a else ""
                    ),
                    (
                        a["state_latency_mean"] / a["compressed_latency_mean"]
                        if a.get("compressed_latency_mean", 0) not in ("", 0)
                        and "state_latency_mean" in a else ""
                    ),
                    (
                        a["original_size_kb_mean"] / a["compressed_size_kb_mean"]
                        if a.get("compressed_size_kb_mean", 0) not in ("", 0)
                        and "original_size_kb_mean" in a else ""
                    ),
                ])

    df = pd.read_csv(experiment_2_benchmark_path)
    plot.plot_perplexity_comparison(df, plot_dir)
    plot.plot_latency_comparison_exp2(df, plot_dir)
    plot.plot_memory_growth_exp2(df, plot_dir)
    return df


def main():
    """Run Experiment 2 using the configured compression sweep."""
    torch.manual_seed(0)
    config = utils.read_config("configs/config1.yaml")

    paths = config["paths"]
    root = Path(__file__).parent.parent
    output_dir = str(root) + "/" + paths["output_dir"]
    text_history_dir = str(root) + "/" + paths["text_history_dir"] + "/history.txt"
    state_dir = str(root) + "/" + paths["state_dir"] + "/state.pt"
    plot_dir = str(root) + "/" + paths["plot_dir"]
    experiment2_path = output_dir + "/experiment2/experiment2.csv"

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
    print(f"Running Experiment 2 on {len(test_sessions)} test sessions")

    latent_dims = [1024, 2048, 4096]
    num_layers = 24

    ae_untrained_experiments = {
        ld: nn.ModuleList([
            autoencoder.Autoencoder(head_dim=64, d_state=128, hidden_dim=ld)
            for _ in range(num_layers)
        ])
        for ld in latent_dims
    }
    for latent_dim, ae_list in ae_untrained_experiments.items():
        for ae in ae_list:
            ae.eval()

    quant_bits_list = [4, 8, 16]
    force_rerun = {}
    df = run_experiment_2(
        model, tokenizer, test_sessions,
        output_dir + "/experiment2",
        experiment2_path,
        plot_dir + "/experiment2",
        device,
        quant_bits_list=quant_bits_list,
        ae_untrained_experiments=ae_untrained_experiments,
        force_rerun=force_rerun
    )


if __name__ == "__main__":
    main()