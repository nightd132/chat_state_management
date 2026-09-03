from src import model_loader, data, autoencoder, utils

import torch
import torch.nn as nn
from pathlib import Path
import argparse


def collect_states(model, tokenizer, sessions, max_seq_len, save_dir, max_samples, force: bool = False):
    """Collect recurrent states for assistant turns from a session split.

    If `force` is True, any existing `sample_*.pt` files under `save_dir` are removed
    so collection starts from zero. If `force` is False and enough samples already
    exist (>= `max_samples`) collection is skipped and the existing count is returned.
    """
    # If samples already exist, resume from there or skip entirely. When forcing,
    # delete existing samples so collection restarts.
    layer0_dir = save_dir / "layer_0"
    existing_count = 0
    if layer0_dir.exists():
        existing_count = len(list(layer0_dir.glob("sample_*.pt")))
        if existing_count >= max_samples and not force:
            print(f"Found {existing_count} existing samples in {save_dir}; skipping collection.")
            return existing_count
    if force and save_dir.exists():
        # remove existing sample files for all layers so collection restarts cleanly
        for layer_path in save_dir.glob("layer_*"):
            for p in layer_path.glob("sample_*.pt"):
                try:
                    p.unlink()
                except Exception:
                    pass
        existing_count = 0
    count = existing_count
    for session in sessions:
        for snap in data.build_turn_snapshots(session):
            if snap["role"] != "assistant":
                continue
            inputs = tokenizer(
                snap["history_text"],
                max_length=max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(inputs["input_ids"], use_cache=True)
            for layer_idx, layer in enumerate(output.cache_params.layers):
                layer_dir = save_dir / f"layer_{layer_idx}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                sample_path = layer_dir / f"sample_{count:06d}.pt"
                if sample_path.exists():
                    # avoid overwriting existing sample files (shouldn't happen when forcing)
                    continue
                torch.save(
                    {"state": layer.recurrent_states.squeeze(0).cpu()},
                    sample_path,
                )
            count += 1
            if count >= max_samples:
                return count
    return count


def load_states(save_dir, layer_idx, count):
    """Load flattened recurrent-state samples for one layer."""
    layer_dir = Path(save_dir) / f"layer_{layer_idx}"
    states = []
    available = 0
    for idx in range(count):
        path = layer_dir / f"sample_{idx:06d}.pt"
        if not path.exists():
            # stop at first missing sample and return what we have
            break
        data = torch.load(path, map_location="cpu")
        st = data.get("state")
        if st is None:
            continue
        # convert to float32 for training, keep shape (nheads, head_dim, d_state)
        st = st.float()
        states.append(st)
        available += 1

    if available == 0:
        raise FileNotFoundError(f"No state samples found in {layer_dir}")

    # Ensure all samples share the same shape
    first_shape = tuple(states[0].shape)
    for i, s in enumerate(states):
        if tuple(s.shape) != first_shape:
            raise ValueError(f"Inconsistent sample shape at index {i}: {s.shape} != {first_shape}")

    # Stack into (N, nheads, head_dim, d_state)
    stacked = torch.stack(states, dim=0)
    return stacked


def reconstruction_loss(ae, states, device):
    """Measure held-out reconstruction MSE without updating model weights."""
    ae.eval()
    with torch.no_grad():
        # Ensure tensor on correct device and flattened to (N, feature_dim)
        values = states.to(device)
        if values.ndim > 2:
            values = values.view(values.shape[0], -1)
        # If training used per-head flattening (fit calls view(-1, input_dim)),
        # reconstruction should also reshape to match the autoencoder input.
        if values.shape[1] % ae.input_dim == 0 and values.shape[1] != ae.input_dim:
            # Collapse any stacked-heads dimension into more rows of input_dim
            values = values.view(-1, ae.input_dim)
        reconstructed, _ = ae(values)
        return nn.functional.mse_loss(reconstructed, values).item()


def main():
    config = utils.read_config("configs/config1.yaml")
    paths = config["paths"]

    root = Path(__file__).parent.parent

    output_dir = root / paths["output_dir"]
    text_history_dir = root / paths["text_history_dir"] / "history.txt"
    state_dir = root / paths["state_dir"] / "state.pt"
    plot_dir = root / paths["plot_dir"]

    max_seq_len = config["data"]["max_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = model_loader.load_model(
        config["model"]["name"],
        device=config["model"]["device"],
        dtype=getattr(torch, config["model"]["dtype"])
    )

    dataset = data.load_data(
        config["data"]["name"],
        split=config["data"]["split"]
    )

    sessions = data.extract_sessions(dataset)

    train_sessions, val_sessions, test_sessions = data.split_sessions(
        sessions, train=0.70, val=0.15, test=0.15, seed=42
    )

    num_layers = 24

    max_sample = 1000
    save_training_dir = root / "autoencoder_training_data"
    save_training_dir.mkdir(parents=True, exist_ok=True)

    split_sessions = {
        "train": train_sessions,
        "validation": val_sessions,
        "test": test_sessions,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-collect", action="store_true", help="Remove existing samples and re-collect training states")
    args = parser.parse_args()

    print("Collecting recurrent states for training autoencoders...")
    split_counts = {}
    for split_name, split in split_sessions.items():
        split_dir = save_training_dir / split_name
        split_counts[split_name] = collect_states(
            model, tokenizer, split, max_seq_len, split_dir, max_sample, force=args.force_collect
        )
        print(f"Collected {split_counts[split_name]} {split_name} state samples")

    print("Training autoencoders for each layer and latent dimension...")
    # AUTOENCODER TRAINING
    latent_dims = [1024, 2048, 4096]
    num_layers = 24

    ae_experiments = {
        ld: nn.ModuleList([
            autoencoder.Autoencoder(head_dim=64, d_state=128, hidden_dim=ld)
            for _ in range(num_layers)
        ])
        for ld in latent_dims
    }

    save_dir = root / "autoencoders"
    save_dir.mkdir(parents=True, exist_ok=True)

    for latent_dim, ae_list in ae_experiments.items():

        for layer_idx in range(num_layers):

            ae = ae_list[layer_idx].to(device)

            print("=" * 50)
            print(f"Training layer {layer_idx}, latent {latent_dim}")
            print("=" * 50)

            train_states = load_states(
                save_training_dir / "train", layer_idx, split_counts["train"]
            )
            validation_states = load_states(
                save_training_dir / "validation", layer_idx, split_counts["validation"]
            )
            test_states = load_states(
                save_training_dir / "test", layer_idx, split_counts["test"]
            )

            ae.fit(
                train_states,
                num_epochs=20,
                batch_size=256,
                device=device,
                validation_states=validation_states,
            )
            test_loss = reconstruction_loss(ae, test_states, device)
            print(f"  Test reconstruction loss: {test_loss:.6f}")

            # SAVE MODEL
            layer_dir = save_dir / f"latent_dim_{latent_dim}" / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ae.state_dict(), layer_dir / "autoencoder.pt")


if __name__ == "__main__":
    main()

