from src import model_loader, state_utils, evaluate as evaluate_module, data, autoencoder, plot, utils

import copy
import yaml
import torch
import pandas as pd
import csv
import time
import torch.nn as nn
import numpy as np
from pathlib import Path


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

    list_of_snapshots = []

    for session in sessions[1000:]:
        train_snapshots = data.build_turn_snapshots(session)
        list_of_snapshots.append(train_snapshots)

    num_layers = 24

    count = 0
    max_sample = 1000
    stop = False

    save_training_dir = root / "autoencoder_training_data"
    save_training_dir.mkdir(parents=True, exist_ok=True)

    # DATA COLLECTION
    for idx, snapshots in enumerate(list_of_snapshots):
        for idx_snap, snap in enumerate(snapshots):

            if snap["role"] == "assistant":
                inputs = tokenizer(
                    snap["history_text"],
                    max_length=max_seq_len, truncation=True,
                    return_tensors="pt"
                ).to(device)
    

                with torch.no_grad():
                    output = model(inputs["input_ids"], use_cache=True)

                cache = output.cache_params

                for layer_idx, layer in enumerate(cache.layers):

                    layer_dir = save_training_dir / f"layer_{layer_idx}"
                    layer_dir.mkdir(parents=True, exist_ok=True)

                    state = layer.recurrent_states.squeeze(0).cpu()

                    save_path = layer_dir / f"sample_{count:06d}.pt"

                    torch.save({"state": state}, save_path)

                count += 1

            if count >= max_sample:
                stop = True
                print(f"Snap: {idx_snap}")
                break

        if stop:
            print(f"Snapshot: {idx}")
            break


    # AUTOENCODER TRAINING
    latent_dims = [256, 512, 1024]
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

            # load dataset
            all_states = []

            for idx in range(max_sample):
                sample = torch.load(
                    save_training_dir / f"layer_{layer_idx}" / f"sample_{idx:06d}.pt"
                )
                all_states.append(sample["state"])

            states = torch.stack(all_states).reshape(len(all_states), -1).float().to(device)

            # TRAIN
            loss = ae.fit(states, num_epochs=20, batch_size=256, device=device)

            # SAVE MODEL
            layer_dir = save_dir / f"latent_dim_{latent_dim}" / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ae.state_dict(), layer_dir / "autoencoder.pt")


if __name__ == "__main__":
    main()


