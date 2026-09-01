from datasets import load_dataset
import random
from collections import defaultdict
import numpy as np

def load_data(dataset_name: str, split: str = "train"):
    """Load one split of a Hugging Face dataset."""
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def extract_sessions(dataset):
    """Extract conversation turn lists from dataset records."""
    sessions = []
    for sample in dataset:
        dialog = sample["responses_create_params"]["input"]
        sessions.append(dialog)
    return sessions


def split_sessions(sessions, train=0.70, val=0.15, test=0.15, seed=42):
    """Shuffle and split sessions into reproducible train/validation/test sets."""
    rng = random.Random(seed)
    indices = list(range(len(sessions)))
    rng.shuffle(indices)

    n = len(sessions)
    n_train = int(n * train)
    n_val   = int(n * val)
    
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_sessions = [sessions[i] for i in train_idx]
    val_sessions   = [sessions[i] for i in val_idx]
    test_sessions  = [sessions[i] for i in test_idx]

    print(f"Split: {len(train_sessions)} train / "
          f"{len(val_sessions)} val / "
          f"{len(test_sessions)} test  "
          f"(seed={seed})")
    return train_sessions, val_sessions, test_sessions


def build_turn_snapshots(session):
    """Build cumulative-history snapshots for each turn in a conversation."""
    snapshots = []
    history_text = ""
    turn_id = 0

    for turn in session:
        role    = turn["role"]
        content = turn["content"]
        line    = f"{role}: {content}\n"
        history_text += line

        snapshots.append({
            "turn_id":      turn_id,
            "role":         role,
            "new_text":     line,
            "history_text": history_text,
        })
        turn_id += 1

    return snapshots


def aggregate_turn_results(all_session_results):
    """Compute per-turn means and standard deviations across sessions."""

    # Collect values per turn per metric
    turn_metric_values = defaultdict(lambda: defaultdict(list))
    for session_result in all_session_results:
        for turn_id, metrics in session_result.items():
            for metric, value in metrics.items():
                turn_metric_values[turn_id][metric].append(value)

    aggregated = {}
    for turn_id in sorted(turn_metric_values):
        aggregated[turn_id] = {}
        n = None
        for metric, values in turn_metric_values[turn_id].items():
            aggregated[turn_id][f"{metric}_mean"] = float(np.mean(values))
            aggregated[turn_id][f"{metric}_std"]  = float(np.std(values))
            n = len(values)
        aggregated[turn_id]["n_sessions"] = n

    return aggregated

