from datasets import load_dataset

def load_data(dataset_name: str, split: str = "train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def extract_sessions(dataset):
    sessions = []

    for sample in dataset:
        dialog = sample["responses_create_params"]["input"]
        sessions.append(dialog)

    return sessions

def build_turn_snapshots(session):
    snapshots = []

    history_text = ""
    turn_id = 0

    for turn in session:
        role = turn["role"]
        content = turn["content"]

        line = f"{role.capitalize()}: {content}\n"
        history_text += line

        snapshots.append({
            "turn_id": turn_id,
            "role": role,
            "new_text": line,
            "history_text": history_text
        })

        turn_id += 1

    return snapshots