from datasets import load_dataset

def load_data(dataset_name: str, split: str = "train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def get_input_from_dataset(dataset, use_saved_state: bool = False):
    innstruction = ""
    input = ""
    dialogs = dataset["responses_create_params"]["input"] # role, content
    for session in dialogs:
        innstruction = ""
        input = ""
        for turn in session:
            if turn["role"] == "system":
                innstruction += f"System: {turn['content']}\n"
            elif turn["role"] == "user":
                input += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                input += f"Assistant: {turn['content']}\n"
    return input