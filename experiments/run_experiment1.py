import yaml
import torch
from data.data_loader import load_data
from models.model_loader import load_model
from storage.state_io import save_state, load_state, feed_synthetic_ssm_state
from benchmarks.latency_memory_bench import measure_baseline_latency, measure_state_management_latency, print_benchmark_results

def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = read_config("configs/config1.yaml")
    
    model, tokenizer = load_model(config["model"]["name"], 
                                  device=config["model"]["device"], 
                                  dtype=getattr(torch, config["model"]["dtype"]))
    
    dataset = load_data(config["data"]["name"], 
                        split=config["data"]["split"])

    dialogs = dataset["responses_create_params"]["input"] # role, content

    for session in dialogs:
        input = ""
        for turn in session:
            if turn["role"] == "system":
                input += f"System: {turn['content']}\n"
            elif turn["role"] == "user":
                input += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                input += f"Assistant: {turn['content']}\n"
            


if __name__ == "__main__":
    main()
    
    
    