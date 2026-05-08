import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer