import torch


class StateCaptureWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_state = None

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)

        self.last_state = self.extract_state(outputs)

        return outputs

    def extract_state(self, outputs):
        if hasattr(outputs, "cache_params"):
            cache = outputs.cache_params
            if hasattr(cache, "ssm_states"):
                return cache.ssm_states

        # if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        #     return outputs.hidden_states

        return None