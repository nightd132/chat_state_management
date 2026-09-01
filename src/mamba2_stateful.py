import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2Mixer,
    apply_mask_to_padding_states,
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)

# Conv helper: run conv1d seeded from a saved rolling buffer

def _seeded_conv1d(
    xBC: torch.Tensor,          # (B, L, conv_dim)
    conv_state: torch.Tensor,   # (B, conv_dim, d_conv)
    conv1d: nn.Conv1d,
    activation_fn,
    d_conv: int,
):
    """Apply the mixer convolution using a saved rolling input buffer."""
    conv_dtype = conv1d.weight.dtype
    device = conv1d.weight.device

    xBC_t = xBC.transpose(1, 2).to(device=device, dtype=conv_dtype)          # (B, conv_dim, L)
    conv_state = conv_state.to(device=device, dtype=conv_dtype)              # (B, conv_dim, d_conv)

    # prepend the last d_conv-1 cached raw inputs
    prefix = conv_state[:, :, -(d_conv - 1):]                                # (B, conv_dim, d_conv-1)
    seeded = torch.cat([prefix, xBC_t], dim=-1)                              # (B, conv_dim, d_conv-1+L)

    out = F.conv1d(
        seeded,
        conv1d.weight,
        bias=conv1d.bias,
        groups=conv1d.groups,
    )                                                                        # (B, conv_dim, L)

    # updated raw-input rolling buffer
    new_conv_state = torch.cat([conv_state, xBC_t], dim=-1)[..., -d_conv:]   # (B, conv_dim, d_conv)

    return activation_fn(out).transpose(1, 2), new_conv_state


def debug_tensor(name, x):
    """Print tensor diagnostics for local debugging."""
    x = x.detach().float().cpu()
    print(f"\n{name}")
    print(" shape :", tuple(x.shape))
    print(" mean  :", x.mean().item())
    print(" std   :", x.std().item())
    print(" max   :", x.max().item())
    print(" min   :", x.min().item())
    print(" first :", x.reshape(-1)[:10])




# StatefulMamba2Mixer

class StatefulMamba2Mixer(Mamba2Mixer):

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,        # (batch, seqlen, hidden_size)
        ssm_state: torch.Tensor,            # (batch, nheads, head_dim, state_size)
        conv_state: torch.Tensor,           # (batch, conv_dim, d_conv)
        attention_mask=None,
        inject_conv=True,
        inject_ssm=True,
    ) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        if not inject_conv and not inject_ssm:
            out = self.torch_forward(
                hidden_states,
                attention_mask=attention_mask,
                cache_params=None
            )


            if isinstance(out, tuple):
                """Run one mixer block while accepting and returning explicit state."""
                out = out[0]


            return out, ssm_state, conv_state



        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        projected_states = self.in_proj(hidden_states)

        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Conv seeded from conv_state

        if inject_conv:
            hidden_states_B_C, new_conv_state = _seeded_conv1d(
                hidden_states_B_C,
                conv_state,
                self.conv1d,
                self.act,
                self.conv_kernel_size,
            )
        else:
            if attention_mask is not None:
                hidden_states_B_C = hidden_states_B_C * attention_mask.unsqueeze(-1)


            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            )


            # Build the conv state exactly as HF would after processing
            xBC_t = hidden_states_B_C.transpose(1, 2)


            new_conv_state = F.pad(
                xBC_t,
                (max(self.conv_kernel_size - xBC_t.shape[-1], 0), 0),
            )[:, :, -self.conv_kernel_size:]

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states_x, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # 3. SSM trainsformation with prior state injected 
        A = -torch.exp(self.A_log.float())

        dt_disc = F.softplus(dt + self.dt_bias)
        dt_disc = torch.clamp(dt_disc, self.time_step_limit[0], self.time_step_limit[1])

        hidden_states_f = hidden_states_x.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B_f = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C_f = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()

        B_f = B_f.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C_f = C_f.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)

        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states_f, pad_size)

        hidden_states_f = hidden_states_f * dt_disc[..., None]
        A_disc = A.to(hidden_states_f.dtype) * dt_disc
        hidden_states_f, A_disc, B_f, C_f = [
            reshape_into_chunks(t, pad_size, self.chunk_size)
            for t in (hidden_states_f, A_disc, B_f, C_f)
        ]

        A_disc = A_disc.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A_disc, dim=-1)

        L = torch.exp(segment_sum(A_disc))
        G_intermediate = C_f[:, :, :, None, :, :] * B_f[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states_f[:, :, None]).sum(dim=3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B_f * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hidden_states_f[..., None]).sum(dim=2)
        # Always inject the provided ssm_state
        if inject_ssm:
            previous_states = ssm_state.to(device=states.device, dtype=torch.float32).unsqueeze(1).float()
        else:
            previous_states = torch.zeros_like(states[:, :1])

        states = torch.cat([previous_states, states], dim=1)

        decay_chunk = torch.exp(
            segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        )
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states, new_ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C_f[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual

        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        y = y.reshape(batch_size, seq_len, -1)
        
        if self.rms_norm:
            y = self.norm(y, gate)

        # 4. Final linear projection
        out = self.out_proj(y.to(dtype))
        
        return out, new_ssm_state.to(ssm_state.dtype), new_conv_state


# Patch utility

def patch_model(model) -> int:
    """Patch Mamba blocks so they expose explicit recurrent-state forwards."""
    patched = 0
    for module in model.modules():
        if type(module).__name__ == "Mamba2Block":
            mixer = module.mixer
            if not isinstance(mixer, StatefulMamba2Mixer):
                mixer.__class__ = StatefulMamba2Mixer
                patched += 1
    if patched == 0:
        raise RuntimeError(
            "No Mamba2Block layers found — is this a Mamba2ForCausalLM model?"
        )
    return patched


# Full-model forward

@torch.no_grad()
def run_forward_with_states(
    model,
    input_ids: torch.Tensor,               # (batch, seqlen)
    ssm_states: torch.Tensor,              # (num_layers, batch, nheads, head_dim, state_size)
    conv_states: torch.Tensor,             # (num_layers, batch, conv_dim, d_conv)
) -> tuple:
    """Run the full model from explicit per-layer SSM and convolution states."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    new_ssm_list = []
    new_conv_list = []

    hidden_states = model.backbone.embeddings(input_ids)

    layer_i = 0
    for block in model.backbone.layers:
        residual = hidden_states

        normed = block.norm(hidden_states.to(block.norm.weight.dtype))
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)

        mixer_out, new_ssm, new_conv = block.mixer.forward_with_state(
            normed,
            ssm_state=ssm_states[layer_i].to(device),
            conv_state=conv_states[layer_i].to(device),
        )

        new_ssm_list.append(new_ssm.cpu().float())
        new_conv_list.append(new_conv.cpu().float())

        hidden_states = residual + mixer_out
        layer_i += 1

    model_dtype = next(model.parameters()).dtype
    hidden_states = model.backbone.norm_f(hidden_states.to(model_dtype))
    logits = model.lm_head(hidden_states)

    new_ssm_states = torch.stack(new_ssm_list, dim=0)
    new_conv_states = torch.stack(new_conv_list, dim=0)

    return logits, new_ssm_states, new_conv_states



