from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from src import data, state_utils, utils
from src.evaluate import evaluate_baseline, evaluate_injected


class StateRunner:
    """Shared state-management loop for experiment runs.

    The experiments all follow the same pattern: for a turn, either run a fresh
    baseline pass or re-use the previous state with injected decoding. This helper
    keeps that logic in one place and lets experiments customize only the
    initialization or carry-over strategy.
    """

    def __init__(
        self,
        state_dir: str | Path,
        state_seed: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
        state_loader: Optional[Callable[[str, str], tuple[torch.Tensor, torch.Tensor]]] = None,
        use_carryover: bool = False,
        carryover_update: Optional[
            Callable[
                [
                    Optional[torch.Tensor],
                    Optional[torch.Tensor],
                    torch.Tensor,
                    torch.Tensor,
                ],
                tuple[torch.Tensor, torch.Tensor],
            ]
        ] = None,
    ):
        self.state_dir = str(state_dir)
        self.state_seed = state_seed
        self.state_loader = state_loader or state_utils.load_state
        self.use_carryover = use_carryover
        self.carryover_update = carryover_update

    def _load_previous_state(self, device: str):
        """Load the state file used for non-initial turns."""
        return self.state_loader(self.state_dir, device=device)

    def _seed_state(self, prev_ssm: torch.Tensor, prev_conv: torch.Tensor):
        """Apply the configured policy to carry-over states."""
        if self.state_seed is None:
            return prev_ssm, prev_conv
        return self.state_seed(prev_ssm, prev_conv)

    def run_turn(
        self,
        model,
        tokenizer,
        snap: dict,
        device: str,
        carryover_ssm: Optional[torch.Tensor] = None,
        carryover_conv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Evaluate one snapshot and return updated states and metrics."""
        turn_id = snap["turn_id"]
        new_text = snap["new_text"]

        if turn_id == 0 and carryover_ssm is not None and self.use_carryover:
            seeded_ssm, seeded_conv = self._seed_state(carryover_ssm, carryover_conv)
            ssm_states, conv_states, latency, ppl = evaluate_injected(
                model,
                tokenizer,
                new_text,
                seeded_ssm,
                seeded_conv,
                device=device,
            )
            return ssm_states, conv_states, latency, ppl

        if turn_id == 0:
            state_output, latency, ppl = evaluate_baseline(
                model,
                tokenizer,
                snap["history_text"],
                new_text,
                device=device,
            )
            ssm_states, conv_states = state_utils.extract_state(state_output)
            return ssm_states, conv_states, latency, ppl

        prev_ssm, prev_conv = self._load_previous_state(device=device)
        ssm_states, conv_states, latency, ppl = evaluate_injected(
            model,
            tokenizer,
            new_text,
            prev_ssm,
            prev_conv,
            device=device,
        )
        return ssm_states, conv_states, latency, ppl

    def run_chain(
        self,
        model,
        tokenizer,
        sessions,
        device: str,
        carryover_ssm: Optional[torch.Tensor] = None,
        carryover_conv: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run all snapshots and collect metrics for assistant turns."""
        output_data = {}
        current_ssm = carryover_ssm
        current_conv = carryover_conv

        for session_id, session in enumerate(sessions):
            session_ssm = None
            session_conv = None
            previous_ssm = current_ssm
            previous_conv = current_conv

            for snap in data.build_turn_snapshots(session):
                ssm_states, conv_states, state_latency, state_ppl = self.run_turn(
                    model,
                    tokenizer,
                    snap,
                    device=device,
                    carryover_ssm=current_ssm,
                    carryover_conv=current_conv,
                )

                state_utils.save_state(ssm_states, conv_states, self.state_dir)
                current_ssm, current_conv = ssm_states.cpu(), conv_states.cpu()
                session_ssm, session_conv = current_ssm, current_conv

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if snap["role"] == "assistant":
                    output_data[(session_id, snap["turn_id"])] = {
                        "state_latency": state_latency,
                        "state_size_kb": utils.get_memory_size_kb(self.state_dir),
                        "state_ppl": state_ppl,
                        "turns_since_boundary": snap["turn_id"],
                        "is_first_session": session_id == 0,
                    }

            if self.carryover_update is not None and session_ssm is not None:
                current_ssm, current_conv = self.carryover_update(
                    previous_ssm,
                    previous_conv,
                    session_ssm,
                    session_conv,
                )

        return output_data