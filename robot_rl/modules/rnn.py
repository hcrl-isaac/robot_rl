# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from robot_rl.utils import unpad_trajectories

HiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
"""Type alias for the hidden state of RNNs (GRU/LSTM).

For GRUs, this is a single tensor while for LSTMs, this is a tuple of two tensors (hidden state and cell state).
"""


class RNN(nn.Module):
    """Recurrent Neural Network.

    This network is used to store the hidden state of the policy. It currently supports GRU and LSTM.
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, num_layers: int = 1, type: str = "lstm") -> None:
        """Initialize a GRU or LSTM module with internal hidden-state storage."""
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.is_lstm = type.lower() == "lstm"
        self.hidden_state = None

    def _materialize_zero_hidden_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        """Lazily allocate ``self.hidden_state`` to zeros for the very first rollout step.

        Matches the TXL memory module. Without this, ``act()`` captures ``None`` at step 0
        of the first rollout, ``_save_hidden_states`` skips the snapshot, and the saved
        hidden-state buffer is missing one entry per env per layer. The PPO update then
        crashes with a GRU/LSTM ``Expected hidden size`` shape mismatch when slicing the
        saved buffer for a mini-batch (saved buffer is short by ``num_envs`` trajectory starts).
        """
        zeros = torch.zeros(self.rnn.num_layers, batch, self.rnn.hidden_size, device=device, dtype=dtype)
        if self.is_lstm:
            self.hidden_state = (zeros, zeros.clone())
        else:
            self.hidden_state = zeros

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Run recurrent inference in rollout mode or batched update mode."""
        batch_mode = masks is not None
        if batch_mode:
            if isinstance(hidden_state, list):
                hidden_state = tuple(hidden_state)
            out, _ = self.rnn(input, hidden_state)
            out = unpad_trajectories(out, masks)
        else:
            # Inference/distillation uses the last step's hidden state; lazy-init to zeros so a prior
            # get_hidden_state snapshot is a valid tensor, not None.
            if self.hidden_state is None:
                self._materialize_zero_hidden_state(input.shape[0], input.device, input.dtype)
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def forward_sequence(
        self, input: torch.Tensor, hidden_state: HiddenState = None, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Run a time-major ``(L, B, F)`` sequence from an explicit state; returns ``(L, B, H)`` and the final state.

        Unlike batch mode this neither unpads nor touches ``self.hidden_state``, so a caller can chain a
        no-grad burn-in segment into a training segment. ``resets`` ``(L, B)`` zeroes the state before
        the marked steps, so one window can span several episodes.
        """
        if isinstance(hidden_state, list):
            hidden_state = tuple(hidden_state)
        if resets is None:
            return self.rnn(input, hidden_state)
        out, state, _ = self._step_sequence(input, hidden_state, resets)
        return out, state

    def forward_sequence_with_states(
        self, input: torch.Tensor, hidden_state: HiddenState = None, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, HiddenState, HiddenState]:
        """Like :meth:`forward_sequence`, also returning the full state after every step, ``(L, layers, B, H)``."""
        if isinstance(hidden_state, list):
            hidden_state = tuple(hidden_state)
        return self._step_sequence(input, hidden_state, resets)

    def _step_sequence(
        self, input: torch.Tensor, hidden_state: HiddenState, resets: torch.Tensor | None
    ) -> tuple[torch.Tensor, HiddenState, HiddenState]:
        """Step the RNN one input row at a time, zeroing the state where ``resets`` is set."""
        seq_len, batch, _ = input.shape
        if hidden_state is None:
            shape = (self.rnn.num_layers, batch, self.rnn.hidden_size)
            zeros = torch.zeros(shape, device=input.device, dtype=input.dtype)
            hidden_state = (zeros, zeros.clone()) if self.is_lstm else zeros
        outs, states = [], []
        state = hidden_state
        for t in range(seq_len):
            if resets is not None:
                keep = (1.0 - resets[t]).view(1, batch, 1)
                state = tuple(x * keep for x in state) if self.is_lstm else state * keep  # type: ignore[operator]
            out, state = self.rnn(input[t : t + 1], state)
            outs.append(out)
            states.append(state)
        stacked = (
            (torch.stack([s[0] for s in states]), torch.stack([s[1] for s in states]))
            if self.is_lstm
            else torch.stack(states)  # type: ignore[arg-type]
        )
        return torch.cat(outs), state, stacked

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset hidden states for all or done environments."""
        if dones is None:  # Reset hidden state
            if hidden_state is None:
                self.hidden_state = None
            else:
                self.hidden_state = hidden_state
        elif self.hidden_state is not None:  # Reset hidden state of done environments
            if hidden_state is None:
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = 0.0  # type: ignore
                else:
                    self.hidden_state[..., dones == 1, :] = 0.0
            else:
                raise NotImplementedError(
                    "Resetting the hidden state of done environments with a custom hidden state is not implemented"
                )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states for all or done environments from the computation graph."""
        if self.hidden_state is not None:
            if dones is None:  # Detach hidden state
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    self.hidden_state = tuple(hidden_state.detach() for hidden_state in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:  # Detach hidden state of done environments
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()
