# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.models.cnn_model import CNNModel
from robot_rl.modules.rnn import RNN, HiddenState


class CNNRNNModel(CNNModel):
    """CNN encoders feeding a recurrent trunk, whose state joins the current features at the MLP head.

    Image groups pass through their CNN encoders and 1D groups are normalized and concatenated, as in
    :class:`CNNModel`; the joint feature drives a GRU/LSTM, and the head reads ``[feature, rnn_state]``,
    so the instantaneous observation reaches the head directly and memory is an extra input rather than
    a bottleneck. In rollout mode the hidden state is carried across steps internally. For a sequence
    update the caller passes a ``(L, B)`` observation batch and an explicit starting state via
    :meth:`encode_sequence`.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        aux_target_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the CNN-RNN model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor").
            output_dim: Dimension of the output.
            rnn_type: "gru" or "lstm".
            rnn_hidden_dim: Recurrent hidden size; also the latent width the MLP head consumes.
            rnn_num_layers: Number of recurrent layers.
            aux_target_dim: Width of an auxiliary linear head on the head input, for a supervised
                prediction target (e.g. a privileged object position); ``0`` adds none.
            **kwargs: Forwarded to :class:`CNNModel` (``hidden_dims``, ``cnn_cfg``, ``distribution_cfg``, ...).
        """
        # read by the parent's head construction, so it must exist before super().__init__
        self.rnn_hidden_dim = rnn_hidden_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.aux_head = nn.Linear(self._aux_input_dim(), aux_target_dim) if aux_target_dim > 0 else None

    def _features(self, obs: TensorDict) -> torch.Tensor:
        """CNN + normalized 1D features for a flat ``(N,)`` batch, shape ``(N, F)``."""
        return super().get_latent(obs)

    def get_latent(
        self, obs: TensorDict, *args: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Rollout-mode latent: one step through the RNN, advancing the internal hidden state."""
        if masks is not None:
            raise ValueError("CNNRNNModel batched updates go through encode_sequence, not masks")
        feats = self._features(obs)
        return self._head_input(feats, self.rnn(feats).squeeze(0))

    def encode_sequence(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Run a time-major ``(L, B)`` observation window from an explicit starting state.

        Args:
            obs: Observations with batch size ``(L, B)``.
            hidden_state: State before the first step, ``(layers, B, hidden)``; ``None`` starts from zeros.
            resets: ``(L, B)`` flags zeroing the state before the marked steps (episode starts).

        Returns:
            The per-step head input ``(L, B, feature + hidden)`` and the state after the last step.
        """
        feats = self._sequence_features(obs)
        out, state = self.rnn.forward_sequence(feats, hidden_state, resets)
        return self._head_input(feats, out), state

    def encode_sequence_with_states(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, HiddenState, HiddenState]:
        """As :meth:`encode_sequence`, also returning the full state after every step, ``(L, layers, B, hidden)``."""
        feats = self._sequence_features(obs)
        out, state, states = self.rnn.forward_sequence_with_states(feats, hidden_state, resets)
        return self._head_input(feats, out), state, states

    def encode_step(self, obs: TensorDict, hidden_state: HiddenState) -> torch.Tensor:
        """One recurrent step for a flat ``(N,)`` batch from explicit per-sample states ``(layers, N, hidden)``."""
        feats = self._features(obs)
        out, _ = self.rnn.forward_sequence(feats.unsqueeze(0), hidden_state)
        return self._head_input(feats, out.squeeze(0))

    def step_state(self, obs: TensorDict, hidden_state: HiddenState) -> HiddenState:
        """Return the recurrent state after consuming a flat ``(N,)`` batch from explicit per-sample states."""
        _, state = self.rnn.forward_sequence(self._features(obs).unsqueeze(0), hidden_state)
        return state

    def aux_prediction(self, latent: torch.Tensor) -> torch.Tensor:
        """Auxiliary-head prediction from a flat head input ``(N, latent)``."""
        return self.aux_head(latent[:, : self._aux_input_dim()])  # type: ignore[misc]

    def _aux_input_dim(self) -> int:
        """Width of the head input the auxiliary head reads (the full head input by default)."""
        return self._get_latent_dim()

    def _head_input(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
        """Combine the current features and the recurrent output into the head input (concatenation)."""
        return torch.cat([feats, rnn_out], dim=-1)

    def _sequence_features(self, obs: TensorDict) -> torch.Tensor:
        """CNN + normalized 1D features for an ``(L, B)`` window, shape ``(L, B, F)``."""
        seq_len, batch = obs.batch_size
        return self._features(obs.reshape(seq_len * batch)).view(seq_len, batch, -1)

    def act_and_log_prob(
        self,
        obs: TensorDict,
        *args: torch.Tensor,
        std_clip: float | None = None,
        hidden_state: HiddenState = None,
        sequence: bool = False,
        resets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and its log-prob; in sequence mode over an ``(L, B)`` window from ``hidden_state``.

        Sequence-mode outputs come back time-major, ``(L, B, A)`` and ``(L, B)``.
        """
        if not sequence:
            return super().act_and_log_prob(obs, *args, std_clip=std_clip)
        seq_len, batch = obs.batch_size
        latent, _ = self.encode_sequence(obs, hidden_state, resets)
        action, log_prob = self.act_and_log_prob_from_latent(latent.reshape(seq_len * batch, -1), std_clip=std_clip)
        return action.view(seq_len, batch, -1), log_prob.view(seq_len, batch)

    def act_and_log_prob_from_latent(
        self, latent: torch.Tensor, std_clip: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and its log-prob from an already computed flat ``(N, hidden)`` latent."""
        self.distribution.update(self.mlp(latent))  # type: ignore
        return self.distribution.sample_and_log_prob(std_clip=std_clip)  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the rollout hidden state for all, or only the done, environments."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self, batch_size: int | None = None, device: torch.device | None = None) -> HiddenState:
        """Return the rollout hidden state, materializing zeros on the very first step if needed."""
        if self.rnn.hidden_state is None and batch_size is not None:
            dev = device if device is not None else next(self.parameters()).device
            self.rnn._materialize_zero_hidden_state(batch_size, dev, next(self.parameters()).dtype)
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the rollout hidden state from the graph."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Not supported yet: the recurrent export wrappers assume 1D inputs."""
        raise NotImplementedError("CNNRNNModel export is not implemented yet")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Not supported yet: the recurrent export wrappers assume 1D inputs."""
        raise NotImplementedError("CNNRNNModel export is not implemented yet")

    def _get_latent_dim(self) -> int:
        """Size the head for the feature concat plus the recurrent state."""
        return super()._get_latent_dim() + self.rnn_hidden_dim
