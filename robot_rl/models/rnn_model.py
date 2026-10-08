# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel
from robot_rl.modules import RNN, HiddenState


class RNNModel(MLPModel):
    """RNN-based neural model.

    The base model's latent (normalized 1D observation groups here, plus CNN encodings in :class:`CNNRNNModel`)
    drives an LSTM/GRU whose output feeds the MLP head. The output can be deterministic or stochastic, in which case
    a distribution module samples it. In rollout mode the hidden state is carried across steps internally; a sequence
    update passes a time-major ``(L, B)`` window and an explicit starting state to :meth:`encode_sequence`.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        aux_target_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            rnn_type: Type of RNN to use ("lstm" or "gru").
            rnn_hidden_dim: Dimension of the RNN hidden state.
            rnn_num_layers: Number of RNN layers.
            aux_target_dim: Width of an auxiliary linear head on the head input, for a supervised
                prediction target (e.g. a privileged object position); ``0`` adds none.
            **kwargs: Forwarded to the base model (``hidden_dims``, ``distribution_cfg``, ``memory_only``, ...).
        """
        # read by the head construction in the base __init__, so it must exist first
        self.rnn_hidden_dim = rnn_hidden_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.rnn = RNN(self._feature_dim(), rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.aux_head = nn.Linear(self._aux_input_dim(), aux_target_dim) if aux_target_dim > 0 else None

    @property
    def latent_dim(self) -> int:
        """Width of the latent the MLP head reads, which a shared-memory head is sized from."""
        return self._get_latent_dim()

    def get_latent(
        self, obs: TensorDict, *args: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Rollout-mode latent advancing the internal hidden state, or a padded trajectory batch with ``masks``."""
        feats = self._features(obs, *args)
        return self._head_input(feats, self.rnn(feats, masks, hidden_state).squeeze(0))

    def encode_sequence(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, HiddenState]:
        """Run a time-major ``(L, B)`` observation window from an explicit starting state.

        Args:
            obs: Observations with batch size ``(L, B)``.
            hidden_state: State before the first step, ``(layers, B, hidden)``; ``None`` starts from zeros.
            resets: ``(L, B)`` flags zeroing the state before the marked steps (episode starts).

        Returns:
            The per-step head input ``(L, B, latent)`` and the state after the last step.
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
        """Sample an action and its log-prob from an already computed flat ``(N, latent)`` head input."""
        self.distribution.update(self.mlp(latent))  # type: ignore
        return self.distribution.sample_and_log_prob(std_clip=std_clip)  # type: ignore

    def aux_prediction(self, latent: torch.Tensor) -> torch.Tensor:
        """Auxiliary-head prediction from a flat head input ``(N, latent)``."""
        return self.aux_head(latent[:, : self._aux_input_dim()])  # type: ignore[misc]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state for all, or only the done, environments."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self, batch_size: int | None = None, device: torch.device | None = None) -> HiddenState:
        """Return the rollout hidden state, materializing zeros on the very first step if needed.

        PPO snapshots the pre-step state on its first rollout step, so the state must exist before any forward.
        """
        if self.rnn.hidden_state is None and batch_size is not None:
            dev = device if device is not None else next(self.parameters()).device
            self.rnn._materialize_zero_hidden_state(batch_size, dev, next(self.parameters()).dtype)
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

    def _features(self, obs: TensorDict, *args: torch.Tensor) -> torch.Tensor:
        """Recurrent input for a flat ``(N,)`` batch: the base model's latent, shape ``(N, F)``."""
        return super().get_latent(obs, *args)

    def _feature_dim(self) -> int:
        """Width of the recurrent input: the base model's latent width."""
        return super()._get_latent_dim()

    def _sequence_features(self, obs: TensorDict) -> torch.Tensor:
        """Recurrent input for an ``(L, B)`` window, shape ``(L, B, F)``."""
        seq_len, batch = obs.batch_size
        return self._features(obs.reshape(seq_len * batch)).view(seq_len, batch, -1)

    def _head_input(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
        """Combine the current features and the recurrent output into the head input (the recurrent output)."""
        return rnn_out

    def _aux_input_dim(self) -> int:
        """Width of the head input the auxiliary head reads (the full head input by default)."""
        return self._get_latent_dim()

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.rnn_hidden_dim


class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of a GRU-based RNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU inference step and update hidden states."""
        x = self.obs_normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported GRU hidden states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based RNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LSTM inference step and update hidden and cell states."""
        x = self.obs_normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(obs)

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
