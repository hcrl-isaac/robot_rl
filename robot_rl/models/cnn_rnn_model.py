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

from robot_rl.models.cnn_model import CNNModel
from robot_rl.models.mlp_model import MLPModel
from robot_rl.models.rnn_model import RNNModel
from robot_rl.modules.rnn import HiddenState


class CNNRNNModel(RNNModel, CNNModel):
    """CNN encoders feeding a recurrent trunk, whose state joins the current features at the MLP head.

    Image groups pass through their CNN encoders and 1D groups are normalized and concatenated, as in
    :class:`CNNModel`; the joint feature drives the recurrent trunk of :class:`RNNModel`, and the head reads
    ``[feature, rnn_state]``, so the instantaneous observation reaches the head directly and memory is an
    extra input rather than a bottleneck.

    An attention pooling built with ``attention_pool_memory_dim`` reads the GRU state from before each step, so the
    CNN maps are computed for the whole window at once and only the pooling runs inside the recurrent loop.
    """

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
            rnn_hidden_dim: Recurrent hidden size.
            rnn_num_layers: Number of recurrent layers.
            aux_target_dim: Width of an auxiliary linear head on the head input, for a supervised
                prediction target (e.g. a privileged object position); ``0`` adds none.
            **kwargs: Forwarded to :class:`CNNModel` (``hidden_dims``, ``cnn_cfg``, ``distribution_cfg``, ...).
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            aux_target_dim=aux_target_dim,
            **kwargs,
        )
        memory_dims = {cnn.memory_dim for cnn in self.cnns.values()}
        self.memory_pool = memory_dims != {0}
        if self.memory_pool and (memory_dims != {rnn_hidden_dim} or rnn_type != "gru"):
            raise ValueError(
                f"A memory-conditioned pooling needs a GRU and memory_dim {rnn_hidden_dim} on every CNN, "
                f"got {rnn_type} and {sorted(memory_dims)}."
            )

    def get_latent(
        self, obs: TensorDict, *args: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Rollout-mode latent: one step through the RNN, advancing the internal hidden state."""
        if masks is not None:
            raise ValueError("CNNRNNModel batched updates go through encode_sequence, not masks")
        if not self.memory_pool:
            return super().get_latent(obs, *args)
        feats = self._features(obs, memory=self._memory(self.rnn.hidden_state, obs.batch_size[0]))
        return self._head_input(feats, self.rnn(feats).squeeze(0))

    def _features(self, obs: TensorDict, *args: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """Recurrent input for a flat ``(N,)`` batch; ``memory`` ``(N, hidden)`` shifts a memory pooling's queries."""
        if not self.memory_pool:
            return super()._features(obs, *args)
        latents = [self.cnns[g](obs[g], memory) for g in self.obs_groups_2d]
        if self.obs_groups:
            latents.insert(0, MLPModel.get_latent(self, obs))
        return torch.cat(latents, dim=-1)

    def _memory(self, state: HiddenState, batch: int) -> torch.Tensor:
        """Return the last GRU layer's state ``(N, hidden)`` the pooling reads; zeros before the first step."""
        if state is None:
            return torch.zeros(batch, self.rnn_hidden_dim, device=next(self.parameters()).device)
        return state[-1]  # type: ignore[index]

    def _memory_sequence(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step a ``(L, B)`` window, pooling each step's maps with the state before it.

        Returns:
            The head input ``(L, B, latent)``, the final state and the state after every step ``(L, layers, B, H)``.
        """
        seq_len, batch = obs.batch_size
        flat = obs.reshape(seq_len * batch)
        fmaps = [self.cnns[g](flat[g], pool=False) for g in self.obs_groups_2d]
        fmaps = [f.view(seq_len, batch, *f.shape[1:]) for f in fmaps]
        lat_1d = MLPModel.get_latent(self, flat).view(seq_len, batch, -1) if self.obs_groups else None
        state = hidden_state
        if state is None:
            state = torch.zeros(self.rnn.rnn.num_layers, batch, self.rnn_hidden_dim, device=fmaps[0].device)
        feats, outs, states = [], [], []
        for t in range(seq_len):
            if resets is not None:
                state = state * (1.0 - resets[t]).view(1, batch, 1)
            latents = [
                self.cnns[g].attention_pool(f[t], state[-1]) for g, f in zip(self.obs_groups_2d, fmaps, strict=True)
            ]
            feat = torch.cat(([lat_1d[t]] if lat_1d is not None else []) + latents, dim=-1)
            out, state = self.rnn.rnn(feat.unsqueeze(0), state)
            feats.append(feat)
            outs.append(out)
            states.append(state)
        return self._head_input(torch.stack(feats), torch.cat(outs)), state, torch.stack(states)

    def encode_sequence(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, HiddenState]:
        """Run a time-major ``(L, B)`` observation window from an explicit starting state."""
        if not self.memory_pool:
            return super().encode_sequence(obs, hidden_state, resets)
        latent, state, _ = self._memory_sequence(obs, hidden_state, resets)
        return latent, state

    def encode_sequence_with_states(
        self, obs: TensorDict, hidden_state: HiddenState, resets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, HiddenState, HiddenState]:
        """As :meth:`encode_sequence`, also returning the full state after every step, ``(L, layers, B, hidden)``."""
        if not self.memory_pool:
            return super().encode_sequence_with_states(obs, hidden_state, resets)
        return self._memory_sequence(obs, hidden_state, resets)

    def encode_step(self, obs: TensorDict, hidden_state: HiddenState) -> torch.Tensor:
        """One recurrent step for a flat ``(N,)`` batch from explicit per-sample states ``(layers, N, hidden)``."""
        if not self.memory_pool:
            return super().encode_step(obs, hidden_state)
        feats = self._features(obs, memory=self._memory(hidden_state, obs.batch_size[0]))
        out, _ = self.rnn.forward_sequence(feats.unsqueeze(0), hidden_state)
        return self._head_input(feats, out.squeeze(0))

    def step_state(self, obs: TensorDict, hidden_state: HiddenState) -> HiddenState:
        """Return the recurrent state after consuming a flat ``(N,)`` batch from explicit per-sample states."""
        if not self.memory_pool:
            return super().step_state(obs, hidden_state)
        feats = self._features(obs, memory=self._memory(hidden_state, obs.batch_size[0]))
        return self.rnn.forward_sequence(feats.unsqueeze(0), hidden_state)[1]

    def _head_input(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
        """Combine the current features and the recurrent output into the head input (concatenation)."""
        return torch.cat([feats, rnn_out], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-ready copy that carries its recurrent state between calls."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchCNNLSTMModel(self)
        return _TorchCNNGRUModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-ready copy that takes its recurrent state as an input and returns the next one."""
        return _OnnxCNNRNNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Size the head for the feature concat plus the recurrent state."""
        return self._feature_dim() + self.rnn_hidden_dim


class _CNNRNNExport(nn.Module):
    """The deterministic CNN-RNN policy, cut loose from its training wrappers.

    Exports run one environment at a time: the recurrent state has batch size 1.
    """

    def __init__(self, model: CNNRNNModel) -> None:
        super().__init__()
        cls = type(model)
        if cls._features is not CNNRNNModel._features or cls._head_input is not CNNRNNModel._head_input:
            raise NotImplementedError(
                f"{cls.__name__} changes the features or the head input; its export is not built."
            )
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.has_1d = bool(model.obs_groups)
        self.obs_dim_1d = model.obs_dim
        self.obs_groups_2d = list(model.obs_groups_2d)
        self.obs_dims_2d = [tuple(d) for d in model.obs_dims_2d]
        self.obs_channels_2d = list(model.obs_channels_2d)

    def features(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor], memory: torch.Tensor) -> torch.Tensor:
        """Build the recurrent input: normalized 1D observations, then each image group's CNN encoding.

        ``memory`` is the last GRU layer's state before this step, which a memory-conditioned pooling reads.
        """
        latents: list[torch.Tensor] = []
        if self.has_1d:
            latents.append(self.obs_normalizer(obs_1d))
        for i, cnn in enumerate(self.cnns):
            latents.append(cnn(obs_2d[i], memory))
        return torch.cat(latents, dim=-1)

    def head(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action from the current features and the recurrent output."""
        return self.deterministic_output(self.mlp(torch.cat([feats, rnn_out], dim=-1)))

    def dummy_observations(self) -> list[torch.Tensor]:
        """Return a zero observation per input, 1D first and then each image group."""
        images = [torch.zeros(1, c, h, w) for c, (h, w) in zip(self.obs_channels_2d, self.obs_dims_2d, strict=True)]
        return [torch.zeros(1, self.obs_dim_1d), *images]


class _TorchCNNGRUModel(_CNNRNNExport):
    """TorchScript export of a GRU CNN-RNN policy; ``reset()`` clears the state it carries.

    Called as ``forward(obs_1d, obs_2d)`` with ``obs_2d`` a list of one image tensor per image group, in the
    model's group order. ``obs_1d`` is always passed, and is ignored by a model without 1D groups.
    """

    def __init__(self, model: CNNRNNModel) -> None:
        super().__init__(model)
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """One step: act on this observation and carry the updated state to the next call."""
        feats = self.features(obs_1d, obs_2d, self.hidden_state[-1])
        out, h = self.rnn(feats.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        return self.head(feats, out.squeeze(0))

    @torch.jit.export
    def reset(self) -> None:
        """Zero the recurrent state, as an episode start does in training."""
        self.hidden_state[:] = 0.0


class _TorchCNNLSTMModel(_CNNRNNExport):
    """TorchScript export of an LSTM CNN-RNN policy; ``reset()`` clears the state it carries.

    Called as ``forward(obs_1d, obs_2d)`` with ``obs_2d`` a list of one image tensor per image group, in the
    model's group order. ``obs_1d`` is always passed, and is ignored by a model without 1D groups.
    """

    def __init__(self, model: CNNRNNModel) -> None:
        super().__init__(model)
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """One step: act on this observation and carry the updated state to the next call."""
        feats = self.features(obs_1d, obs_2d, self.hidden_state[-1])
        out, (h, c) = self.rnn(feats.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.head(feats, out.squeeze(0))

    @torch.jit.export
    def reset(self) -> None:
        """Zero the recurrent state, as an episode start does in training."""
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0


class _OnnxCNNRNNModel(_CNNRNNExport):
    """ONNX export of a CNN-RNN policy that takes its recurrent state as an input and returns the next one.

    Inputs are ``obs``, one per image group under the group's own name, and the state ``h_in`` (``c_in``); outputs
    are the action and the next state. The caller feeds zeros first, then each step's output state.
    """

    is_recurrent: bool = True

    def __init__(self, model: CNNRNNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.is_lstm = isinstance(self.rnn, nn.LSTM)
        reserved = {"obs", "h_in", "c_in"} & set(self.obs_groups_2d)
        if reserved:
            raise ValueError(f"Image groups named {sorted(reserved)} collide with the export's own input names.")

    def forward(self, obs: torch.Tensor, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One step from the given state; returns the action and the state after it."""
        images = list(inputs[: len(self.cnns)])
        feats = self.features(obs, images, inputs[len(self.cnns)][-1])
        if self.is_lstm:
            out, (h, c) = self.rnn(feats.unsqueeze(0), (inputs[len(self.cnns)], inputs[len(self.cnns) + 1]))
            return self.head(feats, out.squeeze(0)), h, c
        out, h = self.rnn(feats.unsqueeze(0), inputs[len(self.cnns)])
        return self.head(feats, out.squeeze(0)), h

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative inputs for tracing: zero observations and a zero state."""
        state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        return (*self.dummy_observations(), state, *((state.clone(),) if self.is_lstm else ()))

    @property
    def input_names(self) -> list[str]:
        """ONNX input names: ``obs``, the image groups', then ``h_in`` (and ``c_in``)."""
        return ["obs", *self.obs_groups_2d, "h_in", *(["c_in"] if self.is_lstm else [])]

    @property
    def output_names(self) -> list[str]:
        """ONNX output names: ``actions``, then ``h_out`` (and ``c_out``)."""
        return ["actions", "h_out", *(["c_out"] if self.is_lstm else [])]
