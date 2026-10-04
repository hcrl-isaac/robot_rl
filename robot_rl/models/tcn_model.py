# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Sequence
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel
from robot_rl.modules import TCNEncoder


class TCNModel(MLPModel):
    """TCN-based neural model over a stacked observation history.

    The observation set must be a time-major flattened history ``[batch, history_length * step_dim]`` (e.g. an
    observation group with ``history_length`` set on the env side). A :class:`TCNEncoder` compresses it to a
    fixed encoding, optionally concatenated with the raw most-recent step, before the MLP head. Unlike
    RNN/TXL models the history window is re-encoded every call, so the model is not recurrent.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        tcn_history_length: int | None = None,
        tcn_step_embed_dim: int = 30,
        tcn_channels: Sequence[int] = (20, 10, 10),
        tcn_kernel_sizes: Sequence[int] = (8, 5, 5),
        tcn_strides: Sequence[int] = (4, 1, 1),
        tcn_encoding_dim: int = 64,
        tcn_append_last_step: bool = True,
        memory_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the TCN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "memory").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP and the encoder.
            obs_normalization: Whether to normalize the observations before feeding them to the encoder.
            distribution_cfg: Configuration dictionary for the output distribution.
            tcn_history_length: Number of stacked steps in the observation set; the per-step dimension is
                derived from the total observation dimension, which must divide evenly.
            tcn_step_embed_dim: Output dimension of the shared per-step embedding layer.
            tcn_channels: Output channels of each conv layer.
            tcn_kernel_sizes: Kernel size of each conv layer.
            tcn_strides: Stride of each conv layer.
            tcn_encoding_dim: Dimension of the TCN encoding.
            tcn_append_last_step: Whether to concatenate the raw most-recent step to the encoding.
            memory_only: When ``True``, skip building the MLP head and output distribution. Used when this
                model is constructed as a shared memory module.
            **kwargs: Ignored extra keyword arguments accepted for cfg-class symmetry with other models.
        """
        if tcn_history_length is None:
            raise ValueError("TCNModel requires 'tcn_history_length'.")
        obs_dim = sum(obs[group].shape[-1] for group in obs_groups[obs_set])
        if obs_dim % tcn_history_length != 0:
            raise ValueError(
                f"Observation dim {obs_dim} of set '{obs_set}' is not divisible by history length {tcn_history_length}."
            )
        self._step_dim = obs_dim // tcn_history_length
        self._append_last_step = tcn_append_last_step
        self.latent_dim = tcn_encoding_dim + (self._step_dim if tcn_append_last_step else 0)

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            memory_only=memory_only,
        )

        # TCN encoder
        self.tcn = TCNEncoder(
            self._step_dim,
            tcn_history_length,
            step_embed_dim=tcn_step_embed_dim,
            channels=tcn_channels,
            kernel_sizes=tcn_kernel_sizes,
            strides=tcn_strides,
            encoding_dim=tcn_encoding_dim,
            activation=activation,
        )

    def get_latent(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Build the model latent by TCN-encoding the normalized observation history."""
        flat = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        flat = self.obs_normalizer(flat)
        latent = self.tcn(flat)
        if self._append_last_step:
            latent = torch.cat([latent, flat[..., -self._step_dim :]], dim=-1)
        if args:
            latent = torch.cat([latent, *args], dim=-1)
        return latent

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchTCNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxTCNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.latent_dim


class _ExportTCNCore(nn.Module):
    """Shared deterministic forward for TCN exports."""

    def __init__(self, model: TCNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.tcn = copy.deepcopy(model.tcn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.last_activation = copy.deepcopy(model.last_activation) or nn.Identity()
        self.step_dim = model._step_dim
        self.append_last_step = model._append_last_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on a flattened observation history."""
        x = self.obs_normalizer(x)
        latent = self.tcn(x)
        if self.append_last_step:
            latent = torch.cat([latent, x[..., -self.step_dim :]], dim=-1)
        out = self.mlp(latent)
        return self.last_activation(self.deterministic_output(out))


class _TorchTCNModel(_ExportTCNCore):
    """Exportable TCN model for JIT."""

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for TCN exports)."""
        pass


class _OnnxTCNModel(_ExportTCNCore):
    """Exportable TCN model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TCNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
