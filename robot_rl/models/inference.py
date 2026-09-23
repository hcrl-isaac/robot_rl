# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inference-policy adapters: exportable modules that chain a shared upstream module into actor inference."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from .mlp_model import MLPModel


class SharedMemoryInferencePolicy(nn.Module):
    """Adapter that chains a shared memory module into actor inference."""

    is_recurrent: bool = True

    def __init__(self, memory: nn.Module, actor: MLPModel) -> None:
        """Wrap the memory module and the actor head it feeds."""
        super().__init__()
        self.memory = memory
        self.actor = actor

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.actor.output_distribution_params

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Advance the memory and run the actor head on its latent."""
        latent = self.memory(obs)
        return self.actor.forward_from_latent(latent, *args, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset the memory (and actor) recurrent state."""
        self.memory.reset(dones)
        self.actor.reset(dones)


class EncoderInferencePolicy(nn.Module):
    """Adapter that chains a shared observation encoder into actor inference (latent as an extra input)."""

    is_recurrent: bool = False

    def __init__(self, encoder: nn.Module, actor: MLPModel) -> None:
        """Wrap the encoder module and the actor consuming its latent.

        Args:
            encoder: The shared observation encoder.
            actor: The actor model taking the encoder latent as an extra input.
        """
        super().__init__()
        self.encoder = encoder
        self.actor = actor

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.actor.output_distribution_params

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the actor with the encoder latent as an extra input."""
        return self.actor(obs, self.encoder(obs), *args, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset the actor state (the encoder is stateless)."""
        self.actor.reset(dones)

    def as_jit(self) -> _TorchEncoderPolicy:
        """Return a TorchScript-friendly copy of the encoder-actor pair."""
        return _TorchEncoderPolicy(self.encoder, self.actor)

    def as_onnx(self, verbose: bool = False) -> _OnnxEncoderPolicy:
        """Return an ONNX-export wrapper around the encoder-actor pair."""
        return _OnnxEncoderPolicy(self.encoder, self.actor, verbose)


def _encoder_export_parts(encoder: MLPModel, actor: MLPModel) -> tuple[nn.Module, ...]:
    """Deep-copy the inference-relevant submodules of an encoder-actor pair for export.

    Returns:
        ``(enc_normalizer, enc_mlp, enc_last_activation, obs_normalizer, mlp, deterministic_output,
        last_activation)``.
    """
    enc_last_activation = copy.deepcopy(encoder.last_activation) or nn.Identity()
    if actor.distribution is not None:
        deterministic_output = actor.distribution.as_deterministic_output_module()
    else:
        deterministic_output = nn.Identity()
    return (
        copy.deepcopy(encoder.obs_normalizer),
        copy.deepcopy(encoder.mlp),
        enc_last_activation,
        copy.deepcopy(actor.obs_normalizer),
        copy.deepcopy(actor.mlp),
        deterministic_output,
        copy.deepcopy(actor.last_activation) or nn.Identity(),
    )


class _TorchEncoderPolicy(nn.Module):
    """Exportable encoder+actor policy for JIT, taking one concatenated input ``[actor_obs ; encoder_obs]``."""

    def __init__(self, encoder: MLPModel, actor: MLPModel) -> None:
        """Create a TorchScript-friendly copy of an encoder and the actor consuming its latent."""
        super().__init__()
        self.obs_dim = actor.obs_dim
        (
            self.enc_normalizer,
            self.enc_mlp,
            self.enc_last_activation,
            self.obs_normalizer,
            self.mlp,
            self.deterministic_output,
            self.last_activation,
        ) = _encoder_export_parts(encoder, actor)
        # empty buffer marks a policy whose ``(mean, std)`` cannot be exported; ``forward_dist`` rejects it
        std = actor.distribution.export_std() if actor.distribution is not None else None
        self.register_buffer("_std", torch.empty(0) if std is None else std)

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        obs = x[..., : self.obs_dim]
        scan = x[..., self.obs_dim :]
        latent = self.enc_last_activation(self.enc_mlp(self.enc_normalizer(scan)))
        fused = torch.cat([self.obs_normalizer(obs), latent], dim=-1)
        return self.deterministic_output(self.mlp(fused))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on the concatenated ``[actor_obs ; encoder_obs]`` input."""
        return self.last_activation(self._mean(x))

    @torch.jit.export
    def forward_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian policy's ``(mean, std)`` for the concatenated ``[actor_obs ; encoder_obs]`` input."""
        if self._std.numel() == 0:
            raise RuntimeError("forward_dist is only supported for a plain GaussianDistribution policy.")
        mean = self._mean(x)
        return mean, self._std.expand_as(mean)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op: both the encoder and the actor are stateless)."""
        pass


class _OnnxEncoderPolicy(nn.Module):
    """Exportable encoder+actor policy for ONNX (single concatenated input, see :class:`_TorchEncoderPolicy`)."""

    is_recurrent: bool = False

    def __init__(self, encoder: MLPModel, actor: MLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an encoder and the actor consuming its latent."""
        super().__init__()
        self.verbose = verbose
        self.obs_dim = actor.obs_dim
        self.input_size = actor.obs_dim + encoder.obs_dim
        (
            self.enc_normalizer,
            self.enc_mlp,
            self.enc_last_activation,
            self.obs_normalizer,
            self.mlp,
            self.deterministic_output,
            self.last_activation,
        ) = _encoder_export_parts(encoder, actor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        obs = x[..., : self.obs_dim]
        scan = x[..., self.obs_dim :]
        latent = self.enc_last_activation(self.enc_mlp(self.enc_normalizer(scan)))
        fused = torch.cat([self.obs_normalizer(obs), latent], dim=-1)
        return self.last_activation(self.deterministic_output(self.mlp(fused)))

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
