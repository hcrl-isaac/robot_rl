# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from collections.abc import Sequence
from tensordict import TensorDict
from typing import Any

from robot_rl.modules import MLP, EmpiricalNormalization, HiddenState
from robot_rl.modules.distribution import Distribution
from robot_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model.

    This model uses a simple multi-layer perceptron (MLP) to process 1D observation groups. Observations can be
    normalized before being passed to the MLP. The output of the model can be either deterministic or
    stochastic, in which case a distribution module is used to sample the outputs.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    loads_own_weights: bool = False
    """Whether the model's weights come from its own configuration rather than from a checkpoint."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        other_input_dims: Sequence[int] = (),
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        first_activation: str | None = None,
        last_activation: str | None = None,
        obs_normalization: bool = False,
        normalize_first_layer: bool = False,
        distribution_cfg: dict | None = None,
        input_dim_override: int | None = None,
        append_obs_groups: bool = False,
        memory_only: bool = False,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            other_input_dims: Dimensions of auxiliary inputs to be concatenated with the observation latent.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            first_activation: Activation function of the first layer. None uses the default model activation.
            last_activation: Activation applied to the model output in :meth:`forward`, skippable at call time via
                ``raw_output`` and baked into the export. None leaves the output linear.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            normalize_first_layer: Whether to normalize the first layer output with LayerNorm.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            input_dim_override: When set, bypass observation extraction and size the MLP head from this dimension
                instead of ``obs_dim``. Used by PPO when actor/critic act as heads on top of a shared memory module
                that produces a precomputed latent. Forward must then be called via :meth:`forward_from_latent`.
            append_obs_groups: When ``True`` alongside ``input_dim_override``, this model still resolves its own
                observation groups and appends them to the precomputed latent at the head (sized as
                ``input_dim_override + obs_dim``). :meth:`forward_from_latent` then extracts them when passed
                ``obs=...``. Used for an asymmetric critic head that consumes privileged obs the actor does not see.
            memory_only: When ``True``, skip building the MLP head and output distribution. Used when a
                memory-bearing model (RNN/TXL) is constructed solely to provide a shared latent to downstream
                heads -- :meth:`forward` then returns the memory module's latent directly.
        """
        super().__init__()

        # Output activation is applied in ``forward`` (gated by ``raw_output``) and baked into the export,
        # rather than into the MLP head, so callers can request the pre-activation output.
        self.last_activation = resolve_nn_activation(last_activation) if last_activation is not None else None

        # Head-only mode bypasses observation handling entirely; the latent is produced upstream. With
        # ``append_obs_groups`` the head still resolves its obs groups to append them to that latent.
        self._input_dim_override = input_dim_override
        self._append_obs_groups = append_obs_groups
        self.memory_only = memory_only
        if input_dim_override is None or append_obs_groups:
            # Resolve observation groups and dimensions
            self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        else:
            self.obs_groups: list[str] = []
            self.obs_dim = 0

        # Resolve total number of inputs and dimensions
        self.num_inputs = 1 + len(other_input_dims)
        self.other_input_dim = int(np.sum(other_input_dims))

        # Observation normalization
        self.obs_normalization = obs_normalization and input_dim_override is None
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Memory-only mode excludes the MLP and distribution (only uses latent)
        if memory_only:
            self.distribution = None
            self.mlp = torch.nn.Identity()
            return

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP
        self.mlp = MLP(
            self._get_latent_dim(),
            mlp_output_dim,
            hidden_dims,
            activation=activation,
            first_activation=first_activation,
            normalize_first_layer=normalize_first_layer,
        )

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        *args: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        std_clip: float | None = None,
        raw_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default. ``raw_output`` skips ``last_activation`` and returns the pre-activation output.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, *args, masks=masks, hidden_state=hidden_state)
        # Memory-only models stop here: their output is the memory latent, consumed by downstream heads
        if self.memory_only:
            return latent
        # MLP forward pass
        mlp_output = self.mlp(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample(std_clip=std_clip)
            return self.distribution.deterministic_output(mlp_output)
        if self.last_activation is not None and not raw_output:
            mlp_output = self.last_activation(mlp_output)
        return mlp_output

    def forward_from_latent(
        self,
        latent: torch.Tensor,
        *args: torch.Tensor,
        obs: TensorDict | None = None,
        masks: torch.Tensor | None = None,
        stochastic_output: bool = False,
        std_clip: float | None = None,
    ) -> torch.Tensor:
        """Apply the MLP head and output distribution to a precomputed latent.

        Used when this model serves as a head on top of a shared memory module that produces the latent upstream.
        Optionally accepts extra observations, which are concatenated with the memory latent.
        """
        if self.obs_groups and obs is not None:
            obs_tensor = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            if masks is not None:
                obs_tensor = unpad_trajectories(obs_tensor, masks)
            latent = torch.cat([latent, obs_tensor], dim=-1)
        if args:
            latent = torch.cat([latent, *args], dim=-1)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample(std_clip=std_clip)
            return self.distribution.deterministic_output(mlp_output)
        if self.last_activation is not None:
            mlp_output = self.last_activation(mlp_output)
        return mlp_output

    def get_latent(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups and additional inputs."""
        # Select and concatenate observations
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        # Normalize observations
        latent = self.obs_normalizer(latent)
        # Add additional tensor input
        if args:
            latent = torch.cat([latent, *args], dim=-1)
        return latent

    def init_weights(self) -> None:
        """Initialize all MLP weights with orthogonal initialization and re-apply distribution-specific init."""
        self.mlp.init_weights(1.0)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def act_and_log_prob(
        self, obs: TensorDict, *args: torch.Tensor, std_clip: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a stochastic action and its log-prob from a single (reparameterized) draw.

        Updates the output distribution from a forward pass and returns ``(action, log_prob)`` derived from the
        same draw. Used by off-policy algorithms (e.g. SAC) that need the action and its log-prob together for the
        entropy term; requires a distribution implementing :meth:`sample_and_log_prob`
        (e.g. ``SquashedTanhGaussianDistribution``).
        """
        latent = self.get_latent(obs, *args)
        mlp_output = self.mlp(latent)
        self.distribution.update(mlp_output)  # type: ignore
        return self.distribution.sample_and_log_prob(std_clip=std_clip)  # type: ignore

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"{self.__class__.__name__} only supports 1D observations, got shape {obs[obs_group].shape} for "
                    f"'{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        base_dim = self._input_dim_override if self._input_dim_override is not None else 0
        return base_dim + self.obs_dim + self.other_input_dim


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: MLPModel) -> None:
        """Create a TorchScript-friendly copy of an MLPModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.last_activation = copy.deepcopy(model.last_activation) or nn.Identity()
        # Empty buffer marks a policy whose ``(mean, std)`` cannot be exported; ``forward_dist`` rejects it.
        std = model.distribution.export_std() if model.distribution is not None else None
        self.register_buffer("_std", torch.empty(0) if std is None else std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.last_activation(self.deterministic_output(out))

    @torch.jit.export
    def forward_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian policy's ``(mean, std)`` for pre-concatenated observations."""
        if self._std.numel() == 0:
            raise RuntimeError("forward_dist is only supported for a plain GaussianDistribution policy.")
        x = self.obs_normalizer(x)
        mean = self.deterministic_output(self.mlp(x))
        return mean, self._std.expand_as(mean)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.last_activation = copy.deepcopy(model.last_activation) or nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.last_activation(self.deterministic_output(out))

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
