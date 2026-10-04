from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import warnings
from collections.abc import Sequence
from tensordict import TensorDict
from typing import Any

from robot_rl.modules import MLP, EmpiricalNormalization, ResMLP
from robot_rl.modules.distribution import Distribution
from robot_rl.utils import resolve_callable

from .mlp_model import MLPModel


class FuseModel(MLPModel):
    """Early fusion MLP-based neural model."""

    head_module: str = "trunk"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        input_dims: Sequence[int],
        output_dim: int,
        embedding_dims: Sequence[int] = (256, 256, 256),
        hidden_dims: Sequence[int] = (256, 256, 256),
        num_parallel: int = 1,
        activation: str = "relu",
        first_activation: str | None = "tanh",
        last_activation: str | None = None,
        obs_normalization: bool = False,
        normalize_first_layer: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the fusion model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to list of observation groups.
            obs_set: Observation set to use for this model (e.g. "actor" or "critic").
            input_dims: A sequence of input dimensions, where each input is concatenated with observations.
            output_dim: Dimension of the output.
            embedding_dims: Hidden dimensions of the input embeddings.
            hidden_dims: Hidden dimension of the model trunk.
            num_parallel: Number of parallel networks.
            activation: Activation function of the model.
            first_activation: Activation function of the first layer of the model.
            last_activation: Activation function of the model output.
            obs_normalization: Whether to normalize the observations before feeding them to the model.
            normalize_first_layer: Whether to normalize the output of the first layer with LayerNorm.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
        """
        nn.Module.__init__(self)

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            model_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            model_output_dim = output_dim

        assert hidden_dims and embedding_dims, "hidden_dims and embedding_dims must both have at least one layer."
        self.num_parallel = num_parallel

        trunk_input_dim = hidden_dims[0] if hidden_dims else output_dim
        embedding_output_dim = trunk_input_dim // len(input_dims)

        # construct modules for each embedding term
        self.embeddings = nn.ModuleList([
            self._make_embedding(
                self.obs_dim + dim,
                embedding_output_dim,
                embedding_dims,
                num_parallel,
                activation,
                first_activation,
                normalize_first_layer,
            )
            for dim in input_dims
        ])
        self.input_dims = input_dims
        self.num_inputs = int(np.sum(np.greater(self.input_dims, 0)))

        # construct layers for main model
        self.trunk = self._make_trunk(
            trunk_input_dim, model_output_dim, hidden_dims, num_parallel, activation, last_activation
        )

        # Initialize distribution-specific weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.trunk)

    def init_weights(self) -> None:
        """Initialize all MLP weights with orthogonal initialization and re-apply distribution-specific init."""
        for emb in self.embeddings:
            emb.init_weights(1.0)
        self.trunk.init_weights(1.0)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.trunk)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT (trace) export."""
        return _TorchFuseModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _TorchFuseModel(self)

    def forward(
        self,
        obs: TensorDict,
        *args: torch.Tensor,
        stochastic_output: bool = False,
        std_clip: float | None = None,
    ) -> torch.Tensor:
        """Forward pass of the fuse model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        if len(args) != self.num_inputs:
            raise ValueError(
                f"Invalid number of inputs received. Expected {self.num_inputs}, but received {len(args)}."
            )
        # Get model input latent
        latent = self.get_latent(obs, *args)
        # Embed each input
        embed_output = [e(x) for e, x in zip(self.embeddings, latent, strict=True)]
        # Trunk forward pass
        trunk_output = self.trunk(torch.cat(embed_output, dim=-1))
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(trunk_output)
                return self.distribution.sample(std_clip=std_clip)
            return self.distribution.deterministic_output(trunk_output)
        return trunk_output

    def get_latent(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Build the model latent.

        Latent is constructed by concatenating and normalizing selected observation groups, then concatenating with
        each provided embedding input. If using parallel networks, each latent is expanded to (num_parallel, ...).
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        latent = self.obs_normalizer(latent)
        # Concatenate with provided embedding inputs
        latents: list[torch.Tensor] = []
        arg_idx = 0
        for dim in self.input_dims:
            new_latent: torch.Tensor = latent
            # Only concatenate if additional input dimension is nonzero
            if dim > 0:
                new_latent = torch.cat([latent, args[arg_idx]], dim=-1)
                arg_idx += 1
            # Reshape for parallel models
            if self.num_parallel > 1:
                new_latent.expand(self.num_parallel, -1, -1)
            latents.append(new_latent)
        return tuple(latents)

    def _make_embedding(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dims: Sequence[int],
        num_parallel: int,
        activation: str,
        first_activation: str | None,
        normalize_first_layer: bool,
    ) -> nn.Module:
        return MLP(
            input_dim,
            output_dim,
            embedding_dims,
            num_parallel,
            activation,
            first_activation=first_activation,
            last_activation=activation,
            normalize_first_layer=normalize_first_layer,
        )

    def _make_trunk(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str,
        last_activation: str | None,
    ) -> nn.Module:
        return MLP(input_dim, output_dim, hidden_dims, num_parallel, activation, last_activation=last_activation)


class ResidualFuseModel(FuseModel):
    """Early fusion residual neural model."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        input_dims: Sequence[int],
        output_dim: int,
        embedding_dims: Sequence[int] = (256, 256, 256),
        hidden_dims: Sequence[int] = (256, 256, 256),
        num_parallel: int = 1,
        activation: str = "mish",
        first_activation: str | None = None,
        last_activation: str | None = None,
        obs_normalization: bool = False,
        normalize_first_layer: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the residual fusion model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to list of observation groups.
            obs_set: Observation set to use for this model (e.g. "actor" or "critic").
            input_dims: A sequence of input dimensions, where each input is concatenated with observations.
            output_dim: Dimension of the output.
            embedding_dims: Hidden dimensions of the input embeddings.
            hidden_dims: Hidden dimension of the model trunk.
            num_parallel: Number of parallel networks.
            activation: Activation function of the model.
            first_activation: Activation function of the first layer of the model (no-op for residual model).
            last_activation: Activation function of the model output.
            obs_normalization: Whether to normalize the observations before feeding them to the model.
            normalize_first_layer: Whether to normalize the output of the first layer with LayerNorm (no-op for
                residual model).
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
        """
        # We redefine init here to change defaults (e.g. default activation is Mish for residual networks)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            input_dims,
            output_dim,
            embedding_dims,
            hidden_dims,
            num_parallel,
            activation,
            first_activation,
            last_activation,
            obs_normalization,
            normalize_first_layer,
            distribution_cfg,
        )

    def _make_embedding(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dims: Sequence[int],
        num_parallel: int,
        activation: str,
        first_activation: str | None,
        normalize_first_layer: bool,
    ) -> nn.Module:
        if first_activation is not None:
            warnings.warn(f"First activation '{first_activation}' will be ignored for residual embedding.")
        if normalize_first_layer:
            warnings.warn("normalize_first_layer is True, but will be ignored for residual embedding.")
        return ResMLP(
            input_dim,
            output_dim,
            embedding_dims,
            num_parallel,
            activation,
            last_activation=activation,
            first_residual=False,
        )

    def _make_trunk(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str,
        last_activation: str | None,
    ) -> nn.Module:
        return ResMLP(
            input_dim, output_dim, hidden_dims, num_parallel, activation, last_activation, first_residual=True
        )


class _TorchFuseModel(nn.Module):
    """Exportable fuse model: deterministic ``forward(obs, *inputs)`` on plain tensors.

    Observations are pre-concatenated. Serves both TorchScript (traced; see ``jit_trace``) and ONNX.
    """

    jit_trace = True
    """FuseModel submodules don't script; exporters must trace with :meth:`get_dummy_inputs`."""

    def __init__(self, model: FuseModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.embeddings = copy.deepcopy(model.embeddings)
        self.trunk = copy.deepcopy(model.trunk)
        self.input_dims = list(model.input_dims)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.obs_dim = int(model.obs_dim)
        self.other_dims = [int(dim) for dim in model.input_dims if dim > 0]

    def forward(self, obs: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(obs)
        embed_outputs = []
        arg_idx = 0
        for embedding, dim in zip(self.embeddings, self.input_dims, strict=True):
            x = latent
            if dim > 0:
                x = torch.cat([latent, inputs[arg_idx]], dim=-1)
                arg_idx += 1
            embed_outputs.append(embedding(x))
        return self.deterministic_output(self.trunk(torch.cat(embed_outputs, dim=-1)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.obs_dim), *(torch.zeros(1, dim) for dim in self.other_dims))

    @property
    def input_names(self) -> list[str]:
        return ["obs", *(f"input_{i}" for i in range(len(self.other_dims)))]

    @property
    def output_names(self) -> list[str]:
        return ["output"]
