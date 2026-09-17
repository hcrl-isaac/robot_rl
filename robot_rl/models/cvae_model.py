# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections.abc import Sequence
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel
from robot_rl.modules import HiddenState


class CVAEModel(nn.Module):
    """Conditional-VAE behavior model: a learned prior, a residual encoder and a fixed-variance decoder.

    ``prior(obs_prior) -> N(mu_p, std_p)``, ``encoder(obs_encoder) -> N(mu_p + d_mu, std_q)`` and
    ``decoder(obs_decoder, z) -> action``. The decoder never sees the goal, so task information has to travel
    through ``z``; rollouts, exports and downstream high-level policies use the prior path. The three observation
    sets come from ``obs_groups`` (``obs_set`` names the decoder's), and the decoder's groups must be a prefix of
    the prior's so a single concatenated input serves both at export time.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        latent_dim: int = 32,
        prior_hidden_dims: Sequence[int] = (512, 512, 512),
        encoder_hidden_dims: Sequence[int] = (1024, 1024, 512),
        decoder_hidden_dims: Sequence[int] = (1024, 1024, 512),
        activation: str = "elu",
        obs_normalization: bool = True,
        fixed_std: float = 0.05,
        min_std: float = 1e-3,
        max_std: float = 2.0,
        prior_obs_set: str = "prior",
        encoder_obs_set: str = "encoder",
        **kwargs: Any,
    ) -> None:
        """Build the three sub-networks.

        Args:
            obs: Observation dictionary.
            obs_groups: Observation sets; must hold ``obs_set``, ``prior_obs_set`` and ``encoder_obs_set``.
            obs_set: The decoder's observation set (the real-world proprioception).
            output_dim: Action dimension.
            latent_dim: Dimension of ``z``.
            prior_hidden_dims: Hidden layers of the prior MLP.
            encoder_hidden_dims: Hidden layers of the encoder MLP.
            decoder_hidden_dims: Hidden layers of the decoder MLP.
            activation: Activation of every MLP.
            obs_normalization: Whether each sub-network normalizes its observations.
            fixed_std: The decoder's fixed output std (rollout exploration noise only).
            min_std: Lower clamp on the prior and encoder stds.
            max_std: Upper clamp on the prior and encoder stds.
            prior_obs_set: Observation set of the prior.
            encoder_obs_set: Observation set of the encoder (may include privileged groups).
            **kwargs: Unused model-config keys.
        """
        super().__init__()
        del kwargs
        self.latent_dim = latent_dim
        self.log_std_range = (math.log(min_std), math.log(max_std))
        common = dict(activation=activation, obs_normalization=obs_normalization)
        self.prior = MLPModel(obs, obs_groups, prior_obs_set, 2 * latent_dim, hidden_dims=prior_hidden_dims, **common)
        self.encoder = MLPModel(
            obs, obs_groups, encoder_obs_set, 2 * latent_dim, hidden_dims=encoder_hidden_dims, **common
        )
        self.decoder = MLPModel(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            other_input_dims=(latent_dim,),
            hidden_dims=decoder_hidden_dims,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": fixed_std, "learn_std": False},
            **common,
        )
        prefix = self.prior.obs_groups[: len(self.decoder.obs_groups)]
        if prefix != self.decoder.obs_groups:
            raise ValueError(
                f"CVAEModel: the decoder's observation groups {self.decoder.obs_groups} must be a prefix of the "
                f"prior's {self.prior.obs_groups}."
            )
        self.obs_groups = self.decoder.obs_groups

    def forward(
        self,
        obs: TensorDict,
        *args: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        use_encoder: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode an action from the prior mean, or from an encoder sample when ``use_encoder`` is set."""
        del args, masks, hidden_state, kwargs
        mu_p, _std_p = self.prior_params(obs)
        if use_encoder:
            mu_q, std_q = self.posterior_params(obs, mu_p)
            z = mu_q + std_q * torch.randn_like(mu_q)
        else:
            z = mu_p
        return self.decode(obs, z, stochastic_output=stochastic_output)

    def act_train(self, obs: TensorDict, prior_ratio: float = 0.0) -> torch.Tensor:
        """Sample a rollout action from an encoder draw, or from the prior mean for ``prior_ratio`` of the envs."""
        mu_p, _std_p = self.prior_params(obs)
        mu_q, std_q = self.posterior_params(obs, mu_p)
        z = mu_q + std_q * torch.randn_like(mu_q)
        if prior_ratio > 0.0:
            use_prior = torch.rand(z.shape[0], 1, device=z.device) < prior_ratio
            z = torch.where(use_prior, mu_p, z)
        return self.decode(obs, z, stochastic_output=True)

    def prior_params(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prior mean and std; shapes ``(N, latent_dim)``."""
        return self._split(self.prior(obs))

    def posterior_params(self, obs: TensorDict, mu_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the encoder mean (residual to the prior mean) and std; shapes ``(N, latent_dim)``."""
        d_mu, std_q = self._split(self.encoder(obs))
        return mu_p + d_mu, std_q

    def decode(self, obs: TensorDict, z: torch.Tensor, stochastic_output: bool = False) -> torch.Tensor:
        """Decode ``z``; the decoder's fixed-std Gaussian is sampled when ``stochastic_output``."""
        return self.decoder(obs, z, stochastic_output=stochastic_output)

    @staticmethod
    def kl_divergence(mu_q: torch.Tensor, std_q: torch.Tensor, mu_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
        """``KL(N(mu_q, std_q) || N(mu_p, std_p))`` summed over the latent; shape ``(N,)``."""
        var_ratio = (std_q / std_p) ** 2
        return 0.5 * (var_ratio + ((mu_q - mu_p) / std_p) ** 2 - 1.0 - torch.log(var_ratio)).sum(dim=-1)

    def _split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = out.split(self.latent_dim, dim=-1)
        return mu, torch.exp(log_std.clamp(*self.log_std_range))

    def update_normalization(self, obs: TensorDict) -> None:
        """Update every sub-network's observation normalizer."""
        for model in (self.prior, self.encoder, self.decoder):
            model.update_normalization(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """No recurrent state."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """No recurrent state."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """No recurrent state."""
        pass

    @property
    def output_std(self) -> torch.Tensor:
        """The decoder's fixed std."""
        return self.decoder.distribution.std_param  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCVAEModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export (prior-mean path only)."""
        return _OnnxCVAEModel(self, verbose)


class _TorchCVAEModel(nn.Module):
    """Exportable CVAE for JIT: ``forward`` decodes the prior mean, ``prior``/``decode`` expose the halves."""

    def __init__(self, model: CVAEModel) -> None:
        """Copy the prior and decoder with their normalizers; the encoder is training-only."""
        super().__init__()
        self.prior_normalizer = copy.deepcopy(model.prior.obs_normalizer)
        self.prior_mlp = copy.deepcopy(model.prior.mlp)
        self.decoder_normalizer = copy.deepcopy(model.decoder.obs_normalizer)
        self.decoder_mlp = copy.deepcopy(model.decoder.mlp)
        self.latent_dim = int(model.latent_dim)
        self.decoder_obs_dim = int(model.decoder.obs_dim)
        self.prior_obs_dim = int(model.prior.obs_dim)
        self.goal_dim = self.prior_obs_dim - self.decoder_obs_dim
        self.min_log_std = float(model.log_std_range[0])
        self.max_log_std = float(model.log_std_range[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the prior mean for the concatenated prior observation ``x``."""
        mu, _std = self.prior(x)
        return self.decode(x[:, : self.decoder_obs_dim], mu)

    @torch.jit.export
    def prior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prior ``(mean, std)`` for the concatenated prior observation ``x``."""
        out = self.prior_mlp(self.prior_normalizer(x))
        mu = out[:, : self.latent_dim]
        log_std = out[:, self.latent_dim :].clamp(self.min_log_std, self.max_log_std)
        return mu, torch.exp(log_std)

    @torch.jit.export
    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode the action mean for the decoder observation ``obs`` and latent ``z``."""
        return self.decoder_mlp(torch.cat([self.decoder_normalizer(obs), z], dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        """No state."""
        pass


class _OnnxCVAEModel(_TorchCVAEModel):
    """Exportable CVAE for ONNX (the prior-mean path)."""

    is_recurrent: bool = False

    def __init__(self, model: CVAEModel, verbose: bool) -> None:
        """Wrap the JIT module with the ONNX metadata."""
        super().__init__(model)
        self.verbose = verbose
        self.input_size = self.prior_obs_dim

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
