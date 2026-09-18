# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel
from robot_rl.modules import HiddenState
from robot_rl.modules.distribution import TangentStepVonMisesFisherDistribution


class TangentStepModel(MLPModel):
    """An MLP actor whose vMF output is a geodesic step from the latent it observes.

    ``base_obs_group`` names the observation group holding the current latent; it is handed to the distribution as
    the base direction before every distribution update, so a zero MLP output reproduces the current latent.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        base_obs_group: str = "latent",
        **kwargs: Any,
    ) -> None:
        """Build the model.

        Args:
            obs: Observation dictionary.
            obs_groups: Observation sets.
            obs_set: This model's observation set.
            output_dim: Ambient latent dimension.
            base_obs_group: Observation group carrying the current latent (the base direction).
            **kwargs: Forwarded to :class:`MLPModel`; ``distribution_cfg`` must name the tangent-step vMF.
        """
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if not isinstance(self.distribution, TangentStepVonMisesFisherDistribution):
            raise ValueError(
                "TangentStepModel needs a TangentStepVonMisesFisherDistribution as its output distribution."
            )
        if base_obs_group not in obs:
            raise ValueError(f"TangentStepModel: base observation group '{base_obs_group}' is not in the observations.")
        self.base_obs_group = base_obs_group

    def _set_base(self, obs: TensorDict) -> None:
        self.distribution.base = obs[self.base_obs_group]  # type: ignore[union-attr]

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
        """Forward pass; the observed latent is the base direction of the output distribution."""
        self._set_base(obs)
        return super().forward(
            obs, *args, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output, std_clip=std_clip
        )

    def act_and_log_prob(
        self, obs: TensorDict, *args: torch.Tensor, std_clip: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample and log-prob with the observed latent as the base direction."""
        self._set_base(obs)
        return super().act_and_log_prob(obs, *args, std_clip=std_clip)
