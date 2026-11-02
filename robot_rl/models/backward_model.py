# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel


class BackwardModel(MLPModel):
    """A frozen exported backward map ``B`` as a distillation teacher: ``normalize(mean_k B(f_k))``.

    The observation set is ``horizon`` reference feature vectors concatenated in order; the horizon is
    inferred from its width and ``B``'s input width, so it follows whatever the environment provides.
    """

    loads_own_weights = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        backward_path: str,
        **kwargs: Any,
    ) -> None:
        """Load ``B`` and resolve the horizon; extra model kwargs are accepted and ignored.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set holding the stacked reference features.
            output_dim: Latent dimension, which must match ``B``'s output.
            backward_path: TorchScript export of ``B``.
            **kwargs: Shared model options (e.g. ``hidden_dims``) that do not apply here.
        """
        super().__init__(obs, obs_groups, obs_set, output_dim, memory_only=True)
        self.backward = torch.jit.load(backward_path, map_location="cpu").eval()
        for param in self.backward.parameters():
            param.requires_grad_(False)
        self.feature_dim = next(p for p in self.backward.parameters() if p.dim() == 2).shape[1]
        if self.obs_dim % self.feature_dim:
            raise ValueError(
                f"BackwardModel: observation set '{obs_set}' is {self.obs_dim}-d, not a multiple of B's "
                f"{self.feature_dim}-d input."
            )
        self.horizon = self.obs_dim // self.feature_dim
        with torch.no_grad():
            latent_dim = self.backward(torch.zeros(1, self.feature_dim)).shape[-1]
        if latent_dim != output_dim:
            raise ValueError(f"BackwardModel: B outputs {latent_dim}-d latents, output is {output_dim}-d.")

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return the normalized mean of ``B`` over the stacked reference features."""
        features = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        latents = self.backward(features.reshape(-1, self.feature_dim)).reshape(features.shape[0], self.horizon, -1)
        return torch.nn.functional.normalize(latents.mean(dim=1), dim=-1)

    def train(self, mode: bool = True) -> BackwardModel:
        """Keep ``B`` in eval mode whatever the teacher's mode, since its normalizer must not update."""
        super().train(mode)
        self.backward.eval()
        return self
