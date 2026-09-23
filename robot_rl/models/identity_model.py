# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.models.mlp_model import MLPModel


class IdentityModel(MLPModel):
    """A parameter-free model whose output is its observation set, concatenated.

    Serves as a distillation teacher when the target action is computed by the environment itself
    (e.g. a privileged observation term), so no checkpoint has to be loaded.
    """

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, **kwargs: Any
    ) -> None:
        """Resolve the observation set; extra model kwargs are accepted and ignored."""
        super().__init__(obs, obs_groups, obs_set, output_dim, memory_only=True)
        if self.obs_dim != output_dim:
            raise ValueError(
                f"IdentityModel: observation set '{obs_set}' is {self.obs_dim}-d, output is {output_dim}-d."
            )

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return the concatenated observation set."""
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)
