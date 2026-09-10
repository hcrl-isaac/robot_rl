# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-expert Wasserstein style discriminators: experts enter PPO as data, each with its own reward stream."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, NoReturn

from robot_rl.modules import MLP, EmpiricalNormalization


class StyleDiscriminators(nn.Module):
    """One Wasserstein discriminator per expert dataset, each producing a style reward stream.

    References:
        - Li et al. "Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations." CoRL (2022).
        - Xu et al. "Composite Motion Learning with Task Control." SIGGRAPH (2023) -- one value head per reward.
    """

    def __init__(
        self,
        num_states: int,
        obs_groups: dict[str, list[str]],
        experts: list[dict[str, Any]],
        hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        state_normalization: bool = True,
        reward_normalization: bool = True,
        gradient_penalty_coef: float = 10.0,
        learning_rate: float = 1e-4,
        num_learning_epochs: int = 2,
        batch_size: int = 1024,
        weight: float = 1.0,
        weight_schedule: dict | None = None,
        gate_threshold: float | None = None,
        reward_clip: float = 5.0,
        load_experts: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize the discriminators and load the expert datasets.

        Args:
            num_states: Dimension of the style state (the concatenated ``style`` observation groups).
            obs_groups: Observation groups dictionary; ``obs_groups["style"]`` names the groups read.
            experts: One dict per expert with ``name``, ``data_path`` (a ``.pt`` file holding a ``states``
                tensor of shape ``(M, num_states)``) and ``weight`` (advantage weight of its reward stream).
            hidden_dims: Hidden dimensions of every discriminator MLP.
            activation: Activation function.
            state_normalization: Normalize the style state with running statistics from the policy rollouts.
            reward_normalization: Standardize each discriminator's reward with its own running statistics.
            gradient_penalty_coef: Coefficient of the gradient penalty on expert/policy interpolates.
            learning_rate: Discriminator learning rate.
            num_learning_epochs: Passes over the rollout's policy states per update.
            batch_size: Mini-batch size for the discriminator update.
            weight: Global multiplier of every style stream's advantage weight.
            weight_schedule: Optional schedule of ``weight`` over env steps, as in the RND extension.
            gate_threshold: When set, a transition gets zero style reward unless at least one normalized
                discriminator score exceeds it (interface states no expert claims are not penalized).
            reward_clip: Symmetric clip on the normalized style rewards.
            load_experts: Load the expert datasets (False for inference-only use; ``update`` then fails).
            device: Device.
        """
        super().__init__()
        self.num_states = num_states
        self.obs_groups = obs_groups
        self.device = device
        self.names = [str(e["name"]) for e in experts]
        self.stream_weights = torch.tensor([float(e.get("weight", 1.0)) for e in experts], device=device)
        self.gate_groups = [e.get("gate_group") for e in experts]
        self.initial_weight = weight
        self.weight = weight
        self.gradient_penalty_coef = gradient_penalty_coef
        self.num_learning_epochs = num_learning_epochs
        self.batch_size = batch_size
        self.gate_threshold = gate_threshold
        self.reward_clip = reward_clip
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization
        self.update_counter = 0

        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None

        self.state_normalizer = (
            EmpiricalNormalization(shape=[num_states], until=int(1.0e8)).to(device)
            if state_normalization
            else nn.Identity()
        )
        self.reward_normalizers = nn.ModuleList([
            EmpiricalNormalization(shape=[1], until=int(1.0e8)).to(device) if reward_normalization else nn.Identity()
            for _ in experts
        ])
        self.discriminators = nn.ModuleList([
            MLP(num_states, 1, hidden_dims, activation=activation).to(device) for _ in experts
        ])
        self.expert_states = [self._load_expert(e["data_path"]) for e in experts] if load_experts else []
        self.optimizer = torch.optim.Adam(self.discriminators.parameters(), lr=learning_rate)

    @property
    def num_experts(self) -> int:
        """Number of expert datasets (reward streams)."""
        return len(self.discriminators)

    def _load_expert(self, path: str) -> torch.Tensor:
        path = os.path.expanduser(path)
        data = torch.load(path, map_location=self.device, weights_only=False)
        states = data["states"] if isinstance(data, dict) else data
        if states.ndim != 2 or states.shape[1] != self.num_states:
            raise ValueError(
                f"Expert data {path} has shape {tuple(states.shape)}; expected (M, {self.num_states}) to match the"
                " style observation groups."
            )
        return states.to(self.device, dtype=torch.float32)

    def get_style_state(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the observation groups that make up the style state."""
        return torch.cat([obs[g] for g in self.obs_groups["style"]], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the state normalizer from policy observations."""
        if self.state_normalization:
            self.state_normalizer.update(self.get_style_state(obs))  # type: ignore[operator]

    def compute_rewards(self, obs: TensorDict) -> torch.Tensor:
        """Per-expert style rewards for the current transitions, shape ``(num_envs, num_experts)``."""
        self.update_counter += 1
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        with torch.no_grad():
            state = self.state_normalizer(self.get_style_state(obs))
            scores = torch.cat([disc(state) for disc in self.discriminators], dim=-1)  # (E, K)
            columns = []
            for k, norm in enumerate(self.reward_normalizers):
                if self.reward_normalization:
                    norm.update(scores[:, k : k + 1])  # type: ignore[operator]
                columns.append(norm(scores[:, k : k + 1]))
            rewards = torch.cat(columns, dim=-1).clamp(-self.reward_clip, self.reward_clip)
            for k, group in enumerate(self.gate_groups):
                # an expert that describes a transient regime (e.g. free flight) must not pay outside it
                if group is not None:
                    rewards[:, k : k + 1] = rewards[:, k : k + 1] * obs[group].reshape(-1, 1)
            if self.gate_threshold is not None:
                # no discriminator claims the transition: it is an interface state, not a style violation
                rewards = rewards * (rewards.max(dim=-1, keepdim=True).values > self.gate_threshold).float()
        return rewards

    def update(self, policy_states: torch.Tensor) -> dict[str, float]:
        """Train every discriminator against the rollout's policy states with WGAN-GP; returns mean losses."""
        losses: dict[str, float] = {}
        num = policy_states.shape[0]
        batch = min(self.batch_size, num)
        num_batches = max(1, num // batch)
        with torch.no_grad():
            policy_states = self.state_normalizer(policy_states)
        for k, (disc, expert) in enumerate(zip(self.discriminators, self.expert_states, strict=True)):
            sums = {"expert": 0.0, "policy": 0.0, "grad_penalty": 0.0}
            count = 0
            for _ in range(self.num_learning_epochs):
                perm = torch.randperm(num, device=self.device)
                for b in range(num_batches):
                    pol = policy_states[perm[b * batch : (b + 1) * batch]]
                    with torch.no_grad():
                        exp = self.state_normalizer(
                            expert[torch.randint(0, expert.shape[0], (pol.shape[0],), device=self.device)]
                        )
                    expert_score = disc(exp).mean()
                    policy_score = disc(pol).mean()
                    penalty = self._gradient_penalty(disc, exp, pol)
                    loss = policy_score - expert_score + self.gradient_penalty_coef * penalty
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    sums["expert"] += expert_score.item()
                    sums["policy"] += policy_score.item()
                    sums["grad_penalty"] += penalty.item()
                    count += 1
            for key, value in sums.items():
                losses[f"style/{self.names[k]}_{key}"] = value / max(count, 1)
        return losses

    @staticmethod
    def _gradient_penalty(disc: nn.Module, expert: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(expert.shape[0], 1, device=expert.device)
        mixed = (alpha * expert + (1.0 - alpha) * policy).requires_grad_(True)
        grad = torch.autograd.grad(disc(mixed).sum(), mixed, create_graph=True)[0]
        return (grad.norm(dim=-1) - 1.0).pow(2).mean()

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Disallow generic forward calls for this module."""
        raise RuntimeError("Forward is not implemented. Use compute_rewards / update instead.")

    def train(self, mode: bool = True) -> StyleDiscriminators:
        """Set training mode for the discriminators and the normalizers."""
        self.discriminators.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizers.train(mode)
        return self

    def eval(self) -> StyleDiscriminators:
        """Set the module to evaluation mode."""
        return self.train(False)

    def _constant_weight_schedule(self, step: int, **kwargs: Any) -> float:
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs: Any) -> float:
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(
        self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs: Any
    ) -> float:
        if step < initial_step:
            return self.initial_weight
        if step > final_step:
            return final_value
        frac = (step - initial_step) / (final_step - initial_step)
        return self.initial_weight + (final_value - self.initial_weight) * frac


def resolve_style_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]]) -> dict:
    """Fill in the style-state dimension and observation groups, or set ``style_cfg`` to None."""
    if "style_cfg" in alg_cfg and alg_cfg["style_cfg"] is not None:
        num_states = 0
        for obs_group in obs_groups["style"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"Style discriminators only support 1D observations, got {obs[obs_group].shape} for '{obs_group}'."
                )
            num_states += obs[obs_group].shape[-1]
        alg_cfg["style_cfg"]["num_states"] = num_states
        alg_cfg["style_cfg"]["obs_groups"] = obs_groups
    else:
        alg_cfg["style_cfg"] = None
    return alg_cfg
