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
        group_dims: dict[str, int] | None = None,
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
            num_states: Dimension of the style state (every style group any expert reads, concatenated).
            obs_groups: Observation groups dictionary; ``obs_groups["style"]`` names the groups read by default.
            experts: One dict per expert with ``name``, ``data_path`` (a ``.pt`` file holding a ``states``
                tensor of shape ``(M, its groups' width)``), ``weight`` (advantage weight of its reward stream),
                optionally ``obs_groups`` (the style groups it reads, in its data's order; default
                ``obs_groups["style"]``) and ``gate_group`` (observation group whose flag limits where it pays).
            group_dims: Width of each style group; required when an expert names its own ``obs_groups``.
            hidden_dims: Hidden dimensions of every discriminator MLP.
            activation: Activation function.
            state_normalization: Normalize the style state with running statistics from the policy rollouts.
            reward_normalization: Standardize each discriminator's reward with its own running statistics.
            gradient_penalty_coef: Coefficient of the gradient penalty on expert/policy interpolates.
            learning_rate: Discriminator learning rate.
            num_learning_epochs: Passes over the rollout's policy states per update.
            batch_size: Mini-batch size for the discriminator update.
            weight: Global multiplier of every style stream's advantage weight.
            weight_schedule: Optional schedule of ``weight`` over collection steps, as in the RND extension.
            gate_threshold: When set, a transition gets zero style reward unless at least one normalized
                discriminator score exceeds it (interface states no expert claims are not penalized).
            reward_clip: Symmetric clip on the normalized style rewards.
            load_experts: Load the expert datasets (False for inference-only use; ``update`` then fails).
            device: Device.
        """
        super().__init__()
        self.num_states = num_states
        self.obs_groups = obs_groups
        expert_groups = [list(e.get("obs_groups") or obs_groups["style"]) for e in experts]
        self.state_groups = list(dict.fromkeys([*obs_groups["style"], *(g for gs in expert_groups for g in gs)]))
        if group_dims is None:
            if self.state_groups != list(obs_groups["style"]):
                raise ValueError("Experts that name their own obs_groups need group_dims.")
            self.expert_columns = [torch.arange(num_states, device=device) for _ in experts]
        else:
            offsets, start = {}, 0
            for g in self.state_groups:
                offsets[g], start = start, start + group_dims[g]
            if start != num_states:
                raise ValueError(f"Style groups {self.state_groups} span {start} columns, not {num_states}.")
            self.expert_columns = [
                torch.cat([torch.arange(offsets[g], offsets[g] + group_dims[g]) for g in gs]).to(device)
                for gs in expert_groups
            ]
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
        self.reward_steps = 0

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
            MLP(len(cols), 1, hidden_dims, activation=activation).to(device) for cols in self.expert_columns
        ])
        self.expert_states = (
            [self._load_expert(e["data_path"], len(cols)) for e, cols in zip(experts, self.expert_columns, strict=True)]
            if load_experts
            else []
        )
        self.optimizer = torch.optim.Adam(self.discriminators.parameters(), lr=learning_rate)

    @property
    def num_experts(self) -> int:
        """Number of expert datasets (reward streams)."""
        return len(self.discriminators)

    def _load_expert(self, path: str, width: int) -> torch.Tensor:
        path = os.path.expanduser(path)
        data = torch.load(path, map_location=self.device, weights_only=False)
        states = data["states"] if isinstance(data, dict) else data
        if states.ndim != 2 or states.shape[1] != width:
            raise ValueError(
                f"Expert data {path} has shape {tuple(states.shape)}; expected (M, {width}) to match its style"
                " observation groups."
            )
        return states.to(self.device, dtype=torch.float32)

    def get_style_state(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate every style group any expert reads."""
        return torch.cat([obs[g] for g in self.state_groups], dim=-1)

    def _normalize_expert(self, states: torch.Tensor, k: int) -> torch.Tensor:
        """Normalize expert ``k``'s data with the state normalizer's statistics for its columns."""
        if not self.state_normalization:
            return states
        cols, norm = self.expert_columns[k], self.state_normalizer
        return (states - norm._mean[:, cols]) / (norm._std[:, cols] + norm.eps)  # type: ignore[index, operator]

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the state normalizer from policy observations."""
        if self.state_normalization:
            self.state_normalizer.update(self.get_style_state(obs))  # type: ignore[operator]

    def gate_masks(self, obs: TensorDict) -> list[torch.Tensor | None]:
        """Per-expert row mask of the transitions its stream pays on; ``None`` for an ungated expert."""
        return [None if group is None else obs[group].reshape(-1) > 0.5 for group in self.gate_groups]

    def compute_rewards(self, obs: TensorDict) -> torch.Tensor:
        """Per-expert style rewards for the current transitions, shape ``(num_envs, num_experts)``.

        A reward is a standardized discriminator score, so it is zero-mean over the states a stream pays on and
        is negative on about half of them; the task stream carries the positive per-step floor.
        """
        self.reward_steps += 1
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.reward_steps, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        with torch.no_grad():
            state = self.state_normalizer(self.get_style_state(obs))
            pairs = zip(self.discriminators, self.expert_columns, strict=True)
            scores = torch.cat([disc(state[:, cols]) for disc, cols in pairs], dim=-1)  # (E, K)
            gates = self.gate_masks(obs)
            columns = []
            for k, norm in enumerate(self.reward_normalizers):
                column = scores[:, k : k + 1]
                if self.reward_normalization:
                    # only the rows the stream pays on: normalizing over the whole batch leaves the in-gate
                    # reward a non-zero mean the gated critic reads as a regime signal
                    rows = column if gates[k] is None else column[gates[k]]
                    if rows.numel():
                        norm.update(rows)  # type: ignore[operator]
                columns.append(norm(column))
            rewards = torch.cat(columns, dim=-1).clamp(-self.reward_clip, self.reward_clip)
            for k, gate in enumerate(gates):
                # an expert that describes a transient regime (e.g. free flight) must not pay outside it
                if gate is not None:
                    rewards[:, k : k + 1] = rewards[:, k : k + 1] * gate.reshape(-1, 1).float()
            if self.gate_threshold is not None:
                # no discriminator claims the transition: it is an interface state, not a style violation
                rewards = rewards * (rewards.max(dim=-1, keepdim=True).values > self.gate_threshold).float()
        return rewards

    def update(self, policy_states: torch.Tensor, gates: list[torch.Tensor | None] | None = None) -> dict[str, float]:
        """Train every discriminator against the rollout's policy states with WGAN-GP; returns mean losses.

        Args:
            policy_states: ``(N, style_dim)`` style states of the rollout.
            gates: Per-expert row mask from :meth:`gate_masks`; a gated expert trains only on its own rows,
                since states it never pays on are not states it should score.
        """
        losses: dict[str, float] = {}
        with torch.no_grad():
            policy_states = self.state_normalizer(policy_states)
        for k, (disc, expert) in enumerate(zip(self.discriminators, self.expert_states, strict=True)):
            gate = None if gates is None else gates[k]
            states = policy_states[:, self.expert_columns[k]]
            states = states if gate is None else states[gate]
            num = states.shape[0]
            if num == 0:
                continue
            batch = min(self.batch_size, num)
            num_batches = max(1, num // batch)
            sums = {"expert": 0.0, "policy": 0.0, "grad_penalty": 0.0}
            count = 0
            for _ in range(self.num_learning_epochs):
                perm = torch.randperm(num, device=self.device)
                for b in range(num_batches):
                    pol = states[perm[b * batch : (b + 1) * batch]]
                    with torch.no_grad():
                        exp = self._normalize_expert(
                            expert[torch.randint(0, expert.shape[0], (pol.shape[0],), device=self.device)], k
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
                losses[f"Style/{self.names[k]}_{key}"] = value / max(count, 1)
        return losses

    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """Return the module state plus the schedule clock, so a resume continues the weight ramp."""
        state = super().state_dict(*args, **kwargs)
        state["reward_steps"] = self.reward_steps
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> Any:
        """Restore the module state and the schedule clock."""
        state_dict = dict(state_dict)
        # "update_counter" is the key checkpoints written before the rename carry
        self.reward_steps = int(state_dict.pop("reward_steps", state_dict.pop("update_counter", 0)))
        return super().load_state_dict(state_dict, strict=strict)

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
    """Fill in the style-state dimension, group widths and observation groups, or set ``style_cfg`` to None."""
    if "style_cfg" in alg_cfg and alg_cfg["style_cfg"] is not None:
        experts = alg_cfg["style_cfg"].get("experts", [])
        groups = dict.fromkeys([*obs_groups["style"], *(g for e in experts for g in (e.get("obs_groups") or []))])
        group_dims = {}
        for obs_group in groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"Style discriminators only support 1D observations, got {obs[obs_group].shape} for '{obs_group}'."
                )
            group_dims[obs_group] = obs[obs_group].shape[-1]
        alg_cfg["style_cfg"]["num_states"] = sum(group_dims.values())
        alg_cfg["style_cfg"]["group_dims"] = group_dims
        alg_cfg["style_cfg"]["obs_groups"] = obs_groups
    else:
        alg_cfg["style_cfg"] = None
    return alg_cfg
