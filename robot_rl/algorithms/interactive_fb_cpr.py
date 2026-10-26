# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FB-CPR over a coupled robot-object state, with a local-goal actor objective (I-BFM)."""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.storage import ReplayBuffer, TrajectoryBuffer
from robot_rl.utils import eval_mode

from .fb_cpr import FbCpr


def sphere_log_map(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Tangent vector at ``base`` pointing along the geodesic to ``target``, both on the same sphere.

    Args:
        base: Points on the sphere of radius ``r``, shape ``(..., d)``.
        target: Points on the same sphere, shape ``(..., d)``.
        eps: Floor on the tangent component's norm.

    Returns:
        Vectors of norm ``r * angle(base, target)``, orthogonal to ``base``, shape ``(..., d)``.
    """
    radius = base.norm(dim=-1, keepdim=True)
    x = base / radius
    y = target / target.norm(dim=-1, keepdim=True)
    cos = (x * y).sum(dim=-1, keepdim=True)
    v = y - cos * x
    v_norm = v.norm(dim=-1, keepdim=True)
    theta = torch.atan2(v_norm, cos)
    return radius * theta * v / v_norm.clamp(min=eps)


class InteractiveFbCpr(FbCpr):
    """FB-CPR for robot-object interaction: an Interactive Behavioral Foundation Model.

    The backward map and critics read the robot, object and contact state; motions without an object get zero object
    groups. The actor objective adds a residual over episode-aligned replay clips: actions sampled under the log map
    from the current state's embedding to the next state's (local) and to the clip end's (goal), each scored by the
    critics for that target latent.

    Reference:
        - Han et al. "I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning."
          arXiv preprint arXiv:2610.06129 (2026).
    """

    replay_keep_terminal = True

    def __init__(
        self,
        *args: Any,
        logo_coef: float = 0.005,
        logo_horizon: int = 8,
        logo_local_weight: float = 4.0 / 7.0,
        expert_absent_groups: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FB-CPR and the local-goal objective.

        Args:
            *args: Positional arguments of :class:`FbCpr`.
            logo_coef: Weight of the local-goal residual on the actor loss.
            logo_horizon: Steps from a clip's anchor state to its goal state.
            logo_local_weight: Share of the residual on the local target; the goal target takes the rest.
            expert_absent_groups: Observation groups the expert motions lack, filled with zeros.
            **kwargs: Keyword arguments of :class:`FbCpr`.
        """
        super().__init__(*args, **kwargs)
        self.logo_coef = logo_coef
        self.logo_horizon = logo_horizon
        self.logo_local_weight = logo_local_weight
        compile_mode = kwargs.get("compile_mode", "reduce-overhead")
        self._logo_actor_step = self._logo_actor_loss
        if compile_mode is not None:
            self._logo_actor_step = torch.compile(self._logo_actor_loss, mode=compile_mode)
        # the base compiles _update_actor; the clip is sampled and normalized outside the compiled graph
        self._update_actor = self._update_actor_with_clip
        self._clip: tuple[TensorDict, TensorDict, TensorDict] | None = None

    @staticmethod
    def prepare_expert_motions(
        expert_buffer: TrajectoryBuffer, obs: TensorDict, obs_groups: dict[str, list[str]], cfg: dict
    ) -> None:
        """Zero-fill the groups the motions lack and drop the stored groups no expert reader uses.

        Args:
            expert_buffer: The loaded expert motions.
            obs: A representative env observation.
            obs_groups: The resolved observation sets.
            cfg: The algorithm cfg, naming ``expert_absent_groups``.
        """
        motions = expert_buffer.motions
        read = {g for s in ("backward", "discriminator", "expert") for g in obs_groups[s]}
        for group in cfg.get("expert_absent_groups") or []:
            if group in motions:
                raise ValueError(f"expert_absent_groups names '{group}', which the motion file already holds")
            width = obs[group].shape[1:]
            motions[group] = torch.zeros(*motions.shape, *width, dtype=obs[group].dtype, device=motions.device)
        for key in list(motions.keys()):
            if key not in read:
                del motions[key]

    def _valid_rows(self, horizon: int, size: int) -> torch.Tensor:
        """Flat replay indices of ``size`` windows of ``horizon`` steps that stay inside one episode.

        Args:
            horizon: Steps per window.
            size: Number of windows.

        Returns:
            Indices ``(size, horizon)``; row ``i`` holds one env's consecutive steps.
        """
        rb = self.replay_buffer
        num_envs, cap_rows = rb.num_envs, rb.capacity_per_env
        filled_rows = cap_rows if rb._is_full else rb._curr_idx // num_envs
        write_row = rb._curr_idx // num_envs
        rows = torch.arange(filled_rows, device=rb.device)
        # a window must not cross the circular write head
        last = rows + horizon - 1
        if rb._is_full:
            safe = torch.where(rows < write_row, last < write_row, last < cap_rows + write_row)
        else:
            safe = last < filled_rows
        valid_rows = rows[safe]
        if valid_rows.numel() == 0:
            raise RuntimeError(f"the replay buffer holds no {horizon}-step window yet")
        # oversample, then keep windows whose every step is mid-episode (its next observation is not a reset)
        draws = 2 * size
        start = valid_rows[torch.randint(valid_rows.numel(), (draws,), device=rb.device)]
        envs = torch.randint(num_envs, (draws,), device=rb.device)
        steps = ((start[:, None] + torch.arange(horizon, device=rb.device)) % cap_rows) * num_envs + envs[:, None]
        crossed = rb.dones[steps].sum(dim=(1, 2)) > 0
        return steps[torch.argsort(crossed.float(), stable=True)[:size]]

    def _sample_replay(self) -> ReplayBuffer.Batch:
        """Draw a mini-batch of transitions whose next observation is not a post-reset one."""
        rb = self.replay_buffer
        idx = self._valid_rows(1, rb._indices.shape[0])[:, 0]
        return ReplayBuffer.Batch(
            rb._read(rb.observations, idx, self.device),
            rb._read(rb.next_observations, idx, self.device),
            rb.actions[idx].to(self.device),
            rb.rewards[idx].to(self.device),
            rb.gammas[idx].to(self.device),
            rb.context[idx].to(self.device),
            rb.next_terminated[idx].to(self.device),
        )

    def _sample_clip(self) -> tuple[TensorDict, TensorDict, TensorDict]:
        """Draw the anchor, next and goal observations of episode-aligned replay clips."""
        rb = self.replay_buffer
        steps = self._valid_rows(self.logo_horizon, self.batch_size)
        return (
            rb._read(rb.observations, steps[:, 0], self.device),
            rb._read(rb.next_observations, steps[:, 0], self.device),
            rb._read(rb.next_observations, steps[:, -1], self.device),
        )

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Sample the local-goal clips, then run the FB-CPR update."""
        self._clip = self._sample_clip()
        return super().update()

    def _update_actor_with_clip(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict]:
        """Actor step on the FB-CPR objective plus the local-goal residual over the sampled clips."""
        with torch.no_grad(), eval_mode(self.obs_normalizer):
            anchor, local, goal = (self.obs_normalizer(td) for td in self._clip)
        loss_dict, extras = self._logo_actor_step(batch, anchor, local, goal)
        return loss_dict, extras

    def _logo_actor_loss(
        self, batch: ReplayBuffer.Batch, anchor: TensorDict, local: TensorDict, goal: TensorDict
    ) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            base_loss, parts = self._actor_objective(batch.observations, batch.context, batch.context)
            with torch.no_grad():
                z_anchor = self.project_z(self.backward_map(anchor))
                z_local = self.project_z(self.backward_map(local))
                z_goal = self.project_z(self.backward_map(goal))
            local_loss, _ = self._actor_objective(anchor, sphere_log_map(z_anchor, z_local), z_local)
            goal_loss, _ = self._actor_objective(anchor, sphere_log_map(z_anchor, z_goal), z_goal)
            residual = self.logo_local_weight * local_loss + (1.0 - self.logo_local_weight) * goal_loss
            actor_loss = base_loss + self.logo_coef * residual
        self._step_actor(actor_loss)
        loss_dict, extras = self._actor_logs(actor_loss, parts)
        with torch.no_grad():
            loss_dict["Actor_Loss/logo_local_loss"] = local_loss.detach()
            loss_dict["Actor_Loss/logo_goal_loss"] = goal_loss.detach()
            extras["logo_goal_angle"] = (sphere_log_map(z_anchor, z_goal).norm(dim=-1) / math.sqrt(self.z_dim)).mean()
        return loss_dict, extras
