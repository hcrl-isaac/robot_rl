"""A deterministic, differentiable world model: dynamics plus reward, termination and observation heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence

from robot_rl.modules import EmpiricalNormalization
from robot_rl.utils import resolve_nn_activation


def _mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int], activation: str, layer_norm: bool) -> nn.Module:
    """Return an MLP with an optional LayerNorm before each hidden activation (``modules.MLP`` norms only the first)."""
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(dim, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(resolve_nn_activation(activation))
        dim = hidden
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """Predicts the next state, reward, termination and noise-free observation of a simulator step.

    Everything runs in normalized coordinates: the dynamics predicts a residual on the normalized state, so an
    imagined rollout chains normalized states and only the critic and the actor's observations are de-normalized.
    The reward and termination heads read both ends of the step, ``(s_t, a_t, s_{t+1}, g_t, xi)``, so they fit
    whichever end the env evaluates them on; ``g_{t+1}`` is left out because after a reset it is the next episode's.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int = 0,
        goal_dim: int = 0,
        obs_dim: int = 0,
        hidden_dims: Sequence[int] = (2048, 2048, 2048),
        head_hidden_dims: Sequence[int] = (512, 512),
        activation: str = "swish",
        layer_norm: bool = True,
        condition_on_goal: bool = False,
        reward_head: bool = True,
        termination_head: bool = True,
    ) -> None:
        """Build the networks and the input normalizers.

        Args:
            state_dim: Width of the predicted state.
            action_dim: Width of the action.
            context_dim: Width of the per-episode simulation context (0 = none).
            goal_dim: Width of the exogenous goal (0 = none).
            obs_dim: Width of the noise-free observation the observation head predicts (0 = no head).
            hidden_dims: Hidden widths of the dynamics network.
            head_hidden_dims: Hidden widths of the reward, termination and observation heads.
            activation: Hidden activation.
            layer_norm: Apply LayerNorm before each hidden activation.
            condition_on_goal: Feed the goal to the dynamics and observation heads (needed when the state or the
                observation holds robot-to-goal errors).
            reward_head: Learn a reward head.
            termination_head: Learn a termination head.
        """
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.context_dim, self.goal_dim, self.obs_dim = context_dim, goal_dim, obs_dim
        self.condition_on_goal = condition_on_goal and goal_dim > 0
        dyn_goal = goal_dim if self.condition_on_goal else 0

        self.state_normalizer = EmpiricalNormalization(state_dim)
        self.context_normalizer = EmpiricalNormalization(context_dim) if context_dim else nn.Identity()
        self.goal_normalizer = EmpiricalNormalization(goal_dim) if goal_dim else nn.Identity()
        self.obs_normalizer = EmpiricalNormalization(obs_dim) if obs_dim else nn.Identity()

        self.dynamics = _mlp(
            state_dim + action_dim + context_dim + dyn_goal, state_dim, hidden_dims, activation, layer_norm
        )
        step_dim = 2 * state_dim + action_dim + context_dim + goal_dim
        self.reward = _mlp(step_dim, 1, head_hidden_dims, activation, layer_norm) if reward_head else None
        self.termination = _mlp(step_dim, 1, head_hidden_dims, activation, layer_norm) if termination_head else None
        self.observation = (
            _mlp(state_dim + context_dim + dyn_goal, obs_dim, head_hidden_dims, activation, layer_norm)
            if obs_dim
            else None
        )
        # a zero last layer starts the dynamics as the identity, the stable fixed point for a long rollout
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)

    @torch.no_grad()
    def update_normalization(
        self, state: torch.Tensor, context: torch.Tensor | None, goal: torch.Tensor | None, obs: torch.Tensor | None
    ) -> None:
        """Update the input normalizers from a batch of logged rows."""
        self.state_normalizer.update(state)
        if context is not None and self.context_dim:
            self.context_normalizer.update(context)
        if goal is not None and self.goal_dim:
            self.goal_normalizer.update(goal)
        if obs is not None and self.obs_dim:
            self.obs_normalizer.update(obs)

    def normalize(
        self, state: torch.Tensor, context: torch.Tensor | None, goal: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize raw state, context and goal (absent inputs become zero-width tensors)."""
        empty = state.new_zeros(*state.shape[:-1], 0)
        ctx = self.context_normalizer(context) if self.context_dim else empty
        g = self.goal_normalizer(goal) if self.goal_dim and goal is not None else empty
        return self.state_normalizer(state), ctx, g

    def denormalize_state(self, state_n: torch.Tensor) -> torch.Tensor:
        """Map a normalized state back to raw units."""
        return self.state_normalizer.inverse(state_n)

    def step(
        self,
        state_n: torch.Tensor,
        action: torch.Tensor,
        context_n: torch.Tensor,
        goal_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One imagined step in normalized coordinates.

        Args:
            state_n: Normalized state, ``(..., state_dim)``.
            action: Action, ``(..., action_dim)``.
            context_n: Normalized context (zero-width when absent).
            goal_n: Normalized goal at ``t`` (zero-width when absent).

        Returns:
            The next normalized state, the reward (``None`` without a head) and the termination logit (``None``
            without a head), the last two of shape ``(...)``.
        """
        dyn_in = [state_n, action, context_n] + ([goal_n] if self.condition_on_goal else [])
        next_state_n = state_n + self.dynamics(torch.cat(dyn_in, dim=-1))
        step_in = torch.cat([state_n, next_state_n, action, context_n, goal_n], dim=-1)
        reward = self.reward(step_in).squeeze(-1) if self.reward is not None else None
        term_logit = self.termination(step_in).squeeze(-1) if self.termination is not None else None
        return next_state_n, reward, term_logit

    def observe(self, state_n: torch.Tensor, context_n: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        """Predict the raw (de-normalized) noise-free observation of a normalized state."""
        return self.obs_normalizer.inverse(self.observe_normalized(state_n, context_n, goal_n))  # type: ignore[union-attr]

    def observe_normalized(self, state_n: torch.Tensor, context_n: torch.Tensor, goal_n: torch.Tensor) -> torch.Tensor:
        """Predict the normalized noise-free observation (the fit target space)."""
        obs_in = [state_n, context_n] + ([goal_n] if self.condition_on_goal else [])
        return self.observation(torch.cat(obs_in, dim=-1))  # type: ignore[misc]
