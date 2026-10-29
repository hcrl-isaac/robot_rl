"""Time-major replay of contiguous per-env trajectories, sampled as fixed-length windows."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from tensordict import TensorDict


class SequenceReplayBuffer:
    """FIFO replay that stores each step once, as rows ``(capacity, num_envs, ...)``, and samples windows.

    Row ``t`` of env ``e`` holds the observation before the step and that step's action, reward and flags. A per-env
    episode-step counter makes window validity a single check: a window lies inside one episode when the counter
    rises by exactly one along it.
    """

    @dataclass
    class Window:
        """A batch of windows of ``L`` transitions, time-major."""

        obs: TensorDict
        """Observation rows ``t .. t+L``, shape ``(L + 1, B, ...)``; row ``L`` is valid only where ``next_valid``."""

        actions: torch.Tensor
        """Executed actions, ``(L, B, A)``."""

        rewards: torch.Tensor
        """Rewards, ``(L, B)``."""

        terminated: torch.Tensor
        """Termination (not time-out) after each step, ``(L, B)``; only the last step can be set."""

        next_valid: torch.Tensor
        """Whether row ``L`` continues the same episode, ``(B,)``."""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        obs: TensorDict,
        groups: list[str],
        num_actions: int,
        device: str = "cpu",
        obs_dtypes: dict[str, str] | None = None,
    ) -> None:
        """Allocate the storage.

        Args:
            num_envs: Number of parallel environments.
            capacity_per_env: Steps kept per environment.
            obs: An observation sample, for the group widths.
            groups: The observation groups to store (only these; nothing else is kept).
            num_actions: Action width.
            device: Storage device.
            obs_dtypes: Per-group storage dtype names (e.g. ``{"goal": "float16"}``); sampled back as float32.
        """
        self.num_envs = num_envs
        self.capacity = capacity_per_env
        self.device = device
        self.groups = list(dict.fromkeys(groups))
        obs_dtypes = obs_dtypes or {}
        shape = (capacity_per_env, num_envs)
        self.obs = {
            g: torch.zeros(*shape, obs[g].shape[-1], dtype=getattr(torch, obs_dtypes.get(g, "float32")), device=device)
            for g in self.groups
        }
        self.actions = torch.zeros(*shape, num_actions, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self.episode_step = torch.zeros(shape, dtype=torch.int32, device=device)
        self._current_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Append one step of every env.

        Args:
            obs: Observations before the step.
            actions: Executed actions.
            rewards: Rewards of the step.
            terminated: Termination flags (dones that are not time-outs).
            dones: All episode ends (terminations and time-outs); the next row starts a new episode.
        """
        row = self.pos
        for g in self.groups:
            self.obs[g][row].copy_(obs[g])
        self.actions[row].copy_(actions)
        self.rewards[row].copy_(rewards.view(-1))
        self.terminated[row].copy_(terminated.view(-1).bool())
        self.episode_step[row].copy_(self._current_step)
        self._current_step = torch.where(dones.view(-1).bool(), 0, self._current_step + 1).to(torch.int32)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def num_valid_starts(self, length: int) -> int:
        """Return how many start rows a window of ``length`` transitions could use (before the episode check)."""
        return max(self.size - length, 0)

    def sample_windows(
        self, batch_size: int, length: int, device: str | None = None, require_next: bool = False, oversample: int = 4
    ) -> SequenceReplayBuffer.Window:
        """Sample windows of ``length`` transitions that stay inside one episode.

        A window's last transition may end its episode (then ``next_valid`` is false); with ``require_next`` it may
        not, so all ``length + 1`` observation rows belong to one episode.

        Args:
            batch_size: Number of windows.
            length: Transitions per window.
            device: Device of the returned tensors (defaults to the storage device).
            require_next: Keep only windows whose row ``length`` continues the episode.
            oversample: Candidate starts drawn per requested window before falling back to resampling valid ones.

        Returns:
            The sampled windows.
        """
        device = device or self.device
        num_starts = self.num_valid_starts(length)
        if num_starts <= 0:
            raise RuntimeError(f"Replay holds {self.size} rows per env; a {length}-step window needs {length + 1}.")
        oldest = self.pos if self.size == self.capacity else 0
        # chronological offset c -> ring row (oldest + c) % capacity; row c + length must exist
        n = batch_size * oversample
        start = torch.randint(0, num_starts, (n,), device=self.device)
        env = torch.randint(0, self.num_envs, (n,), device=self.device)
        offsets = torch.arange(length + 1, device=self.device)
        rows = (oldest + start[None, :] + offsets[:, None]) % self.capacity  # (L+1, n)
        steps = self.episode_step[rows, env[None, :]]
        rising = steps - steps[:1] == offsets[:, None]
        inside = rising[:length].all(0)
        next_valid = rising[length]
        last_terminal = self.terminated[rows[length - 1], env]
        valid = inside & (next_valid if require_next else (next_valid | last_terminal))
        idx = valid.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            raise RuntimeError(f"No {length}-step window lies inside one episode; episodes are too short.")
        if len(idx) >= batch_size:
            idx = idx[:batch_size]
        else:
            idx = idx[torch.randint(0, len(idx), (batch_size,), device=self.device)]
        rows, env, next_valid = rows[:, idx], env[idx], next_valid[idx]
        obs = TensorDict(
            {g: self.obs[g][rows, env[None, :]].float().to(device) for g in self.groups},
            batch_size=[length + 1, batch_size],
            device=device,
        )
        return SequenceReplayBuffer.Window(
            obs=obs,
            actions=self.actions[rows[:length], env[None, :]].to(device),
            rewards=self.rewards[rows[:length], env[None, :]].to(device),
            terminated=self.terminated[rows[:length], env[None, :]].float().to(device),
            next_valid=next_valid.to(device),
        )

    def sample_rows(self, batch_size: int, device: str | None = None) -> TensorDict:
        """Sample single observation rows uniformly (e.g. for a sensor-noise bank).

        Args:
            batch_size: Number of rows.
            device: Device of the returned tensors.

        Returns:
            Observations ``(batch_size, ...)`` of every stored group.
        """
        device = device or self.device
        oldest = self.pos if self.size == self.capacity else 0
        rows = (oldest + torch.randint(0, self.size, (batch_size,), device=self.device)) % self.capacity
        env = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return TensorDict(
            {g: self.obs[g][rows, env].float().to(device) for g in self.groups}, batch_size=[batch_size], device=device
        )
