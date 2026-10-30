"""Time-major replay of contiguous per-env trajectories, sampled as fixed-length windows."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from tensordict import TensorDict


class SequenceReplayBuffer:
    """FIFO replay that stores each step once, as rows ``(capacity, num_envs, ...)``, and samples windows.

    Row ``t`` of env ``e`` holds the observation before the step and that step's action, reward, flags and any
    per-row extras (e.g. a latent). A per-env episode-step counter makes window validity a single check: a window
    lies inside one episode when the counter rises by exactly one along it. Unlike :class:`ReplayBuffer`, nothing is
    stored twice (the next observation is the next row) and a window never spans a reset. The cost: the last
    transition before a time-out has no next row, so windows ending on it are dropped.
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

        extras: dict[str, torch.Tensor] = field(default_factory=dict)
        """Per-row extras for rows ``t .. t+L``, ``(L + 1, B, ...)``."""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        obs: TensorDict,
        groups: list[str],
        num_actions: int,
        device: str = "cpu",
        obs_dtypes: dict[str, str | torch.dtype] | None = None,
        extras: dict[str, int] | None = None,
    ) -> None:
        """Allocate the storage.

        Args:
            num_envs: Number of parallel environments.
            capacity_per_env: Steps kept per environment.
            obs: An observation sample, for the group widths.
            groups: The observation groups to store (only these; nothing else is kept).
            num_actions: Action width.
            device: Storage device.
            obs_dtypes: Per-group storage dtypes (names or torch dtypes, e.g. ``{"goal": "float16"}``); sampled back
                as float32.
            extras: Per-row algorithm-side tensors to store, name to width (e.g. ``{"z": 256}``).
        """
        self.num_envs = num_envs
        self.capacity = capacity_per_env
        self.device = device
        self.groups = list(dict.fromkeys(groups))
        obs_dtypes = obs_dtypes or {}
        shape = (capacity_per_env, num_envs)

        def dtype(g: str) -> torch.dtype:
            d = obs_dtypes.get(g, torch.float32)
            return getattr(torch, d) if isinstance(d, str) else d

        self.obs = {g: torch.zeros(*shape, obs[g].shape[-1], dtype=dtype(g), device=device) for g in self.groups}
        self.extras = {k: torch.zeros(*shape, w, device=device) for k, w in (extras or {}).items()}
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
        extras: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Append one step of every env.

        Args:
            obs: Observations before the step.
            actions: Executed actions.
            rewards: Rewards of the step.
            terminated: Termination flags (dones that are not time-outs).
            dones: All episode ends (terminations and time-outs); the next row starts a new episode.
            extras: Values of the declared extras for this row.
        """
        row = self.pos
        for g in self.groups:
            self.obs[g][row].copy_(obs[g])
        for k, v in (extras or {}).items():
            self.extras[k][row].copy_(v)
        self.actions[row].copy_(actions)
        self.rewards[row].copy_(rewards.view(-1))
        self.terminated[row].copy_(terminated.view(-1).bool())
        self.episode_step[row].copy_(self._current_step)
        self._current_step = torch.where(dones.view(-1).bool(), 0, self._current_step + 1).to(torch.int32)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def mark_episode_start(self, env_ids: torch.Tensor | None = None) -> None:
        """Start a new episode at the next row, for resets outside the env step (e.g. after an eval)."""
        if env_ids is None:
            self._current_step.zero_()
        else:
            self._current_step[env_ids] = 0

    def sample_windows(
        self, batch_size: int, length: int, device: str | None = None, require_next: bool = False, oversample: int = 4
    ) -> SequenceReplayBuffer.Window | None:
        """Sample windows of ``length`` transitions that stay inside one episode.

        A window's last transition may end its episode by termination (then ``next_valid`` is false); with
        ``require_next`` it may not, so all ``length + 1`` observation rows belong to one episode.

        Args:
            batch_size: Number of windows.
            length: Transitions per window.
            device: Device of the returned tensors (defaults to the storage device).
            require_next: Keep only windows whose row ``length`` continues the episode.
            oversample: Candidate starts drawn per requested window before falling back to resampling valid ones.

        Returns:
            The sampled windows, or None when no candidate lies inside one episode (too little data, or episodes
            shorter than the window).
        """
        device = device or self.device
        num_starts = self.size - length
        if num_starts <= 0:
            return None
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
            return None
        if len(idx) >= batch_size:
            idx = idx[:batch_size]
        else:
            idx = idx[torch.randint(0, len(idx), (batch_size,), device=self.device)]
        rows, env, next_valid = rows[:, idx], env[idx], next_valid[idx]
        cols = env[None, :]
        return SequenceReplayBuffer.Window(
            obs=TensorDict(
                {g: self.obs[g][rows, cols].float().to(device) for g in self.groups},
                batch_size=[length + 1, batch_size],
                device=device,
            ),
            actions=self.actions[rows[:length], cols].to(device),
            rewards=self.rewards[rows[:length], cols].to(device),
            terminated=self.terminated[rows[:length], cols].float().to(device),
            next_valid=next_valid.to(device),
            extras={k: v[rows, cols].to(device) for k, v in self.extras.items()},
        )

    def sample_transitions(self, batch_size: int, device: str | None = None) -> SequenceReplayBuffer.Window | None:
        """Sample single transitions (one-step windows), e.g. for one-step critic targets.

        Args:
            batch_size: Number of transitions.
            device: Device of the returned tensors.

        Returns:
            One-step windows: ``obs[0]`` before and ``obs[1]`` after the step (valid where ``next_valid``), or None.
        """
        return self.sample_windows(batch_size, 1, device=device)

    def sample_rows(self, batch_size: int, device: str | None = None) -> tuple[TensorDict, dict[str, torch.Tensor]]:
        """Sample single rows uniformly (imagination starts, a sensor-noise bank).

        Args:
            batch_size: Number of rows.
            device: Device of the returned tensors.

        Returns:
            Observations ``(batch_size, ...)`` of every stored group, and the rows' extras.
        """
        device = device or self.device
        oldest = self.pos if self.size == self.capacity else 0
        rows = (oldest + torch.randint(0, self.size, (batch_size,), device=self.device)) % self.capacity
        env = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        obs = TensorDict(
            {g: self.obs[g][rows, env].float().to(device) for g in self.groups}, batch_size=[batch_size], device=device
        )
        return obs, {k: v[rows, env].to(device) for k, v in self.extras.items()}
