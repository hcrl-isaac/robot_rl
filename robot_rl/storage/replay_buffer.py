# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Storage for the data collected across rollouts.

    The replay storage is populated by adding transitions during the rollout phase.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`ReplayBuffer.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.dones: torch.Tensor | None = None
            """Done flags indicating episode termination or timeout at the current step."""

            self.context: torch.Tensor | None = None
            """Latent context (z) vectors at the current step."""

            self.next_observations: TensorDict | None = None
            """Observations after the current step."""

            self.next_terminated: torch.Tensor | None = None
            """Done flags indicating episode termination after the current step."""

            self.hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
            """Recurrent state BEFORE this step, ``(layers, num_envs, hidden)``; LSTM passes ``(h, c)``."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the replay buffer.

        This class provides named access to mini-batch fields.
        """

        def __init__(
            self,
            observations: TensorDict,
            next_observations: TensorDict,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            gammas: torch.Tensor,
            context: torch.Tensor,
            next_terminated: torch.Tensor | None = None,
            effective_n_steps: torch.Tensor | None = None,
            hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict = observations
            """Batch of observations."""

            self.next_observations: TensorDict = next_observations
            """Batch of next observations."""

            self.actions: torch.Tensor = actions
            """Batch of actions."""

            self.rewards: torch.Tensor = rewards
            """Batch of rewards."""

            self.gammas: torch.Tensor = gammas
            """Batch of gammas."""

            self.context: torch.Tensor = context
            """Batch of latent context (z) vectors."""

            self.next_terminated: torch.Tensor | None = next_terminated
            """Batch of terminated flags after the step (true termination only)."""

            self.effective_n_steps: torch.Tensor | None = effective_n_steps
            """Per-sample number of steps actually aggregated (<= n_steps; 1-step or capped at an episode end)."""
            self.hidden = hidden
            """Stored recurrent state before each step, ``(layers, N, hidden)``; ``None`` if the buffer stores none."""

    class SequenceBatch:
        """A time-major batch of contiguous per-env windows, for recurrent updates.

        Tensors are ``(L, B, ...)``. A window may span several episodes: ``resets`` marks the steps that
        open a new one, where the recurrent state must be zeroed before the step is consumed. The
        leading ``burn_in`` steps are meant to be replayed WITHOUT gradients to re-derive the hidden
        state; only the steps after them carry a loss.
        """

        def __init__(
            self,
            observations: TensorDict,
            next_observations: TensorDict,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            gammas: torch.Tensor,
            context: torch.Tensor,
            next_terminated: torch.Tensor,
            resets: torch.Tensor,
            init_hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
            burn_in: int,
            burn_hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
            ages: torch.Tensor | None = None,
        ) -> None:
            """Initialize a sequence batch over contiguous rollout windows."""
            self.observations: TensorDict = observations
            self.next_observations: TensorDict = next_observations
            self.actions: torch.Tensor = actions
            self.rewards: torch.Tensor = rewards
            self.gammas: torch.Tensor = gammas
            self.context: torch.Tensor = context
            self.next_terminated: torch.Tensor = next_terminated
            self.resets: torch.Tensor = resets
            """``(L, B)`` 1 at steps that begin a new episode (the previous step was a done); row 0 is 0."""
            self.init_hidden = init_hidden
            """Stored recurrent state at the window's first step; ``None`` if the buffer stores none."""
            self.burn_in: int = burn_in
            """Leading steps to replay without gradients before computing any loss."""
            self.burn_hidden = burn_hidden
            """Stored state before step ``burn_in``: what the rollout actor held where the loss segment starts."""
            self.ages = ages
            """``(B,)`` window age as a fraction of the buffer's per-env capacity (0 = just written)."""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        z_dim: int,
        batch_size: int,
        device: str = "cpu",
        keep_terminal: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        hidden_dim: int = 0,
        hidden_layers: int = 1,
        hidden_is_lstm: bool = False,
    ) -> None:
        """Initialize the buffer storage.

        Args:
            num_envs: Number of parallel environments feeding the buffer.
            capacity_per_env: Stored transitions per environment (total capacity = ``capacity_per_env * num_envs``).
            obs: A representative observation TensorDict used to size the obs/next-obs storage.
            actions_shape: Shape of a single environment's action.
            z_dim: Latent-context dimension (use ``0`` for algorithms without a latent, e.g. SAC).
            batch_size: Mini-batch size returned by :meth:`sample_mini_batch`.
            device: Storage device.
            keep_terminal: If ``False`` (default; FbCpr), transitions whose ``dones`` is set are dropped, since
                their stored next-obs is a post-reset state. If ``True`` (SAC), all transitions are kept and the
                caller supplies the true pre-reset ``next_observations`` and ``next_terminated``.
            n_steps: Number of steps for n-step returns; ``1`` (default) is single-step. ``>1`` requires
                ``keep_terminal=True`` and returns the discounted n-step return, stopped at episode boundaries.
            gamma: Discount factor used for the n-step return (unused for ``n_steps == 1``).
            hidden_dim: Recurrent hidden size stored per transition; ``0`` (default) stores none.
            hidden_layers: Number of recurrent layers, for sizing the stored state.
            hidden_is_lstm: Store two slots per step (``h`` and ``c``) instead of one.
        """
        # store inputs
        self.num_envs = num_envs
        self.capacity_per_env = capacity_per_env
        self.capacity = capacity_per_env * num_envs
        self.device = device
        self.keep_terminal = keep_terminal
        self.n_steps = n_steps
        self.gamma = gamma
        if n_steps > 1 and not keep_terminal:
            raise ValueError("n_steps > 1 requires keep_terminal=True (n-step needs the per-env time sequence).")

        # Core
        # We only take value.shape[1:] to ignore num_envs dimension
        self.observations = TensorDict(
            {
                key: torch.zeros(self.capacity, *value.shape[1:], dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=self.capacity,
            device=self.device,
        )
        self.actions = torch.zeros(self.capacity, *actions_shape, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.context = torch.zeros(self.capacity, z_dim, device=self.device)
        self.next_observations: TensorDict = self.observations.clone()
        self.next_terminated = torch.zeros(self.capacity, 1, device=self.device).byte()
        self.dones = torch.zeros(self.capacity, 1, device=self.device).byte()  # episode ends (n-step boundaries)
        self.gammas = torch.zeros(self.capacity, 1, device=self.device)

        # Recurrent state as it was BEFORE each stored step, so a sampled window can be replayed from
        # its own start. Off-policy the stored state goes stale as the network trains, which is what
        # the burn-in prefix in :meth:`sample_sequences` exists to re-derive.
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.hidden_is_lstm = hidden_is_lstm
        self.hidden: torch.Tensor | None = None
        if hidden_dim > 0:
            slots = 2 if hidden_is_lstm else 1
            self.hidden = torch.zeros(self.capacity, slots, hidden_layers, hidden_dim, device=self.device)

        # counter for the number of transitions stored
        self._curr_idx = 0
        self._is_full = False

        # tensor for selecting mini batches
        self._indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)

    def __len__(self) -> int:
        """Get the total number of transitions currently stored in the buffer."""
        return self.capacity if self._is_full else self._curr_idx

    def add_transitions(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        # Drop transitions whose next_obs is a post-reset state (dones set), unless keep_terminal is set (SAC,
        # where next_obs is the true pre-reset obs). If dones is None, all are valid.
        if transition.dones is None or self.keep_terminal:
            valid_idxs = torch.arange(self.num_envs, device=self.device)
        else:
            valid_idxs = torch.argwhere(~transition.dones.view(-1)).flatten()
        num_valid = len(valid_idxs)
        # Exit if no transitions are valid to store
        if num_valid == 0:
            return
        # Raise error if buffer is too small to store transitions
        if num_valid >= self.capacity:
            raise RuntimeError(
                f"Cannot store {num_valid} transitions in replay buffer of size {self.capacity}. "
                "You may need to increase buffer capacity or decrease num_envs."
            )

        buf_idxs = (torch.arange(0, num_valid, device=self.device) + self._curr_idx) % self.capacity
        self.observations.update_at_(transition.observations[valid_idxs].to(self.device), buf_idxs)  # type: ignore
        self.next_observations.update_at_(transition.next_observations[valid_idxs].to(self.device), buf_idxs)  # type: ignore
        self.actions.index_copy_(0, buf_idxs, transition.actions[valid_idxs].to(self.device))  # type: ignore
        self.rewards.index_copy_(0, buf_idxs, transition.rewards[valid_idxs].to(self.device))  # type: ignore
        self.context.index_copy_(0, buf_idxs, transition.context[valid_idxs].to(self.device))  # type: ignore
        self.next_terminated.index_copy_(
            0, buf_idxs, transition.next_terminated[valid_idxs].to(self.device).unsqueeze(-1)
        )  # type: ignore
        # Episode-end flags (for n-step boundaries); zeros when the caller doesn't provide dones.
        dones_src = transition.dones if transition.dones is not None else torch.zeros(self.num_envs, device=self.device)
        self.dones.index_copy_(0, buf_idxs, dones_src.view(-1)[valid_idxs].byte().to(self.device).unsqueeze(-1))

        if self.hidden is not None and transition.hidden_state is not None:
            hs = transition.hidden_state
            parts = list(hs) if isinstance(hs, (tuple, list)) else [hs]
            # (layers, num_envs, hidden) per slot -> (num_valid, slots, layers, hidden)
            stacked = torch.stack([p.detach().to(self.device) for p in parts], dim=0)
            self.hidden.index_copy_(0, buf_idxs, stacked[:, :, valid_idxs].permute(2, 0, 1, 3).contiguous())

        # increment the counter
        self._curr_idx += num_valid
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def sample_mini_batch(self, device: str | None = None) -> Batch:
        """Randomly sample a mini-batch from the replay buffer (with n-step aggregation when ``n_steps > 1``)."""
        if self.n_steps > 1:
            nstep = self._sample_nstep(device)
            if nstep is not None:
                return nstep
            # not enough consecutive data for a full n-step yet -> fall through to single-step sampling

        # Sample indices from uniform distribution
        self._indices.random_(0, len(self))

        return ReplayBuffer.Batch(
            self.observations[self._indices].to(device),
            self.next_observations[self._indices].to(device),
            self.actions[self._indices].to(device),
            self.rewards[self._indices].to(device),
            self.gammas[self._indices].to(device),
            self.context[self._indices].to(device),
            self.next_terminated[self._indices].to(device),
            hidden=self._gather_hidden(self._indices, device) if self.hidden is not None else None,
        )

    def _sample_nstep(self, device: str | None = None) -> Batch | None:
        """Sample a mini-batch with n-step returns over the per-env time sequence (keep_terminal layout).

        A stored index is ``row * num_envs + env``, so env ``e``'s consecutive steps are ``num_envs`` apart. Start
        indices whose n-step window would cross the circular write head are excluded; the discounted return is
        summed until the first episode end, and ``effective_n_steps`` records how many steps were aggregated.
        Returns ``None`` if there is not yet a full n-step window anywhere in the buffer.
        """
        batch_size = self._indices.shape[0]
        cap_rows = self.capacity_per_env
        filled_rows = cap_rows if self._is_full else self._curr_idx // self.num_envs
        write_row = self._curr_idx // self.num_envs
        max_offset = self.n_steps - 1

        # Valid start rows: their n-step window must not cross the write head (same mask for every env).
        rows = torch.arange(filled_rows, device=self.device)
        if self._is_full:
            before = rows < write_row
            safe = torch.where(before, (rows + max_offset) < write_row, (rows + max_offset) < (cap_rows + write_row))
        else:
            safe = (rows + max_offset) < filled_rows
        valid_rows = rows[safe]
        if valid_rows.numel() == 0:
            return None

        # Sample (start row, env) pairs and build the n consecutive flat indices per sample.
        start_rows = valid_rows[torch.randint(valid_rows.numel(), (batch_size,), device=self.device)]
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        start_flat = start_rows * self.num_envs + envs
        offsets = torch.arange(self.n_steps, device=self.device)
        step_rows = (start_rows.unsqueeze(-1) + offsets) % cap_rows  # [B, n]
        step_flat = step_rows * self.num_envs + envs.unsqueeze(-1)  # [B, n]

        all_rewards = self.rewards[step_flat]  # [B, n]
        all_dones = self.dones[step_flat].squeeze(-1).float()  # [B, n]

        # Mask that is 1 up to and including the first episode end, 0 afterwards; discounted n-step return.
        dones_shifted = torch.cat([torch.zeros_like(all_dones[..., :1]), all_dones[..., :-1]], dim=-1)
        done_masks = torch.cumprod(1.0 - dones_shifted, dim=-1)
        discounts = torch.pow(torch.tensor(self.gamma, device=self.device), offsets.float())
        n_step_rewards = (all_rewards * done_masks * discounts.view(1, -1)).sum(dim=-1)  # [B]

        # First episode end within the window -> effective horizon + the index whose next-state we bootstrap from.
        first_done = torch.argmax((all_dones > 0).float(), dim=-1)
        no_dones = all_dones.sum(dim=-1) == 0
        first_done = torch.where(no_dones, torch.full_like(first_done, self.n_steps - 1), first_done)
        effective_n = (first_done + 1).unsqueeze(-1).long()  # [B, 1]
        final_flat = step_flat.gather(1, first_done.unsqueeze(-1)).squeeze(-1)  # [B]

        return ReplayBuffer.Batch(
            self.observations[start_flat].to(device),
            self.next_observations[final_flat].to(device),
            self.actions[start_flat].to(device),
            n_step_rewards.to(device),
            self.gammas[start_flat].to(device),
            self.context[start_flat].to(device),
            self.next_terminated[final_flat].to(device),
            effective_n.to(device),
        )

    def sample_sequences(
        self, seq_len: int, burn_in: int = 0, device: str | None = None, num_windows: int | None = None
    ) -> SequenceBatch | None:
        """Sample contiguous per-env windows for a recurrent update.

        A stored index is ``row * num_envs + env``, so env ``e``'s consecutive steps are ``num_envs``
        apart -- the same walk :meth:`_sample_nstep` uses. Windows that would cross the circular write
        head are excluded. Every step is a valid transition (the buffer keeps terminal steps with their
        true next observation); episode boundaries inside a window are reported as ``resets``.

        Args:
            seq_len: Steps that carry a loss.
            burn_in: Leading steps replayed without gradients to re-derive the hidden state. Stored
                states drift as the network trains, so off-policy this should be nonzero.
            device: Device to move the batch to.
            num_windows: Windows per batch. Defaults to ``batch_size // (burn_in + seq_len)`` so the
                flattened steps match the configured mini-batch; a window batch otherwise multiplies
                the mini-batch by the window length, which with image observations is an easy OOM.

        Returns:
            A :class:`SequenceBatch` of ``L = burn_in + seq_len`` steps, or ``None`` while no window of
            that length fits in the stored data.
        """
        total_len = burn_in + seq_len
        if total_len < 1:
            raise ValueError("sample_sequences needs burn_in + seq_len >= 1")
        batch_size = num_windows if num_windows is not None else max(self._indices.shape[0] // total_len, 1)
        cap_rows = self.capacity_per_env
        filled_rows = cap_rows if self._is_full else self._curr_idx // self.num_envs
        write_row = self._curr_idx // self.num_envs
        max_offset = total_len - 1

        rows = torch.arange(filled_rows, device=self.device)
        if self._is_full:
            before = rows < write_row
            safe = torch.where(before, (rows + max_offset) < write_row, (rows + max_offset) < (cap_rows + write_row))
        else:
            safe = (rows + max_offset) < filled_rows
        valid_rows = rows[safe]
        if valid_rows.numel() == 0:
            return None

        start_rows = valid_rows[torch.randint(valid_rows.numel(), (batch_size,), device=self.device)]
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        offsets = torch.arange(total_len, device=self.device)
        step_rows = (start_rows.unsqueeze(-1) + offsets) % cap_rows  # [B, L]
        step_flat = (step_rows * self.num_envs + envs.unsqueeze(-1)).reshape(-1)  # [B*L]

        def _tm(flat: torch.Tensor) -> torch.Tensor:
            """Gathered [B*L, ...] -> time-major [L, B, ...]."""
            return flat.reshape(batch_size, total_len, *flat.shape[1:]).transpose(0, 1).contiguous()

        # a step opens a new episode when the previous stored step was a done; the window's first step
        # starts from its stored state, which is already zero if it begins an episode
        dones = self.dones[step_flat].reshape(batch_size, total_len).float()
        resets = torch.cat([torch.zeros_like(dones[:, :1]), dones[:, :-1]], dim=1).transpose(0, 1).contiguous()

        init_hidden = burn_hidden = None
        if self.hidden is not None:
            init_hidden = self._gather_hidden(step_rows[:, 0] * self.num_envs + envs, device)
            burn_hidden = self._gather_hidden(step_rows[:, min(burn_in, max_offset)] * self.num_envs + envs, device)
        ages = ((write_row - start_rows) % cap_rows).float() / cap_rows

        obs = self.observations[step_flat].reshape(batch_size, total_len).transpose(0, 1).to(device)
        next_obs = self.next_observations[step_flat].reshape(batch_size, total_len).transpose(0, 1).to(device)
        return ReplayBuffer.SequenceBatch(
            obs,
            next_obs,
            _tm(self.actions[step_flat]).to(device),
            _tm(self.rewards[step_flat]).to(device),
            _tm(self.gammas[step_flat]).to(device),
            _tm(self.context[step_flat]).to(device),
            _tm(self.next_terminated[step_flat]).to(device),
            resets.to(device),
            init_hidden,
            burn_in,
            burn_hidden,
            ages.to(device),
        )

    def _gather_hidden(
        self, flat_idx: torch.Tensor, device: str | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Gather the stored recurrent state at ``flat_idx`` as ``(layers, B, hidden)``; ``(h, c)`` for LSTMs."""
        h = self.hidden[flat_idx].permute(1, 2, 0, 3).contiguous()  # type: ignore[index]  # (slots, layers, B, H)
        return (h[0].to(device), h[1].to(device)) if self.hidden_is_lstm else h[0].to(device)
