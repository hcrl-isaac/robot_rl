# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from robot_rl.modules import HiddenState
from robot_rl.utils import split_and_pad_trajectories


def _hidden_state_to_device(hs: HiddenState | list[torch.Tensor], device: str | None) -> HiddenState:
    """Move a hidden state to ``device``. Handles ``None``, single tensors, and per-layer lists/tuples."""
    if hs is None:
        return None
    if isinstance(hs, (list, tuple)):
        return type(hs)(t.to(device) for t in hs)  # type: ignore[return-value]
    return hs.to(device)


class RolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`RolloutStorage.add_transition` to record the data.
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
            """Done flags indicating episode termination."""

            # For reinforcement learning
            self.values: torch.Tensor | None = None
            """Value estimates at the current step (RL only)."""

            self.actions_log_prob: torch.Tensor | None = None
            """Log probability of the taken actions (RL only)."""

            self.distribution_params: tuple[torch.Tensor, ...] | None = None
            """Parameters of the action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = None
            """Privileged (teacher) actions (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)
            """Hidden states for recurrent networks, e.g., (actor, critic)."""

            self.memory_hidden_state: HiddenState = None
            """Hidden state of a shared memory module (used when ``MetaRlCfg.memory`` is set)."""

            # For meta-reinforcement learning
            self.meta_dones: torch.Tensor | None = None
            """Done flags indicating trial termination."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the rollout storage generators.

        This class provides named access to mini-batch fields. Fields are optional to support different training modes
        (RL vs distillation) and architectures (feedforward vs recurrent).
        """

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            memory_hidden_state: HiddenState = None,
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
            device: str | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict | None = observations
            """Batch of observations."""

            # For reinforcement learning
            self.actions: torch.Tensor | None = actions
            """Batch of actions."""

            self.values: torch.Tensor | None = values
            """Batch of value estimates (RL only)."""

            self.advantages: torch.Tensor | None = advantages
            """Batch of advantage estimates (RL only)."""

            self.returns: torch.Tensor | None = returns
            """Batch of return targets (RL only)."""

            self.old_actions_log_prob: torch.Tensor | None = old_actions_log_prob
            """Batch of log probabilities of the old actions (RL only)."""

            self.old_distribution_params: tuple[torch.Tensor, ...] | None = old_distribution_params
            """Batch of parameters of the old action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = privileged_actions
            """Batch of privileged (teacher) actions (distillation only)."""

            self.dones: torch.Tensor | None = dones
            """Batch of done flags (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = hidden_states
            """Batch of hidden states for recurrent networks (RL recurrent only)."""

            self.memory_hidden_state: HiddenState = memory_hidden_state
            """Batch of hidden states for the shared memory module (RL recurrent only)."""

            self.masks: torch.Tensor | None = masks
            """Batch of trajectory masks for recurrent networks (RL recurrent only)."""

            self._set_device(device)

        def _set_device(self, device: str | None) -> None:
            """Move all populated batch fields to ``device`` (no-op when ``device is None``)."""
            if device is None:
                return
            if self.observations is not None:
                self.observations = self.observations.to(device)
            if self.actions is not None:
                self.actions = self.actions.to(device)
            if self.values is not None:
                self.values = self.values.to(device)
            if self.advantages is not None:
                self.advantages = self.advantages.to(device)
            if self.returns is not None:
                self.returns = self.returns.to(device)
            if self.old_actions_log_prob is not None:
                self.old_actions_log_prob = self.old_actions_log_prob.to(device)
            if self.old_distribution_params is not None:
                self.old_distribution_params = tuple(p.to(device) for p in self.old_distribution_params)
            if self.privileged_actions is not None:
                self.privileged_actions = self.privileged_actions.to(device)
            if self.dones is not None:
                self.dones = self.dones.to(device)
            self.hidden_states = (
                _hidden_state_to_device(self.hidden_states[0], device),
                _hidden_state_to_device(self.hidden_states[1], device),
            )
            self.memory_hidden_state = _hidden_state_to_device(self.memory_hidden_state, device)
            if self.masks is not None:
                self.masks = self.masks.to(device)

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        num_reward_streams: int = 1,
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape.

        ``num_reward_streams`` > 1 stores a reward, value and return per stream (one critic head each); the
        advantage the policy loss reads stays a single column combined by the algorithm.
        """
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.num_reward_streams = num_reward_streams
        streams = num_reward_streams

        # Core
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, streams, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # for reinforcement learning
        elif training_type in ["meta_rl", "rl"]:
            self.values = torch.zeros(num_transitions_per_env, num_envs, streams, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.distribution_params: tuple[torch.Tensor, ...] | None = None  # Lazily initialized on first transition
            self.returns = torch.zeros(num_transitions_per_env, num_envs, streams, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

            if training_type == "meta_rl":
                self.meta_dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For recurrent networks. Hidden states are saved sparsely: only at trajectory-start indices.
        self._pending_traj_a: list[tuple[torch.Tensor, list[torch.Tensor]]] | None = None
        self._pending_traj_c: list[tuple[torch.Tensor, list[torch.Tensor]]] | None = None
        self._pending_traj_m: list[tuple[torch.Tensor, list[torch.Tensor]]] | None = None
        self.saved_hidden_state_a: list[torch.Tensor] | None = None
        self.saved_hidden_state_c: list[torch.Tensor] | None = None
        self.saved_hidden_state_m: list[torch.Tensor] | None = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(self.num_envs, -1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)  # type: ignore
        # For reinforcement learning
        elif self.training_type in ["meta_rl", "rl"]:
            self.values[self.step].copy_(transition.values)  # type: ignore
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if self.distribution_params is None:  # Initialize the distribution parameters
                self.distribution_params = tuple(
                    torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                    for p in transition.distribution_params  # type: ignore
                )
            for i, p in enumerate(transition.distribution_params):  # type: ignore
                self.distribution_params[i][self.step].copy_(p)
            # For meta-reinforcement learning
            if self.training_type == "meta_rl":
                self.meta_dones[self.step].copy_(transition.meta_dones.view(-1, 1))

        # For RNN networks
        self._save_hidden_states(transition.hidden_states, transition.memory_hidden_state)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0
        self._pending_traj_a = None
        self._pending_traj_c = None
        self._pending_traj_m = None
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None
        self.saved_hidden_state_m = None

    # For distillation
    def generator(self, device: str | None = None) -> Generator[Batch, None, None]:
        """Yield per-timestep batches for distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield RolloutStorage.Batch(
                observations=self.observations[i],  # type: ignore
                privileged_actions=self.privileged_actions[i],
                dones=self.dones[i],
                device=device,
            )

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, device: str | None = None
    ) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        if self.training_type not in ["meta_rl", "rl"]:
            raise ValueError(
                "This function is only available for reinforcement learning and meta-reinforcement learning training."
            )
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten the data
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Yield the mini-batch
                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    device=device,
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, device: str | None = None
    ) -> Generator[Batch, None, None]:
        """Yield trajectory mini-batches with masks and recurrent hidden states."""
        if self.training_type not in ["meta_rl", "rl"]:
            raise ValueError(
                "This function is only available for reinforcement learning and meta-reinforcement learning training."
            )
        # Reset memory at trial boundary if in meta RL, otherwise reset at episode boundary
        mem_bounds = self.meta_dones if self.training_type == "meta_rl" else self.dones
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, mem_bounds)
        mini_batch_size = self.num_envs // num_mini_batches
        mem_bounds = mem_bounds.squeeze(-1)

        # Flatten the per-step pending hidden states into env-major time-order [total_trajs, ...] tensors.
        self._finalize_traj_hidden_states()

        # Per-env trajectory counts let us resolve [first_traj, last_traj) into the env-major flat tensor.
        last_was_done = torch.zeros_like(mem_bounds, dtype=torch.bool)
        last_was_done[1:] = mem_bounds[:-1]
        last_was_done[0] = True
        trajs_per_env = last_was_done.sum(dim=0)

        for ep in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                first_traj = int(trajs_per_env[:start].sum().item())
                last_traj = int(trajs_per_env[:stop].sum().item())

                hidden_state_a_batch = self._slice_saved_hidden_states(self.saved_hidden_state_a, first_traj, last_traj)
                hidden_state_c_batch = self._slice_saved_hidden_states(self.saved_hidden_state_c, first_traj, last_traj)
                hidden_state_m_batch = self._slice_saved_hidden_states(self.saved_hidden_state_m, first_traj, last_traj)

                # Yield the mini-batch
                yield RolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(hidden_state_a_batch, hidden_state_c_batch),
                    memory_hidden_state=hidden_state_m_batch,
                    masks=trajectory_masks[:, first_traj:last_traj],
                    device=device,
                )

    @staticmethod
    def _slice_saved_hidden_states(saved: list[torch.Tensor] | None, first_traj: int, last_traj: int) -> HiddenState:
        """Slice the env-major flat hidden-state tensors to a trajectory mini-batch."""
        if saved is None:
            return None
        sliced = [t[first_traj:last_traj].transpose(0, 1).contiguous() for t in saved]
        return sliced[0] if len(sliced) == 1 else sliced

    def _save_hidden_states(
        self,
        hidden_states: tuple[HiddenState, HiddenState],
        memory_hidden_state: HiddenState = None,
    ) -> None:
        """Save recurrent hidden states for actor, critic, and shared memory to the rollout storage.

        Only the hidden states at trajectory-start envs (step 0 for all envs, otherwise envs whose
        previous step was a done) are kept — these are the only ones the recurrent generator reads.
        """
        if hidden_states == (None, None) and memory_hidden_state is None:
            return
        # Wrap GRU/single-tensor states as tuples to match the LSTM/multi-layer format.
        hidden_state_a = (
            None
            if hidden_states[0] is None
            else (hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],))
        )
        hidden_state_c = (
            None
            if hidden_states[1] is None
            else (hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],))
        )
        hidden_state_m = (
            None
            if memory_hidden_state is None
            else (memory_hidden_state if isinstance(memory_hidden_state, tuple) else (memory_hidden_state,))
        )

        # Determine the env indices that start a new trajectory at this step.
        if self.step == 0:
            env_indices_storage = torch.arange(self.num_envs, device=self.device)
            all_envs = True
        else:
            prev_done_field = self.meta_dones if self.training_type == "meta_rl" else self.dones
            prev_done = prev_done_field[self.step - 1].squeeze(-1).bool()
            env_indices_storage = prev_done.nonzero(as_tuple=True)[0]
            if env_indices_storage.numel() == 0:
                return
            all_envs = False

        if hidden_state_a is not None:
            self._pending_traj_a = self._append_traj_starts(
                self._pending_traj_a, hidden_state_a, env_indices_storage, all_envs
            )
        if hidden_state_c is not None:
            self._pending_traj_c = self._append_traj_starts(
                self._pending_traj_c, hidden_state_c, env_indices_storage, all_envs
            )
        if hidden_state_m is not None:
            self._pending_traj_m = self._append_traj_starts(
                self._pending_traj_m, hidden_state_m, env_indices_storage, all_envs
            )

    def _append_traj_starts(
        self,
        pending: list[tuple[torch.Tensor, list[torch.Tensor]]] | None,
        hs_tuple: tuple[torch.Tensor, ...],
        env_indices_storage: torch.Tensor,
        all_envs: bool,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """Slice ``hs_tuple`` to the trajectory-start envs and append a snapshot to ``pending``.

        Each per-layer tensor in ``hs_tuple`` has env at dim 1; output slices have env moved to dim 0
        with shape ``[num_starts, *per_env_shape]`` on storage device.
        """
        compute_device = hs_tuple[0].device
        if all_envs:
            sliced = [hs.movedim(1, 0).contiguous().to(self.device) for hs in hs_tuple]
        else:
            env_idx_compute = env_indices_storage.to(compute_device)
            sliced = [
                hs.index_select(dim=1, index=env_idx_compute).movedim(1, 0).contiguous().to(self.device)
                for hs in hs_tuple
            ]
        if pending is None:
            pending = []
        pending.append((env_indices_storage, sliced))
        return pending

    def _finalize_traj_hidden_states(self) -> None:
        """Flatten per-step ``_pending_traj_X`` lists into env-major time-order tensors.

        Idempotent: re-runs only when the corresponding ``saved_hidden_state_X`` is still ``None``.
        Each pending entry contributes ``num_starts_at_step`` trajectories in env-order; concatenating
        across steps gives time-major order, then a stable argsort on env indices reorders to env-major
        while preserving time-order within each env (matching ``split_and_pad_trajectories``).
        """
        if self.saved_hidden_state_a is None and self._pending_traj_a is not None:
            self.saved_hidden_state_a = self._flatten_traj_hidden_states(self._pending_traj_a)
        if self.saved_hidden_state_c is None and self._pending_traj_c is not None:
            self.saved_hidden_state_c = self._flatten_traj_hidden_states(self._pending_traj_c)
        if self.saved_hidden_state_m is None and self._pending_traj_m is not None:
            self.saved_hidden_state_m = self._flatten_traj_hidden_states(self._pending_traj_m)

    @staticmethod
    def _flatten_traj_hidden_states(
        pending: list[tuple[torch.Tensor, list[torch.Tensor]]],
    ) -> list[torch.Tensor]:
        """Concatenate pending per-step entries and reorder to env-major time-order."""
        env_indices_concat = torch.cat([e for e, _ in pending], dim=0)
        num_layers = len(pending[0][1])
        per_layer_concat = [torch.cat([slices[layer] for _, slices in pending], dim=0) for layer in range(num_layers)]
        perm = torch.argsort(env_indices_concat, stable=True)
        return [t[perm] for t in per_layer_concat]
