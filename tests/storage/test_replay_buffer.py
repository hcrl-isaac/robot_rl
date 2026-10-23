# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization tests for the off-policy ReplayBuffer (locks FbCpr-facing behavior on CPU)."""

import torch
from tensordict import TensorDict

from robot_rl.storage.replay_buffer import ReplayBuffer

OBS_DIM, ACT_DIM, Z_DIM = 5, 3, 4


def _make_buffer(
    num_envs: int = 6, capacity_per_env: int = 4, batch_size: int = 8, keep_terminal: bool = False, z_dim: int = Z_DIM
) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=num_envs)
    return ReplayBuffer(
        num_envs=num_envs,
        capacity_per_env=capacity_per_env,
        obs=obs,
        actions_shape=(ACT_DIM,),
        z_dim=z_dim,
        batch_size=batch_size,
        device="cpu",
        keep_terminal=keep_terminal,
    )


def _make_transition(num_envs: int, dones: torch.Tensor, z_dim: int = Z_DIM) -> ReplayBuffer.Transition:
    tr = ReplayBuffer.Transition()
    tr.observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.next_observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.actions = torch.randn(num_envs, ACT_DIM)
    tr.rewards = torch.randn(num_envs)
    tr.context = torch.randn(num_envs, z_dim)
    tr.dones = dones
    tr.next_terminated = dones.byte()
    return tr


def _nstep_buffer(
    num_envs: int = 4, capacity_per_env: int = 16, batch_size: int = 64, n_steps: int = 3, gamma: float = 0.5
) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=num_envs)
    return ReplayBuffer(
        num_envs=num_envs,
        capacity_per_env=capacity_per_env,
        obs=obs,
        actions_shape=(ACT_DIM,),
        z_dim=0,
        batch_size=batch_size,
        device="cpu",
        keep_terminal=True,
        n_steps=n_steps,
        gamma=gamma,
    )


def _add_step(buf: ReplayBuffer, num_envs: int, reward: float, done: bool) -> None:
    tr = _make_transition(num_envs, torch.full((num_envs,), done).bool(), z_dim=0)
    tr.rewards = torch.full((num_envs,), reward)
    buf.add_transitions(tr)


class TestReplayBufferNStep:
    """n-step return aggregation (keep_terminal, row-aligned layout)."""

    def test_requires_keep_terminal(self) -> None:
        """n_steps > 1 without keep_terminal is rejected (n-step needs the per-env time sequence)."""
        obs = TensorDict({"policy": torch.zeros(2, OBS_DIM)}, batch_size=2)
        import pytest

        with pytest.raises(ValueError, match="keep_terminal"):
            ReplayBuffer(2, 8, obs, (ACT_DIM,), z_dim=0, batch_size=8, device="cpu", n_steps=3)

    def test_nstep_return_no_dones(self) -> None:
        """With reward=1 everywhere and no episode ends, every 3-step return is 1 + g + g^2 and horizon = n."""
        gamma, n = 0.5, 3
        buf = _nstep_buffer(n_steps=n, gamma=gamma)
        for _ in range(8):  # 8 rows -> valid windows exist (rows 0..5 with max_offset 2)
            _add_step(buf, 4, reward=1.0, done=False)
        batch = buf.sample_mini_batch(device="cpu")
        expected = 1.0 + gamma + gamma**2
        assert batch.effective_n_steps is not None
        assert torch.allclose(batch.rewards, torch.full_like(batch.rewards, expected), atol=1e-5)
        assert (batch.effective_n_steps.view(-1) == n).all()

    def test_nstep_stops_at_episode_end(self) -> None:
        """If every step is an episode end, the return collapses to the single-step reward (horizon = 1)."""
        buf = _nstep_buffer(n_steps=3, gamma=0.5)
        for _ in range(8):
            _add_step(buf, 4, reward=1.0, done=True)
        batch = buf.sample_mini_batch(device="cpu")
        assert torch.allclose(batch.rewards, torch.ones_like(batch.rewards), atol=1e-5)
        assert (batch.effective_n_steps.view(-1) == 1).all()

    def test_single_step_batch_has_no_effective_n(self) -> None:
        """The 1-step path leaves effective_n_steps unset (only n-step populates it)."""
        obs = TensorDict({"policy": torch.zeros(4, OBS_DIM)}, batch_size=4)
        buf = ReplayBuffer(4, 16, obs, (ACT_DIM,), z_dim=0, batch_size=32, device="cpu", keep_terminal=True)
        _add_step(buf, 4, reward=1.0, done=False)
        assert buf.sample_mini_batch(device="cpu").effective_n_steps is None


class TestReplayBuffer:
    """Locks the current add/sample/filter/wrap semantics."""

    def test_add_filters_done_transitions(self) -> None:
        """add_transitions stores only non-done transitions (post-reset next_obs would be invalid)."""
        buf = _make_buffer(num_envs=6)
        dones = torch.tensor([0, 1, 0, 1, 0, 0]).bool()  # 4 valid
        buf.add_transitions(_make_transition(6, dones))
        assert len(buf) == 4

    def test_add_none_dones_stores_all(self) -> None:
        """With dones=None, all transitions are stored."""
        buf = _make_buffer(num_envs=6)
        tr = _make_transition(6, torch.zeros(6).bool())
        tr.dones = None
        buf.add_transitions(tr)
        assert len(buf) == 6

    def test_len_and_circular_wrap(self) -> None:
        """The buffer fills to capacity then wraps, capping len() at capacity."""
        buf = _make_buffer(num_envs=6, capacity_per_env=2)  # capacity = 12
        for _ in range(5):  # 5 * 6 = 30 valid adds into capacity 12
            buf.add_transitions(_make_transition(6, torch.zeros(6).bool()))
        assert len(buf) == 12  # capped at capacity
        assert buf._is_full is True

    def test_sample_shapes(self) -> None:
        """sample_mini_batch returns a Batch with the configured batch size and correct field shapes."""
        buf = _make_buffer(num_envs=6, batch_size=8)
        buf.add_transitions(_make_transition(6, torch.zeros(6).bool()))
        batch = buf.sample_mini_batch(device="cpu")
        assert batch.observations["policy"].shape == (8, OBS_DIM)
        assert batch.next_observations["policy"].shape == (8, OBS_DIM)
        assert batch.actions.shape == (8, ACT_DIM)
        assert batch.rewards.shape == (8,)
        assert batch.context.shape == (8, Z_DIM)
        assert batch.gammas.shape == (8, 1)
        assert batch.next_terminated.shape == (8, 1)

    def test_keep_terminal_stores_all(self) -> None:
        """With keep_terminal=True (SAC), done transitions are retained instead of dropped."""
        buf = _make_buffer(num_envs=6, keep_terminal=True, z_dim=0)
        dones = torch.tensor([0, 1, 0, 1, 1, 0]).bool()  # 3 done, but all kept
        buf.add_transitions(_make_transition(6, dones, z_dim=0))
        assert len(buf) == 6

    def test_keep_terminal_records_next_terminated(self) -> None:
        """The next_terminated flags are stored and recoverable when keeping terminals."""
        buf = _make_buffer(num_envs=4, keep_terminal=True, z_dim=0, batch_size=64)
        dones = torch.tensor([0, 1, 0, 1]).bool()
        buf.add_transitions(_make_transition(4, dones, z_dim=0))
        assert torch.equal(buf.next_terminated[:4].view(-1).bool(), dones)

    def test_stored_values_roundtrip(self) -> None:
        """A single all-valid add should be recoverable (values land in the buffer)."""
        buf = _make_buffer(num_envs=3, batch_size=64)
        tr = _make_transition(3, torch.zeros(3).bool())
        buf.add_transitions(tr)
        # All stored actions should be present among the first len() rows.
        assert torch.allclose(buf.actions[:3], tr.actions)
        assert torch.allclose(buf.context[:3], tr.context)


class TestPostResetTransitionIsDropped:
    """Drop filter keys on the dones of the step that PRODUCED next_obs, not the previous step's (regression test)."""

    def test_only_the_cross_reset_pair_is_dropped(self) -> None:
        """Env 0 ends its episode at step 1; exactly that transition must not be stored."""
        num_envs = 2
        buf = _make_buffer(num_envs=num_envs, capacity_per_env=8)

        def step(marker: float, dones: torch.Tensor) -> None:
            t = ReplayBuffer.Transition()
            t.observations = TensorDict({"policy": torch.full((num_envs, OBS_DIM), marker)}, batch_size=num_envs)
            t.next_observations = TensorDict(
                {"policy": torch.full((num_envs, OBS_DIM), marker + 0.5)}, batch_size=num_envs
            )
            t.actions = torch.zeros(num_envs, ACT_DIM)
            t.rewards = torch.zeros(num_envs)
            t.context = torch.zeros(num_envs, Z_DIM)
            t.next_terminated = torch.zeros(num_envs, dtype=torch.uint8)
            t.dones = dones
            buf.add_transitions(t)

        no_done = torch.zeros(num_envs, dtype=torch.bool)
        env0_done = torch.tensor([True, False])

        step(1.0, no_done)  # both envs fine
        step(2.0, env0_done)  # env 0's next_obs (2.5) IS the post-reset state -> drop env 0 only
        step(3.0, no_done)  # env 0's first transition of the NEW episode -> must be KEPT

        markers = buf.observations["policy"][: len(buf), 0]
        # step 1: 2 rows, step 2: 1 row (env 1 only), step 3: 2 rows
        assert len(buf) == 5, "exactly one transition (the cross-reset pair) should have been dropped"
        assert sorted(markers.tolist()) == [1.0, 1.0, 2.0, 3.0, 3.0]
        # the new episode's first transition survived
        assert (markers == 3.0).sum() == 2


def test_a_group_stored_in_half_precision_samples_back_in_its_own_dtype() -> None:
    """A group stored as float16 is written from float32 and sampled back as float32, to its precision."""
    num_envs = 6
    obs = TensorDict(
        {"policy": torch.zeros(num_envs, OBS_DIM), "history": torch.zeros(num_envs, 7)}, batch_size=num_envs
    )
    buf = ReplayBuffer(num_envs, 4, obs, (ACT_DIM,), Z_DIM, 8, "cpu", obs_dtypes={"history": torch.float16})
    assert buf.observations["history"].dtype == torch.float16 and buf.observations["policy"].dtype == torch.float32
    tr = _make_transition(num_envs, torch.zeros(num_envs, dtype=torch.bool))
    history = torch.randn(num_envs, 7)
    tr.observations["history"] = history
    tr.next_observations["history"] = history + 1.0
    buf.add_transitions(tr)
    batch = buf.sample_mini_batch()
    assert batch.observations["history"].dtype == torch.float32
    assert batch.next_observations["history"].dtype == torch.float32
    rows = [int(torch.nonzero((history - h).abs().max(-1).values < 1e-2)[0]) for h in batch.observations["history"]]
    torch.testing.assert_close(batch.observations["history"], history[rows], atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(batch.next_observations["history"], history[rows] + 1.0, atol=2e-3, rtol=1e-3)
