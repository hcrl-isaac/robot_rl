# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the shared OffPolicyRunner drives SAC via its generic loop (dummy VecEnv, no Isaac Sim)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from pathlib import Path
from tensordict import TensorDict

from robot_rl.algorithms import SAC
from robot_rl.runners import OffPolicyRunner

NUM_ENVS, OBS_DIM, NUM_ACTIONS, MAX_EP_LEN = 4, 6, 3, 20


class DummyEnv:
    """Minimal VecEnv exposing the interface the generic off-policy loop needs, incl. time_outs(_obs)."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.common_step_counter = 0
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # noqa: D102
        data = {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}
        return TensorDict(data, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        timed_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.rand(self.num_envs, device=self.device) < 0.05
        dones = (timed_out | terminated).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {
            "time_outs": timed_out.float(),
            "time_outs_obs": TensorDict(
                {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}, batch_size=[self.num_envs]
            ),
        }
        return obs, rewards, dones, extras

    def apply(self, mode: str, env_ids: Sequence[int] | None = None) -> None:  # noqa: D102
        pass

    @property
    def unwrapped(self) -> DummyEnv:  # noqa: D102
        return self


def _make_cfg() -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "log_interval": 1,
        "num_seed_steps_per_env": 1,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "SquashedTanhGaussianDistribution", "init_noise_std": 1.0},
        },
        "critic": {"class_name": "FuseModel", "embedding_dims": [32], "hidden_dims": [32, 32]},
        "algorithm": {
            "class_name": "SAC",
            "replay_buffer_size": 512,
            "mini_batch_size": 16,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "auto_alpha": True,
            "alpha": 0.1,
            "tau": 0.1,
            "gamma": 0.99,
            "n_steps": 1,
            "policy_frequency": 1,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


class TestOffPolicyRunnerSAC:
    """The OffPolicyRunner should construct and drive SAC through its generic (non-FbCpr) loop."""

    def test_construct_dispatches_to_sac(self) -> None:
        """The runner builds a SAC algorithm (no expert buffer -> generic loop will be used)."""
        runner = OffPolicyRunner(DummyEnv(), _make_cfg(), log_dir=None, device="cpu")
        assert isinstance(runner.alg, SAC)
        assert not hasattr(runner.alg, "expert_buffer")

    def test_learn_runs_and_fills_buffer(self) -> None:
        """learn() runs the generic off-policy loop: fills the replay buffer and advances the iteration."""
        runner = OffPolicyRunner(DummyEnv(), _make_cfg(), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2  # ran iters 0,1,2
        assert len(runner.alg.replay_buffer) == 3 * 4 * NUM_ENVS  # 3 iters * 4 steps * num_envs

    def test_random_initial_episode_lengths(self) -> None:
        """init_at_random_ep_len staggers the episodes, which otherwise time out together for the whole run."""
        torch.manual_seed(0)
        env = DummyEnv()
        OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu").learn(0, init_at_random_ep_len=True)
        assert env.episode_length_buf.unique().numel() > 1
        env = DummyEnv()
        OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu").learn(0)
        assert (env.episode_length_buf == 0).all()

    def test_learn_updates_policy(self) -> None:
        """After several iterations past the seed phase, the actor parameters have changed (learning occurred)."""
        runner = OffPolicyRunner(DummyEnv(), _make_cfg(), log_dir=None, device="cpu")
        before = [p.detach().clone() for p in runner.alg.actor_parameters]
        runner.learn(num_learning_iterations=6)
        changed = any(not torch.allclose(b, a) for b, a in zip(before, runner.alg.actor_parameters, strict=True))
        assert changed

    def test_export_policy_to_jit(self, tmp_path: Path) -> None:
        """SAC has no external obs normalizer; export must fall back to the in-model path and produce a file."""
        import os

        runner = OffPolicyRunner(DummyEnv(), _make_cfg(), log_dir=None, device="cpu")
        runner.export_policy_to_jit(str(tmp_path))
        assert os.path.isfile(tmp_path / "policy.pt")
