# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locks the OffPolicyRunner's URL-path loop cadence (seed phase, update gate, eval interval).

Uses a stub URL algorithm (has ``expert_buffer``) so the FbCpr control flow -- ``num_steps_per_env``
env steps per iteration, ``num_agent_updates`` updates every iteration after the seed phase, eval every
``eval_interval`` iterations -- is pinned without needing Isaac Sim or motion data.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.runners import OffPolicyRunner

NUM_ENVS, OBS_DIM, NUM_ACTIONS = 4, 6, 3


class DummyUrlEnv:
    """Minimal URL VecEnv: adds reset/set_expert_buffer/train_mode over the plain dummy env."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.device = device
        self.cfg = {}
        self.expert_buffer = None

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict({"policy": torch.randn(self.num_envs, OBS_DIM)}, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs)
        dones = (torch.rand(self.num_envs) < 0.05).float()
        return obs, rewards, dones, {"time_outs": torch.zeros(self.num_envs)}

    def set_expert_buffer(self, buffer: Any) -> None:  # noqa: D102
        self.expert_buffer = buffer

    def train_mode(self) -> None:  # noqa: D102
        pass

    @property
    def unwrapped(self) -> DummyUrlEnv:  # noqa: D102
        return self


class FakeUrlAlg:
    """Stub URL algorithm: records the runner's call sequence; has an ``expert_buffer`` (URL dispatch)."""

    def __init__(self, num_actions: int, device: str) -> None:  # noqa: D107
        self.expert_buffer = object()
        self.device = device
        self.num_actions = num_actions
        self.calls: list[str] = []

    @staticmethod
    def construct_algorithm(  # noqa: D102
        obs: TensorDict, env: DummyUrlEnv, cfg: dict, device: str, inference: bool = False
    ) -> FakeUrlAlg:
        return FakeUrlAlg(env.num_actions, device)

    def act(self, obs: TensorDict) -> torch.Tensor:  # noqa: D102
        self.calls.append("act")
        return torch.zeros(obs.batch_size[0], self.num_actions)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:  # noqa: D102
        self.calls.append("step")

    def compute_gammas(self) -> None:  # noqa: D102
        self.calls.append("gammas")

    def update(self) -> tuple[dict, dict]:  # noqa: D102
        self.calls.append("update")
        return {"loss": 0.0}, {}

    def eval(self, env: DummyUrlEnv) -> list[dict]:  # noqa: D102
        self.calls.append("eval")
        return [{"emd": torch.tensor(0.0)}]

    def reset_rollout_state(self) -> None:  # noqa: D102
        self.calls.append("reset_rollout")

    def train_mode(self) -> None:  # noqa: D102
        pass


def _make_cfg() -> dict:
    return {
        "num_steps_per_env": 2,
        "num_seed_steps_per_env": 2,
        "log_interval": 1,
        "save_interval": 100,
        "clip_actions": 1.0,
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"]},
        "algorithm": {
            "class_name": FakeUrlAlg,
            "rnd_cfg": None,
            "num_agent_updates": 3,
            "eval_interval": 4,
        },
    }


class TestOffPolicyRunnerUrlCadence:
    """The unified loop must reproduce the FbCpr cadence for URL algorithms."""

    def test_loop_cadence(self) -> None:
        """8 iterations: num_steps_per_env acts/iter; updates every iteration past the seed phase."""
        env = DummyUrlEnv()
        runner = OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=8)
        calls = runner.alg.calls

        # num_steps_per_env (2) env steps per iteration
        assert calls.count("act") == 16
        assert calls.count("step") == 16
        # Update gate: it > start_it + num_seed_steps_per_env (2) -> iterations 3..7 -> 5 x num_agent_updates (3)
        assert calls.count("gammas") == 5
        assert calls.count("update") == 15
        # Eval every eval_interval (4) iterations -> it 0 and 4; env reset follows each eval
        assert calls.count("eval") == 2
        assert calls.count("reset_rollout") == 2
        # Expert buffer was attached to the env before training
        assert env.expert_buffer is runner.alg.expert_buffer

    def test_updates_follow_collection_within_iteration(self) -> None:
        """Within an update iteration the order is act/step -> gammas -> updates (external loop shape)."""
        runner = OffPolicyRunner(DummyUrlEnv(), _make_cfg(), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=5)
        calls = runner.alg.calls
        first_update = calls.index("gammas")
        assert calls[first_update - 1] == "step"  # collection precedes the update phase
        assert calls[first_update + 1 : first_update + 4] == ["update", "update", "update"]


class TestCheckpointCallbacks:
    """Checkpoint callbacks fire at every save, feed their outputs to the algorithm and can reset the env."""

    def _runner(self, monkeypatch: Any, save_interval: int) -> OffPolicyRunner:
        cfg = _make_cfg()
        cfg["save_interval"] = save_interval
        runner = OffPolicyRunner(DummyUrlEnv(), cfg, log_dir=None, device="cpu")
        # a stand-in writer marks this as the main rank without touching disk
        monkeypatch.setattr(runner, "save", lambda path: None)
        monkeypatch.setattr("robot_rl.runners.off_policy_runner.demote_old_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(runner.logger, "init_logging_writer", lambda: None)
        monkeypatch.setattr(runner.logger, "log", lambda **k: None)
        monkeypatch.setattr(runner.logger, "stop_logging_writer", lambda: None)
        runner.logger.writer = object()
        return runner

    def test_fires_at_saves_and_applies_outputs(self, monkeypatch: Any) -> None:
        """Saves at it 0 and 4 plus the final it 7; outputs reach apply_eval_outputs; built-in eval is off."""
        runner = self._runner(monkeypatch, save_interval=4)
        runner.builtin_eval = False
        applied: list[dict] = []
        runner.alg.apply_eval_outputs = applied.append
        seen: list[int] = []

        def callback(r: OffPolicyRunner, path: str, it: int) -> dict:
            seen.append(it)
            return {"motion_priorities": torch.ones(3)}

        runner.add_checkpoint_callback(callback)
        runner.learn(num_learning_iterations=8)
        assert seen == [0, 4, 7]
        assert len(applied) == 3 and torch.equal(applied[0]["motion_priorities"], torch.ones(3))
        assert "eval" not in runner.alg.calls

    def test_env_used_resets_rollout(self, monkeypatch: Any) -> None:
        """A callback that stepped the training env makes the runner reset before collecting again."""
        runner = self._runner(monkeypatch, save_interval=4)
        runner.builtin_eval = False
        runner.add_checkpoint_callback(lambda r, path, it: {"env_used": True})
        runner.learn(num_learning_iterations=8)
        # it 0 and 4 reset; the final save has no collection after it
        assert runner.alg.calls.count("reset_rollout") == 2
