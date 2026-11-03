# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the SAC algorithm (dummy VecEnv, no Isaac Sim)."""

import copy
import torch
from tensordict import TensorDict

from robot_rl.algorithms import SAC

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8


class _DummyVecEnv:
    """A minimal VecEnv: random dynamics with occasional terminations/timeouts, exposing time_outs(_obs)."""

    def __init__(self, num_envs: int = NUM_ENVS, num_actions: int = ACT_DIM, obs_dim: int = OBS_DIM) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.cfg = None

    def _obs(self) -> TensorDict:
        return TensorDict({"policy": torch.randn(self.num_envs, self.obs_dim)}, batch_size=self.num_envs)

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple:
        next_obs = self._obs()
        rewards = torch.randn(self.num_envs)
        terminated = torch.rand(self.num_envs) < 0.05
        timed_out = torch.rand(self.num_envs) < 0.05
        dones = terminated | timed_out
        extras = {
            "time_outs": timed_out,
            "time_outs_obs": TensorDict({"policy": torch.randn(self.num_envs, self.obs_dim)}, batch_size=self.num_envs),
        }
        return next_obs, rewards, dones, extras


def _make_cfg(auto_alpha: bool = True) -> dict:
    return {
        "num_steps_per_env": 4,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "SquashedTanhGaussianDistribution", "init_noise_std": 1.0},
        },
        "critic": {
            "class_name": "FuseModel",
            "embedding_dims": [32],
            "hidden_dims": [32, 32],
        },
        "algorithm": {
            "class_name": "SAC",
            "replay_buffer_size": 256,
            "mini_batch_size": 16,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "alpha_learning_rate": 1e-3,
            "auto_alpha": auto_alpha,
            "alpha": 0.1,
            "tau": 0.1,
            "gamma": 0.99,
            "target_entropy_scale": 1.0,
            "max_grad_norm": 1.0,
            "policy_frequency": 1,
            "n_steps": 1,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _build(**algorithm: object) -> tuple[SAC, _DummyVecEnv]:
    env = _DummyVecEnv()
    obs = env.get_observations()
    cfg = _make_cfg()
    cfg["algorithm"].update(algorithm)
    alg = SAC.construct_algorithm(obs, env, cfg, device="cpu")
    return alg, env


def _collect(alg: SAC, env: _DummyVecEnv, steps: int) -> None:
    obs = env.get_observations()
    for _ in range(steps):
        actions = alg.act(obs)
        next_obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(next_obs, rewards, dones, extras)
        obs = next_obs


class TestSAC:
    """End-to-end CPU behavior of the SAC algorithm."""

    def test_construct(self) -> None:
        """construct_algorithm builds actor, twin critics, targets, and a keep_terminal buffer."""
        alg, _ = _build()
        assert alg.actor is not None and alg.critic_1 is not None and alg.critic_2 is not None
        assert alg.replay_buffer.keep_terminal is True
        assert alg.target_entropy == -1.0 * ACT_DIM

    def test_backup_entropy_switch(self) -> None:
        """The critic target charges alpha * logp by default and nothing with backup_entropy off; both update."""
        alg, env = _build()
        assert float(alg._backup_alpha()) == float(alg.log_alpha.exp())
        off, off_env = _build(backup_entropy=False)
        assert off._backup_alpha() == 0.0
        _collect(off, off_env, steps=8)
        losses = off.update()
        assert all(torch.isfinite(torch.tensor(losses[k])) for k in ("critic_1", "critic_2", "actor", "alpha"))

    def test_act_shape_and_bounds(self) -> None:
        """act() returns a bounded [num_envs, num_actions] action."""
        alg, env = _build()
        actions = alg.act(env.get_observations())
        assert actions.shape == (NUM_ENVS, ACT_DIM)
        assert (actions >= -1.0).all() and (actions <= 1.0).all()

    def test_collect_fills_buffer(self) -> None:
        """Rolling out fills the shared replay buffer (terminals kept)."""
        alg, env = _build()
        _collect(alg, env, steps=10)
        assert len(alg.replay_buffer) == 10 * NUM_ENVS

    def test_update_returns_finite_losses(self) -> None:
        """update() returns the expected loss keys, all finite."""
        alg, env = _build()
        _collect(alg, env, steps=8)
        losses = alg.update()
        for key in ("critic_1", "critic_2", "actor", "alpha", "alpha_value"):
            assert key in losses
            assert torch.isfinite(torch.tensor(losses[key]))

    def test_update_changes_actor_and_targets_track(self) -> None:
        """A few updates change the actor params and move the target critics toward the online critics."""
        alg, env = _build()
        _collect(alg, env, steps=12)
        actor_before = copy.deepcopy([p.detach().clone() for p in alg.actor_parameters])
        target_before = copy.deepcopy([p.detach().clone() for p in alg.critic_1_target.target.parameters()])
        for _ in range(3):
            alg.update()
        actor_changed = any(not torch.allclose(b, a) for b, a in zip(actor_before, alg.actor_parameters, strict=True))
        target_moved = any(
            not torch.allclose(b, t)
            for b, t in zip(target_before, alg.critic_1_target.target.parameters(), strict=True)
        )
        assert actor_changed, "actor parameters should update"
        assert target_moved, "target critic should track the online critic via soft update"

    def test_save_load_roundtrip(self) -> None:
        """save()/load() round-trips actor/critic/alpha and re-syncs targets."""
        alg, env = _build()
        _collect(alg, env, steps=8)
        alg.update()
        state = alg.save()
        alg2, _ = _build()
        alg2.load(state)
        for p1, p2 in zip(alg.actor.parameters(), alg2.actor.parameters(), strict=True):
            assert torch.allclose(p1, p2)
        assert abs(alg.alpha - alg2.alpha) < 1e-6

    def test_nstep_update(self) -> None:
        """SAC trains end-to-end with n-step returns (per-sample effective horizon)."""
        env = _DummyVecEnv()
        cfg = _make_cfg()
        cfg["algorithm"]["n_steps"] = 3
        alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
        assert alg.replay_buffer.n_steps == 3
        _collect(alg, env, steps=12)
        losses = alg.update()
        for key in ("critic_1", "critic_2", "actor", "alpha_value"):
            assert torch.isfinite(torch.tensor(losses[key]))

    def test_fixed_alpha(self) -> None:
        """With auto_alpha=False the temperature is fixed and has no optimizer."""
        env = _DummyVecEnv()
        alg = SAC.construct_algorithm(env.get_observations(), env, _make_cfg(auto_alpha=False), device="cpu")
        assert alg.alpha_optimizer is None
        _collect(alg, env, steps=8)
        losses = alg.update()
        assert losses["alpha"] == 0.0
        assert abs(alg.alpha - 0.1) < 1e-6  # unchanged
