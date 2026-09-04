# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-expert style discriminators and the multi-stream PPO plumbing behind them."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

from robot_rl.algorithms.ppo import PPO
from robot_rl.extensions.style import StyleDiscriminators, resolve_style_config
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage

NUM_ENVS = 16
NUM_STEPS = 8
OBS_DIM = 6
STYLE_DIM = 5
NUM_ACTIONS = 3


def _write_expert(path: Path, center: float, num: int = 2000) -> str:
    torch.save({"states": torch.randn(num, STYLE_DIM) * 0.3 + center}, path)
    return str(path)


def _obs(center: float = 0.0, num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(num_envs, OBS_DIM), "style": torch.randn(num_envs, STYLE_DIM) * 0.3 + center},
        batch_size=[num_envs],
    )


def _make_style(tmp_path: Path, **kwargs: object) -> StyleDiscriminators:
    experts = [
        {"name": "a", "data_path": _write_expert(tmp_path / "a.pt", 2.0), "weight": 1.0},
        {"name": "b", "data_path": _write_expert(tmp_path / "b.pt", -2.0), "weight": 0.5},
    ]
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "learning_rate": 1e-3, "batch_size": 64}
    defaults.update(kwargs)
    return StyleDiscriminators(STYLE_DIM, {"style": ["style"]}, experts, **defaults)  # type: ignore[arg-type]


class TestDiscriminators:
    """Tests for the discriminator module on its own."""

    def test_rewards_shape_and_finite(self, tmp_path: Path) -> None:
        """One reward column per expert, all finite."""
        style = _make_style(tmp_path)
        style.train()
        rewards = style.compute_rewards(_obs())
        assert rewards.shape == (NUM_ENVS, 2)
        assert torch.isfinite(rewards).all()

    def test_training_separates_experts(self, tmp_path: Path) -> None:
        """After training, each discriminator scores its own expert's states above the other's."""
        style = _make_style(tmp_path, reward_normalization=False, state_normalization=False)
        style.train()
        policy_states = torch.randn(512, STYLE_DIM)  # far from both experts
        for _ in range(30):
            losses = style.update(policy_states)
        assert set(losses) == {f"style/{n}_{k}" for n in ("a", "b") for k in ("expert", "policy", "grad_penalty")}
        with torch.no_grad():
            a_on_a = style.discriminators[0](style.expert_states[0]).mean()
            a_on_b = style.discriminators[0](style.expert_states[1]).mean()
            b_on_b = style.discriminators[1](style.expert_states[1]).mean()
            b_on_a = style.discriminators[1](style.expert_states[0]).mean()
        assert a_on_a > a_on_b
        assert b_on_b > b_on_a

    def test_gate_zeroes_unclaimed_transitions(self, tmp_path: Path) -> None:
        """An unreachable gate threshold zeroes every style reward."""
        style = _make_style(tmp_path, reward_normalization=False, state_normalization=False, gate_threshold=1e6)
        style.train()
        assert torch.all(style.compute_rewards(_obs()) == 0.0)

    def test_expert_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        """Expert data whose width differs from the style state is rejected at construction."""
        bad = tmp_path / "bad.pt"
        torch.save({"states": torch.randn(10, STYLE_DIM + 1)}, bad)
        try:
            StyleDiscriminators(STYLE_DIM, {"style": ["style"]}, [{"name": "x", "data_path": str(bad)}])
        except ValueError:
            return
        raise AssertionError("mismatched expert data must raise")

    def test_weight_schedule(self, tmp_path: Path) -> None:
        """The linear schedule anneals the global weight to its final value."""
        style = _make_style(
            tmp_path,
            weight=1.0,
            weight_schedule={"mode": "linear", "initial_step": 0, "final_step": 10, "final_value": 0.0},
        )
        style.train()
        obs = _obs()
        for _ in range(5):
            style.compute_rewards(obs)
        assert 0.0 < style.weight < 1.0
        for _ in range(10):
            style.compute_rewards(obs)
        assert style.weight == 0.0

    def test_resolve_config(self) -> None:
        """The resolver fills in the style dimension or disables the extension."""
        obs = _obs()
        cfg = resolve_style_config({"style_cfg": {"experts": []}}, obs, {"style": ["style"]})
        assert cfg["style_cfg"]["num_states"] == STYLE_DIM
        assert resolve_style_config({}, obs, {})["style_cfg"] is None


def _build_style_ppo(tmp_path: Path) -> tuple[PPO, TensorDict]:
    obs = _obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"], "style": ["style"]}
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    actor = MLPModel(obs, obs_groups, "actor", NUM_ACTIONS, hidden_dims=[32], activation="elu", distribution_cfg=dist)
    critic = MLPModel(obs, obs_groups, "critic", 3, hidden_dims=[32], activation="elu")
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_reward_streams=3)
    style_cfg = {
        "num_states": STYLE_DIM,
        "obs_groups": obs_groups,
        "experts": [
            {"name": "a", "data_path": _write_expert(tmp_path / "a.pt", 2.0), "weight": 1.0},
            {"name": "b", "data_path": _write_expert(tmp_path / "b.pt", -2.0), "weight": 0.5},
        ],
        "hidden_dims": [32],
        "batch_size": 32,
    }
    ppo = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
        style_cfg=style_cfg,
    )
    return ppo, obs


class TestMultiStreamPPO:
    """Tests for PPO with one reward stream and value head per expert."""

    def test_rollout_update_and_checkpoint(self, tmp_path: Path) -> None:
        """Streams flow through storage, returns, the update and the checkpoint round trip."""
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.train_mode()
        assert ppo.num_reward_streams == 3
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.uint8)
            ppo.process_env_step(_obs(), rewards, dones, {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)})
        st = ppo.storage
        assert st.rewards.shape == (NUM_STEPS, NUM_ENVS, 3)
        assert st.values.shape == (NUM_STEPS, NUM_ENVS, 3)
        ppo.compute_returns(obs)
        assert st.returns.shape == (NUM_STEPS, NUM_ENVS, 3)
        assert st.advantages.shape == (NUM_STEPS, NUM_ENVS, 1)
        loss_dict = ppo.update()
        assert "style/a_expert" in loss_dict and "Style/b_reward" in loss_dict
        saved = ppo.save()
        assert "style_state_dict" in saved
        ppo2, _ = _build_style_ppo(tmp_path)
        ppo2.load(saved, None, strict=True)
        for p1, p2 in zip(ppo.style.discriminators.parameters(), ppo2.style.discriminators.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_stream_advantages_are_standardized_and_weighted(self, tmp_path: Path) -> None:
        """A stream with zero weight must not move the combined advantage."""
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.style.stream_weights[:] = torch.tensor([0.0, 0.0])
        ppo.train_mode()
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(_obs(), torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.uint8), {})
        ppo.compute_returns(obs)
        task_adv = ppo.storage.returns[..., :1] - ppo.storage.values[..., :1]
        expected = (task_adv - task_adv.mean()) / (task_adv.std() + 1e-8)
        assert torch.allclose(ppo.storage.advantages, expected, atol=1e-5)
