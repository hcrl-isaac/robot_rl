# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-expert style discriminators and the multi-stream PPO plumbing behind them."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.algorithms.ppo import PPO
from robot_rl.extensions.style import StyleDiscriminators, resolve_style_config
from robot_rl.extensions.symmetry import Symmetry
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage
from tests.extensions.test_symmetry import _negate_augmentation

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
        assert set(losses) == {f"Style/{n}_{k}" for n in ("a", "b") for k in ("expert", "policy", "grad_penalty")}
        with torch.no_grad():
            a_on_a = style.discriminators[0](style.expert_states[0]).mean()
            a_on_b = style.discriminators[0](style.expert_states[1]).mean()
            b_on_b = style.discriminators[1](style.expert_states[1]).mean()
            b_on_a = style.discriminators[1](style.expert_states[0]).mean()
        assert a_on_a > a_on_b
        assert b_on_b > b_on_a

    def test_rewards_are_bounded_and_favor_expert_states(self, tmp_path: Path) -> None:
        """Unnormalized rewards stay in [0, 1], and a trained discriminator pays its expert's states more."""
        style = _make_style(tmp_path, reward_normalization=False, state_normalization=False)
        style.train()
        for _ in range(30):
            style.update(torch.randn(512, STYLE_DIM))
        on_a = style.compute_rewards(_obs(center=2.0))
        off = style.compute_rewards(_obs(center=0.0))
        assert torch.all((on_a >= 0.0) & (on_a <= 1.0)) and torch.all((off >= 0.0) & (off <= 1.0))
        assert on_a[:, 0].mean() > off[:, 0].mean()

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
        assert "Style/a_expert" in loss_dict and "Style/b_reward" in loss_dict
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


class TestGating:
    """Tests for a stream that pays only inside its gate."""

    @staticmethod
    def _gated_style(tmp_path: Path) -> StyleDiscriminators:
        experts = [
            {"name": "a", "data_path": _write_expert(tmp_path / "ga.pt", 2.0), "gate_group": "flight"},
            {"name": "b", "data_path": _write_expert(tmp_path / "gb.pt", -2.0)},
        ]
        return StyleDiscriminators(STYLE_DIM, {"style": ["style"]}, experts, hidden_dims=[32, 32], batch_size=64)

    @staticmethod
    def _gated_obs(open_rows: int) -> TensorDict:
        obs = _obs()
        gate = torch.zeros(NUM_ENVS, 1)
        gate[:open_rows] = 1.0
        obs["flight"] = gate
        return obs

    def test_reward_is_zero_outside_the_gate(self, tmp_path: Path) -> None:
        """The gated stream pays only on rows whose gate is open; the ungated one pays everywhere."""
        style = self._gated_style(tmp_path)
        style.train()
        rewards = style.compute_rewards(self._gated_obs(open_rows=4))
        assert (rewards[4:, 0] == 0).all()
        assert (rewards[:, 1] != 0).any()

    def test_gate_masks_select_the_open_rows(self, tmp_path: Path) -> None:
        """``gate_masks`` returns the open rows for a gated expert and None for an ungated one."""
        style = self._gated_style(tmp_path)
        masks = style.gate_masks(self._gated_obs(open_rows=4))
        assert masks[0] is not None and int(masks[0].sum()) == 4
        assert masks[1] is None

    def test_update_skips_an_expert_with_no_gated_rows(self, tmp_path: Path) -> None:
        """A closed gate leaves its discriminator untrained rather than training it on states it never pays on."""
        style = self._gated_style(tmp_path)
        style.train()
        before = style.discriminators[0][0].weight.clone()
        closed = [torch.zeros(64, dtype=torch.bool), None]
        losses = style.update(torch.randn(64, STYLE_DIM), closed)
        assert not any(k.startswith("Style/a_") for k in losses)
        assert torch.equal(style.discriminators[0][0].weight, before)

    def test_schedule_clock_survives_a_round_trip(self, tmp_path: Path) -> None:
        """The weight ramp continues after a reload instead of restarting."""
        style = self._gated_style(tmp_path)
        for _ in range(5):
            style.compute_rewards(self._gated_obs(open_rows=4))
        restored = self._gated_style(tmp_path)
        restored.load_state_dict(style.state_dict())
        assert restored.reward_steps == style.reward_steps == 5


class TestActorWarmup:
    """Tests for the warm-up that trains only the value heads."""

    @staticmethod
    def _warmup_ppo(tmp_path: Path, warmup: int) -> tuple[PPO, TensorDict]:
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.actor_warmup_iters = warmup
        return ppo, obs

    @staticmethod
    def _rollout(ppo: PPO, obs: TensorDict) -> None:
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(
                _obs(),
                torch.randn(NUM_ENVS),
                torch.zeros(NUM_ENVS, dtype=torch.uint8),
                {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)},
            )
        ppo.compute_returns(obs)

    def test_actor_is_frozen_during_warmup(self, tmp_path: Path) -> None:
        """The value, mirror and encoder losses do not move the actor while it is warming up."""
        ppo, obs = self._warmup_ppo(tmp_path, warmup=10)
        ppo.train_mode()
        before = [p.clone() for p in ppo.actor.parameters()]
        self._rollout(ppo, obs)
        ppo.update()
        assert all(torch.equal(a, b) for a, b in zip(before, ppo.actor.parameters(), strict=True))

    def test_actor_trains_after_warmup(self, tmp_path: Path) -> None:
        """Past the warm-up the actor moves again."""
        ppo, obs = self._warmup_ppo(tmp_path, warmup=0)
        ppo.train_mode()
        before = [p.clone() for p in ppo.actor.parameters()]
        self._rollout(ppo, obs)
        ppo.update()
        assert any(not torch.equal(a, b) for a, b in zip(before, ppo.actor.parameters(), strict=True))

    def test_warmup_counter_survives_a_resume(self, tmp_path: Path) -> None:
        """A resume continues the warm-up instead of freezing the actor again."""
        ppo, obs = self._warmup_ppo(tmp_path, warmup=10)
        ppo.train_mode()
        self._rollout(ppo, obs)
        ppo.update()
        restored, _ = self._warmup_ppo(tmp_path, warmup=10)
        restored.load(ppo.save(), {"actor": True, "critic": True, "iteration": True}, strict=True)
        assert restored.num_updates_done == ppo.num_updates_done == 1


class TestActInference:
    """Tests for the storage-free action path used by dataset collection."""

    def test_act_inference_leaves_storage_untouched(self, tmp_path: Path) -> None:
        """Collecting actions does not record a transition."""
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.eval_mode()
        actions = ppo.act_inference(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert ppo.storage.step == 0

    def test_act_inference_sampling_differs_from_the_mean(self, tmp_path: Path) -> None:
        """``stochastic`` samples the policy, which a rollout's states carry."""
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.eval_mode()
        assert not torch.equal(ppo.act_inference(obs, stochastic=True), ppo.act_inference(obs))


class TestClosedGateWeight:
    """Tests that a stream whose gate never opened carries no weight."""

    @staticmethod
    def _gated_ppo(tmp_path: Path) -> tuple[PPO, TensorDict]:
        obs = _obs()
        obs["flight"] = torch.zeros(NUM_ENVS, 1)  # the gate never opens
        obs_groups = {"actor": ["policy"], "critic": ["policy"], "style": ["style"]}
        dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        actor = MLPModel(
            obs, obs_groups, "actor", NUM_ACTIONS, hidden_dims=[32], activation="elu", distribution_cfg=dist
        )
        critic = MLPModel(obs, obs_groups, "critic", 2, hidden_dims=[32], activation="elu")
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_reward_streams=2)
        style_cfg = {
            "num_states": STYLE_DIM,
            "obs_groups": obs_groups,
            "experts": [{"name": "a", "data_path": _write_expert(tmp_path / "ga.pt", 2.0), "gate_group": "flight"}],
            "hidden_dims": [32],
            "batch_size": 32,
        }
        ppo = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2, style_cfg=style_cfg)
        return ppo, obs

    def test_closed_gate_stream_is_dropped_despite_time_outs(self, tmp_path: Path) -> None:
        """A time-out bootstrap lands in every stream, so the weight must come from the gate, not the rewards."""
        ppo, obs = self._gated_ppo(tmp_path)
        ppo.train_mode()
        for step in range(NUM_STEPS):
            ppo.act(obs)
            time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
            time_outs[0] = step == 0  # one time-out in the rollout
            ppo.process_env_step(
                obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.uint8), {"time_outs": time_outs}
            )
        assert ppo.storage.rewards[..., 1].abs().max() > 0  # the bootstrap did land in the style stream
        ppo.compute_returns(obs)
        task_only = ppo.storage.returns[..., :1] - ppo.storage.values[..., :1]
        task_only = (task_only - task_only.mean()) / (task_only.std() + 1e-8)
        assert torch.allclose(ppo.storage.advantages, task_only, atol=1e-4)


class TestWarmupIsolation:
    """Tests that nothing else moves while the actor is warming up."""

    @staticmethod
    def _ppo(tmp_path: Path, warmup: int) -> tuple[PPO, TensorDict]:
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.actor_warmup_iters = warmup
        ppo.schedule = "adaptive"
        ppo.desired_kl = 0.01
        ppo.adaptive_lr_once_per_iteration = False
        return ppo, obs

    @staticmethod
    def _rollout(ppo: PPO, obs: TensorDict) -> None:
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(
                _obs(),
                torch.randn(NUM_ENVS),
                torch.zeros(NUM_ENVS, dtype=torch.uint8),
                {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)},
            )
        ppo.compute_returns(obs)

    def test_learning_rate_is_not_adapted_during_warmup(self, tmp_path: Path) -> None:
        """A frozen actor reports no KL, which would otherwise ratchet the rate to its maximum."""
        ppo, obs = self._ppo(tmp_path, warmup=0)
        ppo.symmetry = Symmetry(
            env=None, data_augmentation_func=_negate_augmentation, use_mirror_loss=True, mirror_loss_coeff=1.0
        )
        ppo.train_mode()
        self._rollout(ppo, obs)
        ppo.update()  # one real update, so Adam carries momentum into the warm-up
        ppo.actor_warmup_iters = 10
        ppo.num_updates_done = 0
        before = ppo.learning_rate
        self._rollout(ppo, obs)
        # a frozen actor reports no KL on its own, so shift the stored distribution the KL is measured against
        for param in ppo.storage.distribution_params:
            param += 1.0
        ppo.update()
        assert ppo.learning_rate == before

    def test_actor_stays_frozen_under_a_loss_that_reaches_it(self, tmp_path: Path) -> None:
        """The mirror loss reaches the actor during warm-up, and a zeroed grad is still a step under Adam."""
        ppo, obs = self._ppo(tmp_path, warmup=10)
        ppo.symmetry = Symmetry(
            env=None,
            data_augmentation_func=_negate_augmentation,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
        )
        ppo.train_mode()
        ppo.actor_warmup_iters = 0  # one real update first, so Adam carries momentum into the warm-up
        self._rollout(ppo, obs)
        ppo.update()
        ppo.actor_warmup_iters = 10
        ppo.num_updates_done = 0
        before = [p.clone() for p in ppo.actor.parameters()]
        self._rollout(ppo, obs)
        ppo.update()
        assert all(torch.equal(a, b) for a, b in zip(before, ppo.actor.parameters(), strict=True))


class TestPaidAtRewardTime:
    """Tests that the stream weight follows what each stream actually paid on."""

    @staticmethod
    def _step(ppo: PPO, next_obs: TensorDict) -> None:
        ppo.process_env_step(
            next_obs,
            torch.randn(NUM_ENVS),
            torch.zeros(NUM_ENVS, dtype=torch.uint8),
            {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)},
        )

    @staticmethod
    def _obs_with_gate(open_gate: bool) -> TensorDict:
        obs = _obs()
        obs["flight"] = torch.full((NUM_ENVS, 1), float(open_gate))
        return obs

    def test_gate_open_only_on_the_acted_obs_pays_nothing(self, tmp_path: Path) -> None:
        """A gate open only where the action was taken never paid, so the stream carries no weight."""
        ppo, _ = TestClosedGateWeight._gated_ppo(tmp_path)
        ppo.train_mode()
        for step in range(NUM_STEPS):
            ppo.act(self._obs_with_gate(step == 0))  # open on the first acted-on obs only
            self._step(ppo, self._obs_with_gate(False))  # never open where the reward is paid
        ppo.compute_returns(self._obs_with_gate(False))
        task_only = ppo.storage.returns[..., :1] - ppo.storage.values[..., :1]
        task_only = (task_only - task_only.mean()) / (task_only.std() + 1e-8)
        assert torch.allclose(ppo.storage.advantages, task_only, atol=1e-4)

    def test_gate_open_only_on_the_next_obs_is_counted(self, tmp_path: Path) -> None:
        """A gate open only where the reward is paid did pay, so the stream keeps its weight."""
        ppo, _ = TestClosedGateWeight._gated_ppo(tmp_path)
        ppo.train_mode()
        for step in range(NUM_STEPS):
            ppo.act(self._obs_with_gate(False))
            self._step(ppo, self._obs_with_gate(step == NUM_STEPS - 1))  # open on the last next-obs only
        ppo.compute_returns(self._obs_with_gate(False))
        task_only = ppo.storage.returns[..., :1] - ppo.storage.values[..., :1]
        task_only = (task_only - task_only.mean()) / (task_only.std() + 1e-8)
        assert not torch.allclose(ppo.storage.advantages, task_only, atol=1e-4)


class TestSharedMemory:
    """Tests the inference path's contract for a policy carrying a shared memory."""

    def test_act_inference_raises_for_a_shared_memory_policy(self, tmp_path: Path) -> None:
        """The hidden state is not carried here, so a memory policy must be refused rather than mis-stepped."""
        ppo, obs = _build_style_ppo(tmp_path)
        ppo.memory = object()  # any memory at all
        with pytest.raises(NotImplementedError):
            ppo.act_inference(obs)


class TestCheckpointCompat:
    """Tests that a checkpoint written before the counter was renamed still loads."""

    def test_old_counter_key_restores_the_schedule_clock(self, tmp_path: Path) -> None:
        """A pre-rename checkpoint carries ``update_counter``; the ramp must continue, not restart."""
        style = TestGating._gated_style(tmp_path)
        state = style.state_dict()
        state["update_counter"] = state.pop("reward_steps") + 7
        restored = TestGating._gated_style(tmp_path)
        restored.load_state_dict(state)
        assert restored.reward_steps == 7
