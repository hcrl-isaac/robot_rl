# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for Distillation-SAC: a CNN+GRU student under SAC plus cloning of a privileged MLP teacher."""

import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import SAC, DistillationSAC

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 4, 8
IMG = (1, 8, 12)


class _DummyDistillVecEnv:
    """A VecEnv with the student's 1D and image groups, the teacher's privileged group and a critic group."""

    def __init__(self, done_prob: float = 0.1) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = ACT_DIM
        self.done_prob = done_prob
        self.cfg = None

    def _obs(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM),
                "image": torch.rand(self.num_envs, *IMG),
                "teacher": torch.randn(self.num_envs, OBS_DIM + 2),
                "critic": torch.randn(self.num_envs, OBS_DIM + 3),
            },
            batch_size=self.num_envs,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple:
        dones = torch.rand(self.num_envs) < self.done_prob
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool), "time_outs_obs": None}
        return self._obs(), torch.randn(self.num_envs), dones, extras


_VMF = {"class_name": "VonMisesFisherDistribution", "init_std": 0.5}
_CRITIC = {"class_name": "FuseModel", "embedding_dims": [16], "hidden_dims": [32], "obs_normalization": True}
_SAC_ALG = {
    "replay_buffer_size": 512,
    "mini_batch_size": 16,
    "num_learning_epochs": 1,
    "num_mini_batches": 2,
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "alpha_learning_rate": 1e-3,
    "auto_alpha": True,
    "alpha": 0.1,
    "tau": 0.1,
    "gamma": 0.99,
    "target_entropy_scale": 1.0,
    "max_grad_norm": 1.0,
    "policy_frequency": 1,
    "n_steps": 1,
}


def _cfg(**algorithm: object) -> dict:
    return {
        "num_steps_per_env": 4,
        "obs_groups": {"student": ["policy", "image"], "teacher": ["teacher"], "critic": ["critic"]},
        "student": {
            "class_name": "CNNRNNModel",
            "hidden_dims": [32],
            "obs_normalization": True,
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "cnn_cfg": {
                "output_channels": [4, 4],
                "kernel_size": 3,
                "stride": 2,
                "activation": "elu",
                "global_pool": "avg",
            },
            "distribution_cfg": dict(_VMF),
        },
        "teacher": {
            "class_name": "MLPModel",
            "hidden_dims": [32],
            "obs_normalization": True,
            "distribution_cfg": dict(_VMF),
        },
        "critic": dict(_CRITIC),
        "algorithm": {
            "class_name": "DistillationSAC",
            **_SAC_ALG,
            "seq_len": 4,
            "burn_in": 2,
            "bc_weight": 1.0,
            "bc_weight_final": 0.1,
            "bc_anneal_updates": 4,
            **algorithm,
        },
    }


def _teacher_checkpoint(env: _DummyDistillVecEnv) -> dict:
    """Return a plain SAC checkpoint over the teacher group, as a teacher training run would save it."""
    cfg = {
        "num_steps_per_env": 4,
        "obs_groups": {"actor": ["teacher"], "critic": ["critic"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32],
            "obs_normalization": True,
            "distribution_cfg": dict(_VMF),
        },
        "critic": dict(_CRITIC),
        "algorithm": {"class_name": "SAC", **_SAC_ALG},
    }
    teacher = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
    with torch.no_grad():
        teacher.log_alpha.fill_(-3.0)
    return teacher.save()


def _build(load_teacher: bool = True, **algorithm: object) -> tuple[DistillationSAC, _DummyDistillVecEnv, dict]:
    env = _DummyDistillVecEnv()
    alg = DistillationSAC.construct_algorithm(env.get_observations(), env, _cfg(**algorithm), device="cpu")
    ckpt = _teacher_checkpoint(env)
    if load_teacher:
        assert not alg.load(ckpt)
    alg.train_mode()
    return alg, env, ckpt


def _collect(alg: DistillationSAC, env: _DummyDistillVecEnv, steps: int = 12) -> None:
    obs = env.get_observations()
    for _ in range(steps):
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)


def _params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _moved(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters(), strict=True))


class TestDistillationSAC:
    """The combined update clones the teacher, anneals the weight, gates the RL term and checkpoints all parts."""

    def test_update_reports_losses_and_moves_student_and_critics(self) -> None:
        """One update logs the cloning diagnostics and steps the student and both critics."""
        torch.manual_seed(0)
        alg, env, _ = _build()
        student_before, critic_before = _params(alg.actor), _params(alg.critic_1)
        teacher_before = _params(alg.teacher)
        _collect(alg, env)
        losses, diag = alg.update()
        for key in ("critic_1", "critic_2", "actor", "bc_weight", "rl_on"):
            assert key in losses and torch.isfinite(torch.tensor(losses[key])), key
        for key in ("Distill/behavior", "Distill/teacher_cos", "Distill/q_scale"):
            assert key in diag and torch.isfinite(torch.tensor(diag[key])), key
        assert _moved(student_before, alg.actor)
        assert _moved(critic_before, alg.critic_1)
        assert not _moved(teacher_before, alg.teacher)

    def test_refuses_to_update_without_a_teacher(self) -> None:
        """Training never starts from an unloaded teacher."""
        alg, env, _ = _build(load_teacher=False)
        _collect(alg, env)
        with pytest.raises(ValueError, match="Teacher"):
            alg.update()

    def test_bc_weight_anneals_linearly_then_holds(self) -> None:
        """The cloning weight falls linearly to its final value over the anneal window and then stays there."""
        torch.manual_seed(0)
        alg, env, _ = _build()
        weights = []
        for _ in range(6):
            _collect(alg, env, 4)
            weights.append(alg.update()[0]["bc_weight"])
        assert torch.allclose(torch.tensor(weights), torch.tensor([1.0, 0.775, 0.55, 0.325, 0.1, 0.1]))

    def test_critic_warmup_keeps_the_rl_term_out_of_the_actor(self) -> None:
        """With no cloning, the student stays put until the warm-up ends, while the critics already learn."""
        torch.manual_seed(0)
        alg, env, _ = _build(bc_weight=0.0, bc_weight_final=0.0, critic_warmup_updates=1)
        student_before, critic_before = _params(alg.actor), _params(alg.critic_1)
        _collect(alg, env)
        losses, _ = alg.update()
        assert losses["rl_on"] == 0.0
        assert not _moved(student_before, alg.actor)
        assert _moved(critic_before, alg.critic_1)
        _collect(alg, env)
        assert alg.update()[0]["rl_on"] == 1.0
        assert _moved(student_before, alg.actor)

    def test_cloning_raises_the_teacher_action_likelihood(self) -> None:
        """Cloning alone makes the teacher's action more likely under the student on the replayed steps."""
        torch.manual_seed(0)
        alg, env, _ = _build(loss_type="nll", bc_weight=1.0, bc_weight_final=1.0, critic_warmup_updates=100)
        _collect(alg, env, 16)

        def teacher_log_prob() -> float:
            with torch.no_grad():
                batch = alg.replay_buffer.sample_mini_batch("cpu")
                alg.actor.reset()
                alg.actor(batch.observations, stochastic_output=True)
                return alg.actor.get_output_log_prob(alg.teacher(batch.observations)).mean().item()

        torch.manual_seed(1)
        before = teacher_log_prob()
        for _ in range(15):
            alg.update()
        torch.manual_seed(1)
        assert teacher_log_prob() > before

    def test_cosine_cloning_aligns_the_mean_and_leaves_the_width_alone(self) -> None:
        """Cosine cloning turns the student's mean toward the teacher without touching the concentration."""
        torch.manual_seed(0)
        alg, env, _ = _build(bc_weight=1.0, bc_weight_final=1.0, critic_warmup_updates=100)
        log_kappa = alg.actor.distribution.log_kappa.detach().clone()
        _collect(alg, env, 16)
        first = alg.update()[1]["Distill/teacher_cos"]
        for _ in range(15):
            last = alg.update()[1]["Distill/teacher_cos"]
        assert last > first
        assert torch.equal(alg.actor.distribution.log_kappa.detach(), log_kappa)

    def test_teacher_checkpoint_fills_the_teacher_and_optionally_the_critics(self) -> None:
        """An RL checkpoint loads only the teacher by default, and also the critics and temperature on request."""
        torch.manual_seed(0)
        alg, _, ckpt = _build()
        assert alg.teacher_loaded
        for a, b in zip(alg.teacher.state_dict().values(), ckpt["actor_state_dict"].values(), strict=True):
            assert torch.equal(a, b)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(alg.critic_1.state_dict().values(), ckpt["critic_1_state_dict"].values(), strict=True)
        )
        warm, _, ckpt = _build(init_critic_from_teacher=True)
        for name in ("critic_1", "critic_2"):
            for a, b in zip(
                getattr(warm, name).state_dict().values(), ckpt[f"{name}_state_dict"].values(), strict=True
            ):
                assert torch.equal(a, b)
        assert warm.log_alpha.item() == pytest.approx(-3.0)

    def test_checkpoint_round_trip_resumes_every_part(self) -> None:
        """A Distillation-SAC checkpoint restores the student, teacher, critics and anneal clock."""
        torch.manual_seed(0)
        alg, env, _ = _build()
        _collect(alg, env)
        alg.update()
        saved = alg.save()
        assert "actor_state_dict" not in saved
        other, _, _ = _build(load_teacher=False)
        assert other.load(saved)
        assert other.teacher_loaded and other.num_updates == 1
        for name in ("actor", "teacher", "critic_1"):
            for a, b in zip(
                getattr(alg, name).state_dict().values(), getattr(other, name).state_dict().values(), strict=True
            ):
                assert torch.equal(a, b)

    def test_rejects_a_feedforward_student_and_n_step_returns(self) -> None:
        """Only the recurrent one-step path is supported."""
        env = _DummyDistillVecEnv()
        cfg = _cfg()
        cfg["student"] = {"class_name": "MLPModel", "hidden_dims": [32], "distribution_cfg": dict(_VMF)}
        cfg["obs_groups"]["student"] = ["policy"]
        with pytest.raises(ValueError, match="recurrent"):
            DistillationSAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
        with pytest.raises(ValueError, match="n_steps"):
            DistillationSAC.construct_algorithm(env.get_observations(), env, _cfg(n_steps=3), device="cpu")
