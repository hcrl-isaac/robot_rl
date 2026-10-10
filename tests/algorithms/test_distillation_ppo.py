"""CPU tests for Distillation-PPO: a CNN+GRU student cloning a privileged MLP teacher under a PPO term."""

import torch
from tensordict import TensorDict

from robot_rl.algorithms import DistillationPPO

NUM_ENVS = 6
OBS_DIM = 5
IMG = (1, 8, 8)
ACT_DIM = 3
STEPS = 4


class _DummyDistillVecEnv:
    """A VecEnv with the student's 1D and image groups, the teacher's privileged group and a critic group."""

    def __init__(self, done_prob: float = 0.2) -> None:
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
                "critic": torch.randn(self.num_envs, OBS_DIM + 2),
            },
            batch_size=self.num_envs,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple:
        dones = torch.rand(self.num_envs) < self.done_prob
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool)}
        return self._obs(), torch.randn(self.num_envs), dones, extras


def _cfg(**algorithm: object) -> dict:
    return {
        "num_steps_per_env": STEPS,
        "obs_groups": {"student": ["policy", "image"], "teacher": ["teacher"], "critic": ["critic"]},
        "student": {
            "class_name": "CNNRNNModel",
            "hidden_dims": [16],
            "obs_normalization": True,
            "rnn_type": "gru",
            "rnn_hidden_dim": 8,
            "rnn_num_layers": 1,
            "cnn_cfg": {
                "output_channels": [2],
                "kernel_size": 3,
                "stride": 2,
                "activation": "elu",
                "global_pool": "avg",
            },
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "teacher": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16], "obs_normalization": True},
        "algorithm": {
            "class_name": "DistillationPPO",
            "num_learning_epochs": 2,
            "gradient_length": 2,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "loss_type": "mse",
            "bc_weight": 1.0,
            "bc_weight_final": 0.1,
            "bc_anneal_updates": 4,
            **algorithm,
        },
        "multi_gpu": None,
    }


def _build(**algorithm: object) -> tuple[DistillationPPO, _DummyDistillVecEnv]:
    env = _DummyDistillVecEnv()
    alg = DistillationPPO.construct_algorithm(env.get_observations(), env, _cfg(**algorithm), device="cpu")
    alg.teacher_loaded = True
    alg.train_mode()
    return alg, env


def _rollout(alg: DistillationPPO, env: _DummyDistillVecEnv) -> None:
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(STEPS):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)


class TestDistillationPPO:
    """The combined update trains both models, anneals the cloning weight and checkpoints the critic."""

    def test_update_reports_every_loss_and_moves_both_models(self) -> None:
        """One update logs the cloning, surrogate, value and entropy losses and steps the student and the critic."""
        torch.manual_seed(0)
        alg, env = _build()
        before_s = [p.detach().clone() for p in alg.student.parameters()]
        before_c = [p.detach().clone() for p in alg.critic.parameters()]
        _rollout(alg, env)
        losses = alg.update()
        for key in ("behavior", "surrogate", "value", "entropy", "clip_fraction", "bc_weight"):
            assert key in losses and torch.isfinite(torch.tensor(losses[key])), key
        assert abs(losses["bc_weight"] - 1.0) < 1e-9
        assert any(not torch.equal(b, p) for b, p in zip(before_s, alg.student.parameters(), strict=True))
        assert any(not torch.equal(b, p) for b, p in zip(before_c, alg.critic.parameters(), strict=True))

    def test_bc_weight_anneals_linearly_then_holds(self) -> None:
        """The cloning weight falls linearly to its final value over the anneal window and then stays there."""
        torch.manual_seed(0)
        alg, env = _build()
        weights = []
        for _ in range(6):
            _rollout(alg, env)
            weights.append(alg.update()["bc_weight"])
        assert torch.allclose(torch.tensor(weights), torch.tensor([1.0, 0.775, 0.55, 0.325, 0.1, 0.1]))

    def test_critic_warmup_keeps_the_surrogate_out_of_the_loss(self) -> None:
        """During warm-up the student's parameters only move by the cloning and entropy-free losses."""
        torch.manual_seed(0)
        alg, env = _build(critic_warmup_updates=1, bc_weight=0.0, bc_weight_final=0.0)
        before = [p.detach().clone() for p in alg.student.parameters()]
        _rollout(alg, env)
        alg.update()
        # a zero cloning weight and no surrogate leave the student untouched; the critic still learns
        assert all(torch.equal(b, p) for b, p in zip(before, alg.student.parameters(), strict=True))
        _rollout(alg, env)
        alg.update()
        assert any(not torch.equal(b, p) for b, p in zip(before, alg.student.parameters(), strict=True))

    def test_checkpoint_round_trip_and_teacher_only_load(self) -> None:
        """A full checkpoint restores the critic; an RL checkpoint loads only the teacher."""
        torch.manual_seed(0)
        alg, env = _build()
        _rollout(alg, env)
        alg.update()
        saved = alg.save()
        assert {"student_state_dict", "teacher_state_dict", "critic_state_dict", "optimizer_state_dict"} <= set(saved)
        other, _ = _build()
        assert other.load(saved, None, True)
        for a, b in zip(alg.critic.parameters(), other.critic.parameters(), strict=True):
            assert torch.equal(a, b)
        fresh, _ = _build()
        critic_before = [p.detach().clone() for p in fresh.critic.parameters()]
        assert not fresh.load({"actor_state_dict": alg.teacher.state_dict()}, None, True)
        assert fresh.teacher_loaded
        assert all(torch.equal(b, p) for b, p in zip(critic_before, fresh.critic.parameters(), strict=True))
