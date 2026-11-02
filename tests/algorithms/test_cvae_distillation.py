# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CVAE distillation algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from robot_rl.algorithms import CvaeDistillation
from robot_rl.models import CVAEModel, MLPModel
from robot_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 12
NUM_ACTIONS = 3


def _make_obs() -> TensorDict:
    return TensorDict(
        {"history": torch.randn(NUM_ENVS, 6), "goal": torch.randn(NUM_ENVS, 4), "state": torch.randn(NUM_ENVS, 5)},
        batch_size=[NUM_ENVS],
    )


def _make_alg(**kwargs: float) -> tuple[CvaeDistillation, TensorDict]:
    obs = _make_obs()
    obs_groups = {
        "student": ["history"],
        "prior": ["history", "goal"],
        "encoder": ["history", "state", "goal"],
        "teacher": ["history", "state"],
    }
    student = CVAEModel(
        obs,
        obs_groups,
        "student",
        NUM_ACTIONS,
        latent_dim=3,
        prior_hidden_dims=[16],
        encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
    )
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[16])
    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = CvaeDistillation(student, teacher, storage, gradient_length=3, num_learning_epochs=2, **kwargs)
    return alg, obs


def _rollout(alg: CvaeDistillation, obs: TensorDict) -> None:
    for _ in range(NUM_STEPS):
        actions = alg.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})


def test_losses_decrease_and_kl_anneals() -> None:
    """The cloning loss falls over updates and the KL weight anneals to its final value."""
    alg, obs = _make_alg(kl_weight=0.1, kl_weight_final=0.01, kl_anneal_iterations=5)
    alg.train_mode()
    losses = []
    for _ in range(6):
        _rollout(alg, obs)
        losses.append(alg.update())
    assert set(losses[0]) == {"behavior", "kl", "prior_behavior", "kl_weight"}
    assert losses[-1]["behavior"] < losses[0]["behavior"]
    assert losses[0]["kl_weight"] > losses[-1]["kl_weight"]
    assert abs(losses[-1]["kl_weight"] - 0.01) < 1e-9


def test_save_load_round_trip() -> None:
    """Checkpoints round-trip and a PPO checkpoint loads the teacher only."""
    alg, obs = _make_alg()
    saved = alg.save()
    assert {"student_state_dict", "teacher_state_dict", "optimizer_state_dict"} <= set(saved)
    alg2, _ = _make_alg()
    alg2.load(saved, None, strict=True)
    assert torch.allclose(alg2.student(obs), alg.student(obs))
    # a PPO checkpoint loads only the teacher
    assert alg2.load({"actor_state_dict": saved["teacher_state_dict"]}, None, strict=True) is False
    assert alg2.teacher_loaded
