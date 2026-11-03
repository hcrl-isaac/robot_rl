# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Distillation algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms.distillation import Distillation
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 12
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_distillation_setup(gradient_length: int = 3, num_learning_epochs: int = 1) -> tuple:
    """Build a Distillation instance with small networks."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32])
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])

    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    alg = Distillation(
        student,
        teacher,
        storage,
        num_learning_epochs=num_learning_epochs,
        gradient_length=gradient_length,
        learning_rate=1e-3,
    )
    return alg, obs, storage


def _fill_distillation_storage(alg: Distillation, obs: TensorDict) -> None:
    """Fill the distillation storage with transitions."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.privileged_actions = alg.teacher(obs).detach()
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


class TestDistillationLoss:
    """Tests for distillation loss computation."""

    def test_loss_decreases_over_updates(self) -> None:
        """Behavior loss should decrease over repeated update() calls (learning signal works)."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3, num_learning_epochs=2)
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_distillation_storage(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["behavior"])

        # Loss should generally decrease; allow some noise — check first vs last
        assert losses[-1] < losses[0], f"Loss should decrease over updates, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_accumulation_step_count(self) -> None:
        """Optimizer should step floor(num_transitions / gradient_length) times per epoch."""
        gradient_length = 4
        alg, obs, _storage = _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=1)
        alg.train_mode()

        _fill_distillation_storage(alg, obs)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()

        expected_steps = NUM_STEPS // gradient_length
        assert step_count == expected_steps, f"Expected {expected_steps} optimizer steps, got {step_count}"

    def test_update_changes_student_but_not_teacher(self) -> None:
        """Student parameters should change after update, while teacher parameters remain frozen."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3)
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.student.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_distillation_storage(alg, obs)
        alg.update()

        any_student_changed = any(
            not torch.equal(p, student_before[name]) for name, p in alg.student.named_parameters()
        )
        assert any_student_changed, "Student parameters should change after an update"

        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during student update"


class TestTeacherReadiness:
    """A teacher is ready when it loads its own weights or gets them from a checkpoint."""

    @staticmethod
    def _alg() -> Distillation:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"student": ["policy"], "teacher": ["policy"]}
        student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32])
        teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32])
        storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        return Distillation(student, teacher, storage)

    def test_weighted_teacher_needs_a_checkpoint(self) -> None:
        """A teacher with weights is not ready until loaded, and an empty teacher checkpoint is refused."""
        alg = self._alg()
        assert not alg.teacher_loaded
        with pytest.raises(ValueError, match="no teacher weights"):
            alg.load({"student_state_dict": alg.student.state_dict(), "teacher_state_dict": {}}, None, strict=True)

    def test_rl_actor_loads_as_teacher(self) -> None:
        """An RL checkpoint's actor becomes the teacher."""
        alg = self._alg()
        source = MLPModel(make_obs(NUM_ENVS, OBS_DIM), {"actor": ["policy"]}, "actor", NUM_ACTIONS, hidden_dims=[32])
        alg.load({"actor_state_dict": source.state_dict()}, None, strict=True)
        assert alg.teacher_loaded
        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, source.state_dict()[name])


class TestRecurrentTeacher:
    """A recurrent teacher's memory must survive the update between rollouts."""

    def test_update_keeps_teacher_hidden_state(self) -> None:
        """The teacher's state after ``update()`` is the state its rollout ended in."""
        from robot_rl.models import RNNModel

        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"student": ["policy"], "teacher": ["policy"]}
        student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32])
        teacher = RNNModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32], rnn_hidden_dim=16)
        storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        alg = Distillation(student, teacher, storage, num_learning_epochs=2, gradient_length=3, learning_rate=1e-3)
        alg.train_mode()
        for _ in range(2):
            for _ in range(NUM_STEPS):
                alg.act(make_obs(NUM_ENVS, OBS_DIM))
                alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
            before = [h.clone() for h in alg.teacher.get_hidden_state()]
            alg.update()
            assert all(torch.equal(a, b) for a, b in zip(alg.teacher.get_hidden_state(), before, strict=True))


class TestTeacherDrivenRollouts:
    """DAgger mixing: drawn envs execute the teacher's action, the fraction decays, labels stay the teacher's."""

    def _setup(self, prob: float, decay: int = 0) -> tuple:
        alg, obs, storage = _make_distillation_setup()
        alg.teacher_action_prob_init = alg.teacher_action_prob = prob
        alg.teacher_action_decay_updates = decay
        return alg, obs, storage

    def test_default_executes_the_student(self) -> None:
        """With mixing off, the executed action is the student's."""
        alg, obs, _ = _make_distillation_setup()
        actions = alg.act(obs)
        assert torch.equal(actions, alg.transition.actions)
        assert alg._teacher_driven is None

    def test_all_teacher_driven_executes_the_teacher(self) -> None:
        """With every env drawn, the executed action is the teacher's while storage keeps the student's."""
        alg, obs, _ = self._setup(1.0)
        actions = alg.act(obs)
        assert torch.allclose(actions, alg.teacher(obs))
        # the stored action and the label are unchanged by the mixing
        assert torch.equal(alg.transition.privileged_actions, alg.teacher(obs).detach())
        assert not torch.equal(alg.transition.actions, actions)

    def test_fraction_decays_and_resets_redraw(self) -> None:
        """The fraction decays per update, and only resetting envs redraw at the current fraction."""
        alg, obs, _ = self._setup(1.0, decay=2)
        alg.act(obs)
        assert alg._teacher_driven.all()
        _fill_distillation_storage(alg, obs)
        alg.update()
        assert alg.teacher_action_prob == pytest.approx(0.5)
        _fill_distillation_storage(alg, obs)
        loss = alg.update()
        assert alg.teacher_action_prob == pytest.approx(0.0)
        assert loss["teacher_action_prob"] == pytest.approx(0.0)
        # envs keep their draw until they reset, then redraw at the current (zero) fraction
        assert alg._teacher_driven.all()
        alg.act(obs)
        dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
        alg.process_env_step(obs, torch.zeros(NUM_ENVS), dones, {})
        assert alg._teacher_driven.tolist() == [False, True, False, True]
