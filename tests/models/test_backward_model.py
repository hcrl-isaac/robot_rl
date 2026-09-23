# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the BackwardModel distillation teacher."""

from __future__ import annotations

import pathlib
import torch
from tensordict import TensorDict

import pytest

from robot_rl.models import BackwardModel

NUM_ENVS = 5
FEATURE_DIM = 6
LATENT_DIM = 4


@pytest.fixture
def backward_path(tmp_path: pathlib.Path) -> str:
    """Save a small scripted ``B`` and return its path."""
    torch.manual_seed(0)
    backward = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, 16), torch.nn.ELU(), torch.nn.Linear(16, LATENT_DIM))
    path = str(tmp_path / "backward.pt")
    torch.jit.script(backward).save(path)
    return path


def _obs(width: int) -> TensorDict:
    return TensorDict({"teacher": torch.randn(NUM_ENVS, width)}, batch_size=[NUM_ENVS])


@pytest.mark.parametrize("horizon", [1, 4])
def test_matches_per_frame_mean(backward_path: str, horizon: int) -> None:
    """The output is the normalized mean of B over each frame, with the horizon inferred from the width."""
    obs = _obs(horizon * FEATURE_DIM)
    model = BackwardModel(obs, {"teacher": ["teacher"]}, "teacher", LATENT_DIM, backward_path=backward_path)
    assert model.horizon == horizon

    backward = torch.jit.load(backward_path)
    frames = obs["teacher"].split(FEATURE_DIM, dim=-1)
    expected = torch.nn.functional.normalize(torch.stack([backward(f) for f in frames]).mean(dim=0), dim=-1)
    torch.testing.assert_close(model(obs), expected)


def test_rejects_width_not_a_multiple(backward_path: str) -> None:
    """A width that is not a whole number of frames fails at construction."""
    with pytest.raises(ValueError, match="not a multiple"):
        BackwardModel(
            _obs(FEATURE_DIM + 1), {"teacher": ["teacher"]}, "teacher", LATENT_DIM, backward_path=backward_path
        )


def test_rejects_latent_dim_mismatch(backward_path: str) -> None:
    """B's latent width must match the model's output width."""
    with pytest.raises(ValueError, match="latents"):
        BackwardModel(
            _obs(FEATURE_DIM), {"teacher": ["teacher"]}, "teacher", LATENT_DIM + 1, backward_path=backward_path
        )


def test_frozen_and_round_trips(backward_path: str, tmp_path: pathlib.Path) -> None:
    """B takes no gradient, stays in eval mode, and its weights survive a state-dict round trip."""
    obs = _obs(2 * FEATURE_DIM)
    model = BackwardModel(obs, {"teacher": ["teacher"]}, "teacher", LATENT_DIM, backward_path=backward_path)
    model.train()
    assert not model.backward.training
    assert all(not p.requires_grad for p in model.parameters())

    other_path = str(tmp_path / "other.pt")
    torch.jit.script(
        torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, 16), torch.nn.ELU(), torch.nn.Linear(16, LATENT_DIM))
    ).save(other_path)
    other = BackwardModel(obs, {"teacher": ["teacher"]}, "teacher", LATENT_DIM, backward_path=other_path)
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(other(obs), model(obs))


def test_old_empty_teacher_checkpoint_loads(backward_path: str) -> None:
    """A checkpoint saved with a weightless teacher loads its student under a B teacher, which keeps its own B."""
    from robot_rl.algorithms.distillation import Distillation
    from robot_rl.models import MLPModel
    from robot_rl.storage import RolloutStorage

    obs = TensorDict(
        {"policy": torch.randn(NUM_ENVS, 3), "teacher": torch.randn(NUM_ENVS, 2 * FEATURE_DIM)}, batch_size=[NUM_ENVS]
    )
    groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = MLPModel(obs, groups, "student", LATENT_DIM, hidden_dims=[8])
    teacher = BackwardModel(obs, groups, "teacher", LATENT_DIM, backward_path=backward_path)
    alg = Distillation(student, teacher, RolloutStorage("distillation", NUM_ENVS, 4, obs, [LATENT_DIM]))
    assert alg.teacher_loaded
    before = teacher(obs)
    checkpoint = {"student_state_dict": student.state_dict(), "teacher_state_dict": {}}
    alg.load(checkpoint, {"student": True, "teacher": True}, strict=True)
    torch.testing.assert_close(alg.teacher(obs), before)
