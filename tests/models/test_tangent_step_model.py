# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the tangent-step vMF actor."""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from robot_rl.models import TangentStepModel
from robot_rl.modules.distribution import TangentStepVonMisesFisherDistribution

NUM_ENVS = 5
DIM = 8


def _make_obs(latent: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": torch.randn(NUM_ENVS, 6), "latent": latent}, batch_size=[NUM_ENVS])


def _unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def test_zero_step_holds_and_turn_angle_is_arc_length() -> None:
    """A zero tangent step keeps the base direction; a step of length theta turns by theta."""
    dist = TangentStepVonMisesFisherDistribution(DIM, init_std=0.1)
    base = _unit(torch.randn(NUM_ENVS, DIM))
    dist.base = base
    assert torch.allclose(dist.mean_direction(torch.zeros(NUM_ENVS, DIM)), base, atol=1e-6)
    step = torch.randn(NUM_ENVS, DIM)
    tangent = step - (step * base).sum(-1, keepdim=True) * base
    theta = tangent.norm(dim=-1).clamp(max=dist.max_step)
    mu = dist.mean_direction(step)
    assert torch.allclose(mu.norm(dim=-1), torch.ones(NUM_ENVS), atol=1e-6)
    assert torch.allclose(torch.acos((mu * base).sum(-1).clamp(-1, 1)), theta, atol=1e-4)
    # a purely radial step does not move the mean
    assert torch.allclose(dist.mean_direction(0.7 * base), base, atol=1e-6)
    # no base yet: the step itself is the direction
    dist.base = torch.zeros(NUM_ENVS, DIM)
    assert torch.allclose(dist.mean_direction(step), _unit(step), atol=1e-6)


def test_model_uses_observed_latent_and_samples_on_sphere() -> None:
    """The model reads the base from the latent group; samples are unit vectors with finite log-probs."""
    latent = math.sqrt(DIM) * _unit(torch.randn(NUM_ENVS, DIM))
    obs = _make_obs(latent)
    model = TangentStepModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        DIM,
        hidden_dims=[16],
        distribution_cfg={"class_name": "TangentStepVonMisesFisherDistribution", "init_std": 0.5},
    )
    # zero the MLP so the deterministic output must equal the (normalized) observed latent
    for p in model.mlp.parameters():
        torch.nn.init.zeros_(p)
    assert torch.allclose(model(obs), _unit(latent), atol=1e-6)
    sample, log_prob = model.act_and_log_prob(obs)
    assert sample.shape == (NUM_ENVS, DIM) and torch.allclose(sample.norm(dim=-1), torch.ones(NUM_ENVS), atol=1e-5)
    assert torch.isfinite(log_prob).all() and log_prob.shape == (NUM_ENVS,)
    assert torch.allclose(model.output_mean, _unit(latent), atol=1e-6)
