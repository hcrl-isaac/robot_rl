# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CVAE behavior model."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from robot_rl.models import CVAEModel

NUM_ENVS = 6
NUM_ACTIONS = 5
LATENT_DIM = 4


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "history": torch.randn(NUM_ENVS, 10),
            "goal": torch.randn(NUM_ENVS, 7),
            "state": torch.randn(NUM_ENVS, 9),
        },
        batch_size=[NUM_ENVS],
    )


OBS_GROUPS = {
    "student": ["history"],
    "prior": ["history", "goal"],
    "encoder": ["history", "state", "goal"],
    "teacher": ["history", "state"],
}


def _make_model() -> CVAEModel:
    return CVAEModel(
        _make_obs(),
        OBS_GROUPS,
        "student",
        NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        prior_hidden_dims=[16],
        encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
    )


def test_shapes_and_paths() -> None:
    """Every path returns the right shape and the deterministic path decodes the prior mean."""
    model = _make_model()
    obs = _make_obs()
    mu_p, std_p = model.prior_params(obs)
    mu_q, std_q = model.posterior_params(obs, mu_p)
    assert mu_p.shape == std_p.shape == mu_q.shape == std_q.shape == (NUM_ENVS, LATENT_DIM)
    assert (std_p > 0).all() and (std_q > 0).all()
    assert model(obs).shape == (NUM_ENVS, NUM_ACTIONS)
    assert model(obs, use_encoder=True).shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.act_train(obs, prior_ratio=0.5).shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.output_std.shape == (NUM_ACTIONS,)
    # the deterministic path decodes the prior mean
    assert torch.allclose(model(obs), model.decode(obs, mu_p))


def test_kl_divergence() -> None:
    """KL is zero between identical Gaussians and positive otherwise."""
    mu = torch.randn(3, LATENT_DIM)
    std = torch.rand(3, LATENT_DIM) + 0.1
    assert torch.allclose(CVAEModel.kl_divergence(mu, std, mu, std), torch.zeros(3), atol=1e-6)
    assert (CVAEModel.kl_divergence(mu + 1.0, std, mu, std) > 0).all()


def test_decoder_groups_must_prefix_prior() -> None:
    """A decoder set that is not a prefix of the prior set is rejected."""
    groups = dict(OBS_GROUPS, prior=["goal", "history"])
    with pytest.raises(ValueError, match="prefix"):
        CVAEModel(_make_obs(), groups, "student", NUM_ACTIONS, latent_dim=LATENT_DIM)


def test_jit_export_matches_model() -> None:
    """The scripted export reproduces the python model on every exported method."""
    model = _make_model().eval()
    obs = _make_obs()
    scripted = torch.jit.script(model.as_jit())
    x = torch.cat([obs["history"], obs["goal"]], dim=-1)
    assert torch.allclose(scripted(x), model(obs), atol=1e-6)
    mu, std = scripted.prior(x)
    mu_ref, std_ref = model.prior_params(obs)
    assert torch.allclose(mu, mu_ref, atol=1e-6) and torch.allclose(std, std_ref, atol=1e-6)
    z = torch.randn(NUM_ENVS, LATENT_DIM)
    assert torch.allclose(scripted.decode(obs["history"], z), model.decode(obs, z), atol=1e-6)
    assert scripted.goal_dim == 7 and scripted.latent_dim == LATENT_DIM
