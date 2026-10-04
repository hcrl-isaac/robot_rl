# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where FB-CPR places the expert motions, and how it bounds its actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms.fb_cpr import FbCpr, _expert_storage_device
from robot_rl.modules import TruncatedGaussianDistribution

CLIP_ACTIONS = 5.0


class SubclassedTruncatedGaussian(TruncatedGaussianDistribution):
    """A truncated Gaussian under another class name."""


TRUNCATED_GAUSSIAN_NAMES = [
    "TruncatedGaussianDistribution",
    "robot_rl.modules.distribution:TruncatedGaussianDistribution",
    f"{__name__}:SubclassedTruncatedGaussian",
]
"""A bare name, a module-qualified one and a subclass: every spelling the bounds rule must recognise."""


def _cfg(storage_device: str | None, **algorithm: str | None) -> dict:
    return {"storage_device": storage_device, "algorithm": algorithm}


@pytest.mark.parametrize("algorithm", [{}, {"expert_storage_device": None}])
@pytest.mark.parametrize("storage_device", ["cpu", "cuda:1"])
def test_an_unset_expert_device_follows_the_storage_device(storage_device: str, algorithm: dict) -> None:
    """Without ``expert_storage_device`` the motions sit where the replay buffer does."""
    assert _expert_storage_device(_cfg(storage_device, **algorithm), "cuda:0") == storage_device


def test_an_explicit_expert_device_overrides_the_storage_device() -> None:
    """``expert_storage_device`` places the motions whatever ``storage_device`` says."""
    assert _expert_storage_device(_cfg("cuda", expert_storage_device="cpu"), "cuda:0") == "cpu"
    assert _expert_storage_device(_cfg("cpu", expert_storage_device="cuda:2"), "cuda:0") == "cuda:2"


@pytest.mark.parametrize("cfg", [_cfg("cpu", expert_storage_device="cuda"), _cfg("cuda")])
def test_a_bare_cuda_names_the_runner_gpu(cfg: dict) -> None:
    """A bare ``"cuda"``, given or inherited, is the runner's GPU and not torch's current device."""
    assert _expert_storage_device(cfg, "cuda:3") == "cuda:3"
    assert _expert_storage_device(cfg, "cpu") == "cuda"


class _Env:
    num_envs, num_actions, max_episode_length = 4, 3, 10


def _train_cfg(distribution_class: str) -> dict:
    fuse = {"class_name": "ResidualFuseModel", "embedding_dims": [16, 16], "hidden_dims": [16, 16]}
    mlp = {"class_name": "MLPModel", "hidden_dims": [16, 16]}
    return {
        "actor": {**fuse, "distribution_cfg": {"class_name": distribution_class, "low": -1.0, "high": 1.0}},
        "algorithm": {
            "class_name": "FbCpr",
            "z_dim": 4,
            "z_buffer_capacity": 16,
            "motion_path": "",
            "expert_sequence_length": 2,
            "steps_per_z_update": 1,
            "storage_scale": 1,
            "batch_size": 4,
            "compile_mode": None,
            "forward_map": dict(fuse),
            "backward_map": dict(mlp),
            "disc_critic": dict(fuse),
            "aux_critic": dict(fuse),
            "discriminator": dict(mlp),
        },
        "obs_groups": {k: ["policy"] for k in ("actor", "critic", "backward", "discriminator", "expert")},
        "storage_device": "cpu",
        "multi_gpu": None,
        "clip_actions": CLIP_ACTIONS,
    }


@pytest.mark.parametrize("class_name", TRUNCATED_GAUSSIAN_NAMES)
def test_training_bounds_a_truncated_gaussian_actor_by_clip_actions(class_name: str) -> None:
    """The actor FB-CPR trains is bounded by ``clip_actions`` over the +-1 its cfg lists, however the class is named."""
    obs = TensorDict({"policy": torch.zeros(_Env.num_envs, 8)}, batch_size=[_Env.num_envs])
    alg = FbCpr.construct_algorithm(obs, _Env(), _train_cfg(class_name), "cpu", inference=True)
    saturated = alg.actor.distribution.deterministic_output(torch.tensor([-50.0, 50.0]))
    torch.testing.assert_close(saturated, torch.tensor([-CLIP_ACTIONS, CLIP_ACTIONS]))


def test_training_leaves_another_head_the_bounds_its_cfg_gives() -> None:
    """Only the truncated Gaussian takes its bounds from ``clip_actions``."""
    obs = TensorDict({"policy": torch.zeros(_Env.num_envs, 8)}, batch_size=[_Env.num_envs])
    cfg = _train_cfg("SquashedTanhGaussianDistribution")
    alg = FbCpr.construct_algorithm(obs, _Env(), cfg, "cpu", inference=True)
    saturated = alg.actor.distribution.deterministic_output(torch.tensor([-50.0, 50.0]))
    torch.testing.assert_close(saturated, torch.tensor([-1.0, 1.0]))
