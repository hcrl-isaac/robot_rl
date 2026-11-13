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


def _instruction_alg(tmp_path: object, instruction: bool) -> tuple[FbCpr, TensorDict]:
    motions = TensorDict({"policy": torch.randn(3, 6, 8)}, batch_size=[3, 6])
    motion_path = f"{tmp_path}/motions.pt"
    torch.save(motions, motion_path)
    cfg = _train_cfg("GaussianDistribution")
    cfg["actor"].pop("distribution_cfg")
    cfg["algorithm"]["motion_path"] = motion_path
    cfg["algorithm"]["expert_rollout_length"] = 4
    for critic in ("forward_map", "disc_critic", "aux_critic"):
        cfg["algorithm"][critic]["num_parallel"] = 2
    if instruction:
        cfg["algorithm"]["instruction_critic"] = dict(cfg["algorithm"]["aux_critic"])
        cfg["algorithm"]["instruction_reward_key"] = "instruction_reward"
        cfg["algorithm"]["instruction_reg_coef"] = 1.0
    obs = TensorDict(
        {"policy": torch.zeros(_Env.num_envs, 8), "instruction_reward": torch.zeros(_Env.num_envs, 1)},
        batch_size=[_Env.num_envs],
    )
    return FbCpr.construct_algorithm(obs, _Env(), cfg, "cpu"), obs


def _fill_and_update(alg: FbCpr, obs: TensorDict, instruction_reward: float) -> tuple[dict, dict]:
    no_dones = torch.zeros(_Env.num_envs, dtype=torch.bool)
    # the runner collects under inference mode
    with torch.inference_mode():
        for _ in range(8):
            alg.act(obs)
            next_obs = obs.clone()
            next_obs["policy"] = torch.randn(_Env.num_envs, 8)
            next_obs["instruction_reward"] = torch.full((_Env.num_envs, 1), instruction_reward)
            rewards = torch.full((_Env.num_envs,), 1.0 + instruction_reward)
            alg.process_env_step(next_obs, rewards, no_dones, {"time_outs": no_dones})
        alg.compute_gammas()
    return alg.update()


def test_an_instruction_critic_learns_its_reward_and_the_aux_critic_the_rest(tmp_path: object) -> None:
    """The instruction reward goes to its own critic and actor term; the auxiliary critic sees the remainder."""
    alg, obs = _instruction_alg(tmp_path, instruction=True)
    losses, _ = _fill_and_update(alg, obs, instruction_reward=0.5)
    assert "Loss/critic_instruction" in losses and "Train/actor/q_instruction" in losses
    assert alg.instruction_critic in alg.models
    # the auxiliary normaliser saw the env reward minus the instruction part (1.0), not 1.5
    torch.testing.assert_close(
        alg.aux_reward_normalizer.mean / (1 - alg.aux_reward_normalizer.tau), torch.tensor([1.0])
    )
    saved = alg.save()
    assert {"instruction_critic_state_dict", "target_instruction_critic_state_dict"} <= saved.keys()


def test_without_an_instruction_critic_the_aux_critic_keeps_the_whole_reward(tmp_path: object) -> None:
    """No instruction critic: the update is unchanged and nothing instruction-related is saved."""
    alg, obs = _instruction_alg(tmp_path, instruction=False)
    losses, _ = _fill_and_update(alg, obs, instruction_reward=0.5)
    assert "Loss/critic_instruction" not in losses
    torch.testing.assert_close(
        alg.aux_reward_normalizer.mean / (1 - alg.aux_reward_normalizer.tau), torch.tensor([1.5])
    )
    assert "instruction_critic_state_dict" not in alg.save()
