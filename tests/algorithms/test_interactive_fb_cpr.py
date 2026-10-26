# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The interactive FB-CPR's sphere log map, expert-motion adaptation, episode-aligned clips and update."""

from __future__ import annotations

import math
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import InteractiveFbCpr
from robot_rl.algorithms.interactive_fb_cpr import sphere_log_map
from robot_rl.storage import ReplayBuffer

NUM_ENVS, NUM_ACTIONS, EPISODE = 4, 3, 10
WIDTHS = {"proprio": 6, "state": 5, "object": 4, "expert": 13 + 2 * NUM_ACTIONS}


class _Env:
    num_envs, num_actions, max_episode_length = NUM_ENVS, NUM_ACTIONS, EPISODE


def _obs(batch: tuple[int, ...] = (NUM_ENVS,), groups: tuple[str, ...] = tuple(WIDTHS)) -> TensorDict:
    return TensorDict({g: torch.randn(*batch, WIDTHS[g]) for g in groups}, batch_size=list(batch))


def _cfg(motion_path: str) -> dict:
    fuse = {"class_name": "ResidualFuseModel", "embedding_dims": [16, 16], "hidden_dims": [16, 16]}
    mlp = {"class_name": "MLPModel", "hidden_dims": [16, 16]}
    return {
        "actor": {
            **fuse,
            "distribution_cfg": {"class_name": "TruncatedGaussianDistribution", "low": -1.0, "high": 1.0},
        },
        "algorithm": {
            "class_name": "InteractiveFbCpr",
            "z_dim": 4,
            "z_buffer_capacity": 16,
            "motion_path": motion_path,
            "expert_sequence_length": 2,
            "steps_per_z_update": 1,
            "storage_scale": 2,
            "batch_size": 8,
            "compile_mode": None,
            "logo_horizon": 3,
            "expert_absent_groups": ["object"],
            "expert_rollout_length": 4,
            "forward_map": {**fuse, "num_parallel": 2},
            "backward_map": dict(mlp),
            "disc_critic": {**fuse, "num_parallel": 2},
            "aux_critic": {**fuse, "num_parallel": 2},
            "discriminator": dict(mlp),
        },
        "obs_groups": {
            "actor": ["proprio", "object"],
            "critic": ["proprio", "state", "object"],
            "backward": ["state", "object"],
            "discriminator": ["state"],
            "expert": ["expert"],
        },
        "storage_device": "cpu",
        "multi_gpu": None,
        "clip_actions": 1.0,
    }


@pytest.fixture
def alg(tmp_path: Path) -> InteractiveFbCpr:
    """Build an interactive FB-CPR over motions without the object group."""
    # the motions carry an unread group (an actor history of another width) and lack the object group
    motions = _obs((3, 8), ("proprio", "state", "expert"))
    motions["history"] = torch.zeros(3, 8, 7)
    path = tmp_path / "motions.pt"
    torch.save(motions, path)
    return InteractiveFbCpr.construct_algorithm(_obs(), _Env(), _cfg(str(path)), "cpu")


def test_the_log_map_has_the_geodesic_length_and_is_tangent() -> None:
    """The log map's norm is ``r`` times the angle between the points, and it is orthogonal to the base."""
    r = math.sqrt(4)
    base = torch.tensor([[r, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, r, 0.0, 0.0]])
    m = sphere_log_map(base, target)
    torch.testing.assert_close(m, torch.tensor([[0.0, r * math.pi / 2, 0.0, 0.0]]))
    assert not sphere_log_map(base, base).any()


def test_expert_motions_gain_zero_absent_groups_and_drop_unread_ones(alg: InteractiveFbCpr) -> None:
    """The object group is zeros of the env's width; groups no expert reader uses are gone."""
    motions = alg.expert_buffer.motions
    assert motions["object"].shape == (3, 8, WIDTHS["object"])
    assert not motions["object"].any()
    assert set(motions.keys()) == {"state", "object", "expert"}


def test_the_replay_keeps_every_step_so_windows_stay_in_one_episode(alg: InteractiveFbCpr) -> None:
    """Every transition is stored in time order, and no sampled window contains an episode end."""
    rb = alg.replay_buffer
    for t in range(2 * EPISODE):
        tr = ReplayBuffer.Transition()
        tr.observations, tr.next_observations = _obs(), _obs()
        tr.actions, tr.rewards = torch.zeros(NUM_ENVS, NUM_ACTIONS), torch.zeros(NUM_ENVS)
        tr.context, tr.next_terminated = torch.zeros(NUM_ENVS, 4), torch.zeros(NUM_ENVS, dtype=torch.uint8)
        tr.dones = torch.full((NUM_ENVS,), float(t % EPISODE == EPISODE - 1))
        rb.add_transitions(tr)
    assert len(rb) == 2 * EPISODE * NUM_ENVS
    steps = alg._valid_rows(3, 64)
    assert rb.dones[steps].sum() == 0
    # consecutive steps of a window are one env's consecutive rows
    assert torch.all(torch.diff(steps, dim=1) % (NUM_ENVS * rb.capacity_per_env) == NUM_ENVS)


def test_an_update_runs_and_logs_the_local_goal_residual(alg: InteractiveFbCpr) -> None:
    """A full update over a filled buffer trains every model and reports both residual terms."""
    obs = _obs()
    time_outs = {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)}
    # the runner collects under inference mode
    with torch.inference_mode():
        for _ in range(2 * EPISODE):
            alg.act(obs)
            next_obs = _obs()
            alg.process_env_step(next_obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), time_outs)
            obs = next_obs
        alg.compute_gammas()
    before = [p.clone() for p in alg.actor.parameters()]
    loss_dict, extras = alg.update()
    assert torch.isfinite(loss_dict["Actor_Loss/logo_local_loss"])
    assert torch.isfinite(loss_dict["Actor_Loss/logo_goal_loss"])
    assert "logo_goal_angle" in extras
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.actor.parameters(), strict=True))
