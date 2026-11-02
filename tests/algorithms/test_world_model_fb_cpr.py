"""CPU tests for FB-CPR with an imagined actor objective (the world-model AC's toy env, no Isaac Sim)."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import FbCpr, WorldModelFbCpr
from robot_rl.env import ObsTermLayout, WorldModelSpec
from tests.algorithms.test_world_model_ac import _TrackingEnv

Z_DIM, BATCH = 8, 64


class _ExpertBuffer:
    """Stands in for the motion dataset: random expert states."""

    def __init__(self, motion_path: str, groups: list[str], device: str | None) -> None:
        self.groups = groups

    def sample(self, n: int, device: str, seq_length: int = 1) -> tuple[TensorDict, TensorDict]:
        def draw() -> TensorDict:
            return TensorDict({g: torch.randn(n, 4) for g in self.groups}, batch_size=[n])

        return draw(), draw()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _env() -> _TrackingEnv:
    # FB-CPR keeps every env's episode in step, so nothing terminates early; it conditions on z, not a goal
    env = _TrackingEnv(num_envs=16, max_error=float("inf"))
    env.world_model_spec = WorldModelSpec(
        state_group="state",
        obs_group="obs_bar",
        context_group="context",
        action_terms=["last_action"],
        layouts={
            "policy": [ObsTermLayout("pos", 2, 2), ObsTermLayout("vel", 2, 2), ObsTermLayout("last_action", 2, 2)],
            "obs_bar": [ObsTermLayout("pos", 2), ObsTermLayout("vel", 2)],
        },
    )
    return env


def _cfg(horizon: int, actor_start_update: int = 0) -> dict:
    fuse = {"class_name": "ResidualFuseModel", "embedding_dims": [32], "hidden_dims": [32, 32], "num_parallel": 2}
    mlp = {"class_name": "MLPModel", "hidden_dims": [32]}
    return {
        "actor": {
            "class_name": "ResidualFuseModel",
            "embedding_dims": [32],
            "hidden_dims": [32, 32],
            "distribution_cfg": {"class_name": "TruncatedGaussianDistribution", "init_std": 0.25},
        },
        "algorithm": {
            "class_name": "WorldModelFbCpr",
            "z_dim": Z_DIM,
            "z_buffer_capacity": 128,
            "motion_path": "",
            "expert_sequence_length": 2,
            "steps_per_z_update": 10,
            "storage_scale": 2,
            "batch_size": BATCH,
            "compile_mode": None,
            "imagination_compile_mode": None,
            "forward_map": dict(fuse),
            "backward_map": {**mlp, "last_activation": "ball_norm"},
            "disc_critic": dict(fuse),
            "aux_critic": dict(fuse),
            "discriminator": {**mlp, "last_activation": "sigmoid"},
            "world_model": {"hidden_dims": [64, 64], "head_hidden_dims": [32]},
            "wm_window": 4,
            "wm_batch_size": 64,
            "wm_mini_batches": 2,
            "imagination_horizon": horizon,
            "noise_bank_size": 256,
            "actor_start_update": actor_start_update,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["state", "context"],
            "backward": ["state"],
            "discriminator": ["state"],
            "expert": ["state"],
        },
        "storage_device": "cpu",
        "multi_gpu": None,
        "clip_actions": 5.0,
        "num_seed_steps_per_env": 0,
    }


@pytest.fixture(autouse=True)
def _stub_motions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("robot_rl.algorithms.fb_cpr.TrajectoryBuffer", _ExpertBuffer)


def _build(horizon: int, actor_start_update: int = 0) -> tuple[WorldModelFbCpr, _TrackingEnv]:
    env = _env()
    alg = WorldModelFbCpr.construct_algorithm(env.get_observations(), env, _cfg(horizon, actor_start_update), "cpu")
    obs = env.get_observations()
    for _ in range(30):
        with torch.inference_mode():
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    return alg, env


@pytest.mark.parametrize("horizon", [0, 4])
def test_update_trains_every_part_and_round_trips(horizon: int) -> None:
    """An update logs finite FB, critic, world-model and imagined-actor losses, and a checkpoint restores."""
    alg, env = _build(horizon)
    before = copy.deepcopy(alg.actor.state_dict())
    loss_dict, extras = alg.update()
    for key in ("Forward_Backward_Loss/fb_loss", "Actor_Loss/total_loss", "world_model/state"):
        assert key in loss_dict and torch.isfinite(loss_dict[key]).all(), (key, loss_dict)
    assert int(loss_dict["Actor_Loss/horizon"]) == horizon
    assert ("imagined_fb_reward" in extras) == (horizon > 0)
    assert any((before[k] != v).any() for k, v in alg.actor.state_dict().items())
    restored = WorldModelFbCpr.construct_algorithm(env.get_observations(), env, _cfg(horizon), "cpu")
    restored.load(copy.deepcopy(alg.save()), None, strict=True)
    for k, v in alg.world_model.state_dict().items():
        torch.testing.assert_close(restored.world_model.state_dict()[k], v)


def test_actor_imagines_from_the_start_update_at_its_interval() -> None:
    """The actor takes FB-CPR's one-step loss before ``actor_start_update`` and between imagined updates."""
    alg, _ = _build(horizon=4, actor_start_update=2)
    alg.imagined_actor_interval, alg.num_imagined = 2, 16
    horizons = [int(alg.update()[0]["Actor_Loss/horizon"]) for _ in range(5)]
    assert horizons == [0, 0, 4, 0, 4]


def test_horizon_zero_is_fb_cpr_actor_loss() -> None:
    """With no imagined steps the actor loss equals FB-CPR's, on the same batch, latents and action noise."""
    alg, _ = _build(horizon=0)
    win = alg.learner.replay_buffer.sample_transitions(BATCH)
    batch = alg._as_batch(win)
    alg.obs_normalizer(batch.observations)
    with torch.no_grad():
        batch.observations = alg._normalize(batch.observations)
        batch.next_observations = alg._normalize(batch.next_observations)
    z = alg.project_z(torch.randn(BATCH, Z_DIM))
    batch.context = z
    reference = copy.deepcopy(alg)
    torch.manual_seed(0)
    imagined, _ = alg._update_imagined_actor(win.obs[0], z, batch.next_observations, horizon=0)
    torch.manual_seed(0)
    one_step, _ = FbCpr._update_actor(reference, batch)
    torch.testing.assert_close(imagined["Actor_Loss/total_loss"], one_step["Actor_Loss/total_loss"])
