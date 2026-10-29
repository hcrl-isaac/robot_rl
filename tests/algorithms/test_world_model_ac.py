"""CPU tests for the world-model actor-critic (a toy tracking env, no Isaac Sim)."""

from __future__ import annotations

import copy
import torch
from collections.abc import Sequence
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import WorldModelAC
from robot_rl.algorithms.imagination import ImaginedObservation
from robot_rl.env import ObsTermLayout, WorldModelSpec
from robot_rl.runners import OffPolicyRunner
from robot_rl.storage import SequenceReplayBuffer

DT, MAX_EP_LEN, FUTURE = 0.1, 50, 3


class _TrackingEnv:
    """A 2-D point mass with a per-episode actuator gain, tracking a sinusoid it reads ``FUTURE`` frames ahead.

    Groups mirror a tracking task: a noisy ``policy`` history, its noise-free single-frame copy ``obs_bar``, the
    ``state``, the exogenous ``goal`` window and the ``context`` (the gain).
    """

    def __init__(self, num_envs: int = 32, noise: float = 0.05, seed: int = 0) -> None:
        self.num_envs, self.num_actions, self.noise = num_envs, 2, noise
        self.max_episode_length = MAX_EP_LEN
        self.device = "cpu"
        self.cfg = {}
        self.common_step_counter = 0
        self.gen = torch.Generator().manual_seed(seed)
        n = num_envs
        self.pos, self.vel = torch.zeros(n, 2), torch.zeros(n, 2)
        self.gain, self.phase = torch.ones(n, 1), torch.zeros(n, 2)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.last_action = torch.zeros(n, 2)
        self.history = torch.zeros(n, 2, 6)
        self.world_model_spec = WorldModelSpec(
            state_group="state",
            obs_group="obs_bar",
            context_group="context",
            goal_groups=["goal"],
            action_terms=["last_action"],
            layouts={
                "policy": [
                    ObsTermLayout("pos", 2, 2),
                    ObsTermLayout("vel", 2, 2),
                    ObsTermLayout("last_action", 2, 2),
                ],
                "obs_bar": [ObsTermLayout("pos", 2), ObsTermLayout("vel", 2)],
            },
        )
        self._reset(torch.arange(n))

    def _ref(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(DT * t.float()[:, None] + self.phase)

    def _reset(self, ids: torch.Tensor) -> None:
        k = len(ids)
        self.phase[ids] = torch.rand(k, 2, generator=self.gen) * 6.28
        self.gain[ids] = 0.5 + torch.rand(k, 1, generator=self.gen)
        self.episode_length_buf[ids] = 0
        self.pos[ids] = self._ref(self.episode_length_buf)[ids]
        self.vel[ids] = 0.0
        self.last_action[ids] = 0.0
        frame = self._frame()[ids]
        self.history[ids] = frame[:, None, :].expand(-1, 2, -1).clone()

    def _frame(self) -> torch.Tensor:
        noisy = torch.cat([self.pos, self.vel], dim=-1) + self.noise * torch.randn(self.num_envs, 4, generator=self.gen)
        return torch.cat([noisy, self.last_action], dim=-1)

    def get_observations(self) -> TensorDict:
        h = self.history
        policy = torch.cat([h[:, :, 0:2].flatten(1), h[:, :, 2:4].flatten(1), h[:, :, 4:6].flatten(1)], dim=-1)
        t = self.episode_length_buf
        goal = torch.cat([self._ref(t + i) for i in range(1, FUTURE + 1)], dim=-1)
        state = torch.cat([self.pos, self.vel], dim=-1)
        groups = {"policy": policy, "obs_bar": state.clone(), "state": state, "goal": goal, "context": self.gain}
        return TensorDict(groups, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = actions.clamp(-5, 5)
        self.vel = 0.9 * self.vel + DT * 4.0 * self.gain * actions
        self.pos = self.pos + DT * self.vel
        self.episode_length_buf += 1
        self.common_step_counter += 1
        err = (self.pos - self._ref(self.episode_length_buf)).norm(dim=-1)
        rewards = torch.exp(-(err**2) / 0.05) - 0.01 * actions.pow(2).sum(-1)
        terminated = err > 1.0
        time_outs = self.episode_length_buf >= MAX_EP_LEN
        dones = terminated | time_outs
        self.last_action = actions
        self.history = torch.cat([self.history[:, 1:], self._frame()[:, None]], dim=1)
        if dones.any():
            self._reset(dones.nonzero(as_tuple=True)[0])
        return self.get_observations(), rewards, dones.float(), {"time_outs": time_outs.float()}

    def apply(self, mode: str, env_ids: Sequence[int] | None = None) -> None:
        pass

    @property
    def unwrapped(self) -> _TrackingEnv:
        return self


def _make_cfg(critic_mode: str = "replay") -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 1000,
        "log_interval": 1000,
        "num_seed_steps_per_env": 4,
        "obs_groups": {"actor": ["policy", "goal"], "critic": ["state", "goal", "context"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [64, 64], "obs_normalization": True},
        "critic": {"class_name": "MLPModel", "hidden_dims": [64, 64], "obs_normalization": True},
        "algorithm": {
            "class_name": "WorldModelAC",
            "world_model": {"hidden_dims": [128, 128], "head_hidden_dims": [64], "condition_on_goal": True},
            "replay_buffer_size": 32 * 200,
            "learning_rate": 3e-3,
            "wm_window": 4,
            "wm_batch_size": 256,
            "wm_mini_batches": 4,
            "imagination_horizon": 8,
            "num_imagined": 256,
            "critic_mode": critic_mode,
            "critic_window": 4,
            "critic_batch_size": 256,
            "critic_epochs": 2,
            "critic_mini_batches": 2,
            "critic_tau": 0.05,
            "noise_bank_size": 1024,
            "actor_grad_clip_value": None,
        },
    }


def _evaluate(policy: torch.nn.Module | None, steps: int = MAX_EP_LEN) -> float:
    """Mean per-step reward of a deterministic policy (``None`` = zero action) on fresh episodes."""
    env = _TrackingEnv(num_envs=64, seed=123)
    obs, total = env.get_observations(), 0.0
    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs) if policy is not None else torch.zeros(env.num_envs, 2)
            obs, rewards, _, _ = env.step(actions)
            total += rewards.mean().item()
    return total / steps


def test_replay_windows_stay_inside_one_episode() -> None:
    """Sampled windows never cross a reset, and ``require_next`` keeps the row after the window in-episode."""
    env = _TrackingEnv(num_envs=8)
    obs = env.get_observations()
    buf = SequenceReplayBuffer(8, 40, obs, ["state"], 2)
    for _ in range(120):  # wraps the ring three times
        buf.add(
            env.get_observations(), torch.zeros(8, 2), torch.zeros(8), torch.zeros(8), env.step(torch.zeros(8, 2))[2]
        )
    for require_next in (False, True):
        win = buf.sample_windows(64, 6, require_next=require_next)
        steps = win.obs["state"]  # positions only rise smoothly within an episode
        assert steps.shape == (7, 64, 4)
        if require_next:
            assert win.next_valid.all()
        assert (win.terminated[:-1] == 0).all()


def test_imagined_observation_rolls_history_and_reproduces_noise() -> None:
    """History terms shift by one frame, the action term takes the action, and the bank holds the sensor noise."""
    env = _TrackingEnv(num_envs=4)
    obs = env.get_observations()
    io = ImaginedObservation(env.world_model_spec, obs, ["policy", "goal"])
    bank = io.noise_bank(obs)
    # newest noisy frame minus the noise-free copy is exactly the env's sensor noise
    newest = torch.cat([obs["policy"][:, 2:4], obs["policy"][:, 6:8]], dim=-1)
    assert torch.allclose(bank["policy"], newest - obs["obs_bar"])
    action = torch.ones(4, 2)
    nxt = io.step(obs.select("policy", "goal"), obs["obs_bar"], action, obs.select("goal"), bank)
    assert torch.allclose(nxt["policy"][:, 0:2], obs["policy"][:, 2:4])  # oldest pos frame <- previous newest
    assert torch.allclose(nxt["policy"][:, 2:4], newest[:, 0:2])
    assert torch.allclose(nxt["policy"][:, 10:12], action)


@pytest.mark.parametrize("critic_mode", ["replay", "imagination"])
def test_update_runs_and_logs(critic_mode: str) -> None:
    """One update in each critic mode logs finite losses, and a checkpoint round-trips."""
    env = _TrackingEnv()
    obs = env.get_observations()
    alg = WorldModelAC.construct_algorithm(obs, env, _make_cfg(critic_mode), device="cpu")
    for _ in range(40):
        with torch.inference_mode():
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    logs = alg.update()
    for key in ("world_model/state", "world_model/k_step_error", "actor/imagined_return", "critic/loss"):
        assert key in logs and torch.isfinite(torch.tensor(logs[key]))
    restored = WorldModelAC.construct_algorithm(obs, env, _make_cfg(critic_mode), device="cpu")
    assert restored.load(copy.deepcopy(alg.save()))


@pytest.mark.parametrize("critic_mode", ["replay", "imagination"])
def test_learns_to_track_through_the_world_model(tmp_path: Path, critic_mode: str) -> None:
    """Through the runner, the actor trained in imagination beats both its initialization and a do-nothing policy."""
    torch.manual_seed(0)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))  # small nets: more threads only oversubscribe
    env = _TrackingEnv()
    runner = OffPolicyRunner(env, _make_cfg(critic_mode), log_dir=str(tmp_path), device="cpu")  # type: ignore[arg-type]
    before = _evaluate(runner.alg.get_policy())
    runner.learn(num_learning_iterations=150)
    after = _evaluate(runner.alg.get_policy())
    idle = _evaluate(None)
    torch.set_num_threads(threads)
    assert after > idle + 0.1 and after > before + 0.1, (before, after, idle)
