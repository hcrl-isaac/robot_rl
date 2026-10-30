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
    ``state``, the exogenous ``goal`` window, the ``context`` (the gain) and the ``goal_key`` (time and phase) from
    which :meth:`world_model_goals` recomputes the goal.
    """

    def __init__(self, num_envs: int = 32, noise: float = 0.05, seed: int = 0, provide_goals: bool = True) -> None:
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
            goal_key_group="goal_key" if provide_goals else None,
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

    @staticmethod
    def _goal(t: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sin(DT * (t.float()[:, None] + i) + phase) for i in range(1, FUTURE + 1)], dim=-1)

    def world_model_goals(self, keys: torch.Tensor, steps: torch.Tensor) -> TensorDict:
        goal = self._goal(keys[:, 0] + steps, keys[:, 1:])
        return TensorDict({"goal": goal}, batch_size=[keys.shape[0]])

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
        state = torch.cat([self.pos, self.vel], dim=-1)
        groups = {
            "policy": policy,
            "obs_bar": state.clone(),
            "state": state,
            "goal": self._goal(t, self.phase),
            "goal_key": torch.cat([t.float()[:, None], self.phase], dim=-1),
            "context": self.gain,
        }
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


def _make_cfg(critic_mode: str = "replay", critic_obs: list[str] | None = None) -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 1000,
        "log_interval": 1000,
        "num_seed_steps_per_env": 4,
        "obs_groups": {"actor": ["policy", "goal"], "critic": critic_obs or ["state", "goal", "context"]},
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


def _collect(alg: WorldModelAC, env: _TrackingEnv, steps: int) -> TensorDict:
    obs = env.get_observations()
    for _ in range(steps):
        with torch.inference_mode():
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    return obs


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
    """Under random terminations and time-outs, a window never crosses a reset; ``require_next`` also holds row L."""
    torch.manual_seed(0)
    num_envs, capacity = 6, 37
    obs = TensorDict({"t": torch.zeros(num_envs, 2)}, batch_size=[num_envs])
    buf = SequenceReplayBuffer(num_envs, capacity, obs, ["t"], 1)
    episode, step = torch.zeros(num_envs), torch.zeros(num_envs)
    for _ in range(300):  # wraps the ring several times
        terminated = torch.rand(num_envs) < 0.08
        dones = terminated | (torch.rand(num_envs) < 0.05)
        row = TensorDict({"t": torch.stack([episode, step], dim=-1)}, batch_size=[num_envs])
        buf.add(row, torch.zeros(num_envs, 1), torch.zeros(num_envs), terminated, dones)
        episode, step = torch.where(dones, episode + 1, episode), torch.where(dones, 0.0, step + 1)
    for require_next in (False, True):
        win = buf.sample_windows(2000, 5, require_next=require_next)
        assert win is not None
        ep, st = win.obs["t"][..., 0], win.obs["t"][..., 1]
        assert (ep[:5] == ep[:1]).all() and (st[:5] - st[:1] == torch.arange(5)[:, None]).all()
        same_next = (ep[5] == ep[0]) & (st[5] - st[0] == 5)
        assert (same_next == win.next_valid).all()
        assert (win.next_valid | (win.terminated[-1] > 0)).all()
        assert (win.terminated[:-1] == 0).all()
        if require_next:
            assert win.next_valid.all()
    buf.mark_episode_start()
    buf.add(row, torch.zeros(num_envs, 1), torch.zeros(num_envs), torch.zeros(num_envs), torch.zeros(num_envs))
    assert (buf.episode_step[(buf.pos - 1) % capacity] == 0).all()
    assert SequenceReplayBuffer(num_envs, capacity, obs, ["t"], 1).sample_windows(4, 5) is None


def test_imagined_observation_rolls_history_and_reproduces_noise() -> None:
    """History terms shift by one frame, the action term takes the action, and the bank holds the sensor noise."""
    env = _TrackingEnv(num_envs=4)
    obs = env.get_observations()
    io = ImaginedObservation(env.world_model_spec, obs, ["policy", "goal", "state", "context"])
    bank = io.noise_bank(obs)
    # newest noisy frame minus the noise-free copy is exactly the env's sensor noise
    newest = torch.cat([obs["policy"][:, 2:4], obs["policy"][:, 6:8]], dim=-1)
    assert torch.allclose(bank["policy"], newest - obs["obs_bar"])
    action = torch.ones(4, 2)
    current = obs.select(*io.groups)
    nxt = io.step(current, obs["state"], obs["context"], obs["obs_bar"], action, obs.select("goal"), bank)
    assert torch.allclose(nxt["policy"][:, 0:2], obs["policy"][:, 2:4])  # oldest pos frame <- previous newest
    assert torch.allclose(nxt["policy"][:, 2:4], newest[:, 0:2])
    assert torch.allclose(nxt["policy"][:, 10:12], action)
    assert torch.equal(nxt["state"], obs["state"]) and torch.equal(nxt["context"], obs["context"])


@pytest.mark.parametrize(
    "critic_mode,provide_goals,critic_obs",
    [
        ("replay", True, None),
        ("imagination", True, None),
        ("replay", False, None),
        ("imagination", False, ["policy", "state", "goal"]),  # a critic over a history group, as FB-CPR's
    ],
)
def test_update_runs_and_logs(critic_mode: str, provide_goals: bool, critic_obs: list[str] | None) -> None:
    """One update logs finite losses for each critic mode and goal source, and a checkpoint round-trips."""
    env = _TrackingEnv(provide_goals=provide_goals)
    cfg = _make_cfg(critic_mode, critic_obs)
    alg = WorldModelAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
    assert ("goal" in alg.replay_buffer.groups) != provide_goals
    obs = _collect(alg, env, 40)
    logs = alg.update()
    for key in ("world_model/state", "world_model/k_step_error", "actor/imagined_return", "critic/loss"):
        assert key in logs and torch.isfinite(torch.tensor(logs[key])), (key, logs)
    restored = WorldModelAC.construct_algorithm(obs, env, cfg, device="cpu")
    assert restored.load(copy.deepcopy(alg.save()))


def test_updates_skip_until_windows_exist() -> None:
    """With fewer rows than any window, an update is skipped and logged instead of raising."""
    env = _TrackingEnv(provide_goals=False)
    alg = WorldModelAC.construct_algorithm(env.get_observations(), env, _make_cfg("replay"), device="cpu")
    _collect(alg, env, 3)
    logs = alg.update()
    assert logs["world_model/skipped"] == 1.0 and logs["actor/skipped"] == 1.0 and logs["critic/skipped"] == 1.0


def test_eval_rolls_out_and_restores_training_mode() -> None:
    """The recorder's rollout steps the env max_steps times and leaves the models training."""
    env = _TrackingEnv()
    alg = WorldModelAC.construct_algorithm(env.get_observations(), env, _make_cfg(), device="cpu")
    steps = []
    step = env.step
    env.step = lambda actions: (steps.append(actions), step(actions))[1]
    assert alg.eval(env, max_steps=6, stochastic=True, action_repeat=2) == []
    assert len(steps) == 6 and torch.equal(steps[0], steps[1]) and alg.actor.training


def test_provider_must_match_the_goal_observation() -> None:
    """A goal provider that disagrees with the env's goal observation is caught at construction."""
    env = _TrackingEnv()
    env.world_model_goals = lambda keys, steps: TensorDict({"goal": torch.zeros(len(keys), 2 * FUTURE)}, [len(keys)])
    with pytest.raises(ValueError, match="disagrees"):
        WorldModelAC.construct_algorithm(env.get_observations(), env, _make_cfg(), device="cpu")


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
