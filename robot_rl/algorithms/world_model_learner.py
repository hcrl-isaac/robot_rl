"""The world-model half of a model-based algorithm: logging, model fitting and imagined rollouts."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Callable
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.imagination import ImaginedObservation
from robot_rl.env import VecEnv, WorldModelSpec
from robot_rl.models import WorldModel
from robot_rl.storage import SequenceReplayBuffer
from robot_rl.utils import resolve_callable

GoalProvider = Callable[[torch.Tensor, torch.Tensor], TensorDict]
"""``(keys (N, K), steps (N,)) -> goal groups (N, ...)``: the goals ``steps`` env steps after each key."""

Policy = Callable[[TensorDict], torch.Tensor]
"""Raw observation groups ``(B, ...)`` -> actions ``(B, A)``; gradients reach whatever the caller trains."""

GradReducer = Callable[[Any], None]
"""Averages (across ranks) and clips the gradients of the given parameters before an optimizer step."""


@dataclass
class Rollout:
    """An imagined rollout, time-major, still attached to the policy's graph."""

    obs: list[TensorDict]
    """Every model's groups at steps ``0 .. H``, raw (unnormalized)."""

    actions: list[torch.Tensor]
    """The policy's actions at steps ``0 .. H - 1``."""

    rewards: torch.Tensor
    """Predicted rewards, ``(H, B)``."""

    dones: torch.Tensor
    """Predicted termination probabilities after each step, ``(H, B)``."""

    survival: torch.Tensor
    """Soft survival weights ``c_0 .. c_H``, ``(H + 1, B)``."""


class WorldModelLearner:
    """Owns a world model, the sequence replay it is fit on, and the imagined observations it rolls out.

    An algorithm logs every env step here, calls :meth:`fit` each update, and draws actor-training rollouts with
    :meth:`sample_starts` and :meth:`imagine` under its own policy. The env decides what the model sees (its
    :class:`WorldModelSpec`); the runner cfg decides the model and these fitting knobs.
    """

    def __init__(
        self,
        world_model: WorldModel,
        replay_buffer: SequenceReplayBuffer,
        spec: WorldModelSpec,
        imagined_obs: ImaginedObservation,
        goal_provider: GoalProvider | None = None,
        learning_rate: float = 2e-4,
        wm_window: int = 8,
        wm_batch_size: int = 2048,
        wm_mini_batches: int = 10,
        reward_loss_coef: float = 1.0,
        obs_loss_coef: float = 1.0,
        termination_loss_coef: float = 1.0,
        imagination_horizon: int = 32,
        noise_bank_size: int = 65536,
        device: str = "cpu",
    ) -> None:
        """Initialize the learner.

        Args:
            world_model: The dynamics model and its heads.
            replay_buffer: Logged trajectories.
            spec: The env's world-model spec.
            imagined_obs: Rebuilds the models' groups in imagination.
            goal_provider: The env's goal lookup by key; None reads goals along the logged trajectory.
            learning_rate: World-model learning rate.
            wm_window: Steps per autoregressive fit window.
            wm_batch_size: Windows per fit minibatch.
            wm_mini_batches: Fit gradient steps per update.
            reward_loss_coef: Weight of the reward-head loss.
            obs_loss_coef: Weight of the observation-head loss.
            termination_loss_coef: Weight of the termination-head loss.
            imagination_horizon: Imagined steps per rollout.
            noise_bank_size: Logged rows drawn per rollout batch for the sensor-noise bank.
            device: Training device.
        """
        self.device = device
        self.world_model = world_model.to(device)
        self.replay_buffer = replay_buffer
        self.spec = spec
        self.imagined_obs = imagined_obs
        self.goal_provider = goal_provider
        self.wm_window = wm_window
        self.wm_batch_size = wm_batch_size
        self.wm_mini_batches = wm_mini_batches
        self.reward_loss_coef = reward_loss_coef
        self.obs_loss_coef = obs_loss_coef
        self.termination_loss_coef = termination_loss_coef
        self.horizon = imagination_horizon
        self.noise_bank_size = noise_bank_size
        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=learning_rate)

    """
    Logging.
    """

    def add(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        row_extras: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Log one env step and update the model's normalizers.

        Args:
            obs: Observations before the step.
            actions: Executed actions.
            rewards: Rewards of the step.
            dones: All episode ends.
            extras: The env's step extras (its ``time_outs`` separate truncations from terminations).
            row_extras: Values of the buffer's declared per-row extras (e.g. a latent).
        """
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).view(-1).bool()
        terminated = dones.view(-1).bool() & ~time_outs
        self.replay_buffer.add(obs, actions, rewards, terminated, dones, row_extras)
        self.world_model.update_normalization(*self.wm_inputs(obs))

    """
    Fitting.
    """

    def fit(self, reduce: GradReducer) -> dict[str, float]:
        """Fit the world model on autoregressive replay windows rolled on the logged actions.

        Args:
            reduce: Gradient averaging and clipping applied before each step.

        Returns:
            Mean losses under ``world_model/``, or ``world_model/skipped`` when no window fits yet.
        """
        sums = {"state": 0.0, "reward": 0.0, "obs": 0.0, "termination": 0.0, "k_step_error": 0.0}
        k, done = self.wm_window, 0
        wm = self.world_model
        for _ in range(self.wm_mini_batches):
            win = self.replay_buffer.sample_windows(self.wm_batch_size, k, device=self.device)
            if win is None:
                continue
            obs = self.with_goals(win.obs, self.window_goals(win))
            state_raw, ctx_raw, goal_raw, obs_raw = self.wm_inputs(obs)
            state_n, ctx_n, goal_n = wm.normalize(state_raw, ctx_raw, goal_raw)  # (k+1, B, .)
            ctx_n = ctx_n[0]
            last_mask = win.next_valid.float()
            s = state_n[0]
            state_loss = reward_loss = obs_loss = term_loss = torch.zeros((), device=self.device)
            for j in range(k):
                s, reward, term_logit = wm.step(s, win.actions[j], ctx_n, goal_n[j])
                mask = last_mask if j == k - 1 else torch.ones_like(last_mask)
                err = (s - state_n[j + 1]).pow(2).mean(-1)
                state_loss = state_loss + (err * mask).sum() / mask.sum().clamp(min=1)
                if reward is not None:
                    reward_loss = reward_loss + nn.functional.mse_loss(reward, win.rewards[j])
                if term_logit is not None:
                    term_loss = term_loss + nn.functional.binary_cross_entropy_with_logits(
                        term_logit, win.terminated[j]
                    )
                if wm.observation is not None:
                    target = wm.obs_normalizer(obs_raw[j + 1])
                    oerr = (wm.observe_normalized(s, ctx_n, goal_n[j + 1]) - target).pow(2).mean(-1)
                    obs_loss = obs_loss + (oerr * mask).sum() / mask.sum().clamp(min=1)
            loss = (
                state_loss
                + self.reward_loss_coef * reward_loss
                + self.obs_loss_coef * obs_loss
                + self.termination_loss_coef * term_loss
            ) / k
            self.optimizer.zero_grad()
            loss.backward()
            reduce(self.world_model.parameters())
            self.optimizer.step()
            sums["state"] += state_loss.item() / k
            sums["reward"] += reward_loss.item() / k
            sums["obs"] += obs_loss.item() / k
            sums["termination"] += term_loss.item() / k
            sums["k_step_error"] += ((err * last_mask).sum() / last_mask.sum().clamp(min=1)).item()
            done += 1
        if done == 0:
            return {"world_model/skipped": 1.0}
        return {f"world_model/{key}": v / done for key, v in sums.items()}

    """
    Imagination.
    """

    def sample_starts(
        self, num: int
    ) -> tuple[TensorDict, TensorDict | None, dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
        """Draw rollout start states from replay (None when no start is available yet).

        Without goals, or with a goal provider, any logged row can start a rollout; logged goals need ``H`` steps of
        the same episode ahead of the start.

        Args:
            num: Number of rollouts.

        Returns:
            The start rows, their goals at steps ``0 .. H`` (or None), a sensor-noise bank, and the start rows'
            buffer extras.
        """
        H = self.horizon
        if not self.spec.goal_groups:
            starts, row_extras = self.replay_buffer.sample_rows(num, device=self.device)
            goals = None
        elif self.goal_provider is not None:
            starts, row_extras = self.replay_buffer.sample_rows(num, device=self.device)
            goals = self.provided_goals(starts[self.spec.goal_key_group], H)
        else:
            win = self.replay_buffer.sample_windows(num, max(H, 1), device=self.device, require_next=True)
            if win is None:
                return None
            starts, row_extras = win.obs[0], {k: v[0] for k, v in win.extras.items()}
            goals = win.obs.select(*self.spec.goal_groups)[: H + 1] if self.spec.goal_groups else None
        return starts, goals, self.noise_bank(), row_extras

    def noise_bank(self) -> dict[str, torch.Tensor]:
        """Sensor-noise rows per group, drawn from logged rows (see :meth:`ImaginedObservation.noise_bank`)."""
        bank_rows, _ = self.replay_buffer.sample_rows(self.noise_bank_size, device=self.device)
        return self.imagined_obs.noise_bank(bank_rows)

    def imagine(
        self, policy: Policy, starts: TensorDict, goals: TensorDict | None, noise_bank: dict[str, torch.Tensor]
    ) -> Rollout:
        """Roll ``policy`` ``H`` steps through the world model (its parameters frozen; the graph reaches the policy).

        Args:
            policy: Maps the raw imagined groups to actions.
            starts: Logged rows to start from, ``(B, ...)``.
            goals: The goal groups at steps ``0 .. H``, ``(H + 1, B, ...)`` (None without goals).
            noise_bank: Sensor-noise rows per group, from :meth:`noise_bank`.

        Returns:
            The rollout, attached to the policy's graph.
        """
        wm, H = self.world_model, self.horizon
        B = starts.batch_size[0]
        if goals is not None:
            starts = self.with_goals(starts, goals[0])
        state_raw, ctx_raw, _, _ = self.wm_inputs(starts)
        goal_all = torch.cat([goals[g] for g in self.spec.goal_groups], dim=-1) if goals is not None else None
        s, ctx_n, _ = wm.normalize(state_raw, ctx_raw, None)
        goal_n = wm.goal_normalizer(goal_all) if goal_all is not None else s.new_zeros(H + 1, B, 0)
        bank_size = next(iter(noise_bank.values())).shape[0] if noise_bank else 1

        wm.requires_grad_(False)
        obs = starts.select(*self.imagined_obs.groups)
        obs_seq, actions, rewards, dones = [obs], [], [], []
        survival = [torch.ones(B, device=self.device)]
        for t in range(H):
            action = policy(obs)
            s, reward, term_logit = wm.step(s, action, ctx_n, goal_n[t])
            done = torch.sigmoid(term_logit) if term_logit is not None else torch.zeros_like(survival[-1])
            idx = torch.randint(0, bank_size, (B,), device=self.device)
            obs_bar = wm.observe(s, ctx_n, goal_n[t + 1]) if self.imagined_obs.reads_obs_head else None
            obs = self.imagined_obs.step(
                obs,
                wm.denormalize_state(s),
                ctx_raw,
                obs_bar,
                action,
                goals[t + 1] if goals is not None else None,
                {g: b[idx] for g, b in noise_bank.items()},
            )
            obs_seq.append(obs)
            actions.append(action)
            rewards.append(reward if reward is not None else torch.zeros_like(done))
            dones.append(done)
            survival.append(survival[-1] * (1.0 - done))
        wm.requires_grad_(True)
        empty = torch.zeros(0, B, device=self.device)
        return Rollout(
            obs_seq,
            actions,
            torch.stack(rewards) if rewards else empty,
            torch.stack(dones) if dones else empty,
            torch.stack(survival),
        )

    """
    Helpers.
    """

    def wm_inputs(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Split observations into the world model's raw state, context, goal and noise-free observation."""
        spec = self.spec
        ctx = obs[spec.context_group] if spec.context_group else None
        goal = torch.cat([obs[g] for g in spec.goal_groups], dim=-1) if spec.goal_groups else None
        obs_bar = obs[spec.obs_group] if spec.obs_group else None
        return obs[spec.state_group], ctx, goal, obs_bar

    def provided_goals(self, keys: torch.Tensor, steps: int) -> TensorDict:
        """Return the goal groups at ``0 .. steps`` env steps after each key, ``(steps + 1, B, ...)``."""
        B = keys.shape[0]
        offsets = torch.arange(steps + 1, device=keys.device).repeat_interleave(B)
        goals = self.goal_provider(keys.repeat(steps + 1, 1), offsets)  # type: ignore[misc]
        return goals.reshape(steps + 1, B)

    def window_goals(self, win: SequenceReplayBuffer.Window) -> TensorDict | None:
        """Return the goal groups along a replay window: from the provider when there is one, else as logged."""
        if not self.spec.goal_groups:
            return None
        if self.goal_provider is not None:
            return self.provided_goals(win.obs[self.spec.goal_key_group][0], win.actions.shape[0])
        return win.obs.select(*self.spec.goal_groups)

    @staticmethod
    def with_goals(obs: TensorDict, goals: TensorDict | None) -> TensorDict:
        """Observations with their goal groups set (they are not stored when a provider supplies them)."""
        if goals is None:
            return obs
        obs = obs.clone(recurse=False)
        obs.update(goals)
        return obs

    def state_dict(self) -> dict[str, Any]:
        """Return the model and optimizer states, keyed as checkpoints store them."""
        return {
            "world_model_state_dict": self.world_model.state_dict(),
            "wm_optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, loaded: dict[str, Any], optimizer: bool = True, strict: bool = True) -> None:
        """Load the model (and optionally optimizer) state from a checkpoint dict.

        Args:
            loaded: The checkpoint dict.
            optimizer: Also restore the optimizer.
            strict: Require every model key to match.
        """
        self.world_model.load_state_dict(loaded["world_model_state_dict"], strict=strict)
        if optimizer:
            self.optimizer.load_state_dict(loaded["wm_optimizer_state_dict"])

    @staticmethod
    def build(
        obs: TensorDict,
        env: VecEnv,
        wm_cfg: dict[str, Any],
        model_groups: list[str],
        capacity_per_env: int,
        storage_device: str,
        device: str,
        obs_dtypes: dict[str, str] | None = None,
        row_extras: dict[str, int] | None = None,
        **learner_kwargs: Any,
    ) -> WorldModelLearner:
        """Build the learner from the env's world-model spec (``env.world_model_spec``) and a model cfg dict.

        With a ``goal_key_group`` in the spec, goals come from ``env.world_model_goals`` and are checked against the
        env's own goal observations once here.

        Args:
            obs: An observation sample.
            env: The env, carrying ``world_model_spec`` (and ``world_model_goals`` with a goal key).
            wm_cfg: The world-model cfg (``class_name`` plus constructor kwargs).
            model_groups: Every group a trained model reads, rebuilt in imagination.
            capacity_per_env: Replay steps kept per env.
            storage_device: Replay storage device.
            device: Training device.
            obs_dtypes: Per-group replay dtypes.
            row_extras: Per-row algorithm tensors to log, name to width.
            **learner_kwargs: Fitting and imagination knobs of :class:`WorldModelLearner`.

        Returns:
            The learner.
        """
        spec: WorldModelSpec | None = getattr(env, "world_model_spec", None)
        if spec is None:
            raise ValueError("A world-model algorithm needs the env's world_model_spec (the env cfg's world_model).")
        wm_cfg = dict(wm_cfg)
        wm_class: type[WorldModel] = resolve_callable(wm_cfg.pop("class_name", "WorldModel"))  # type: ignore

        goal_provider = None
        if spec.goal_key_group is not None:
            goal_provider = env.world_model_goals  # type: ignore[attr-defined]
            keys = obs[spec.goal_key_group]
            provided = goal_provider(keys, torch.zeros(keys.shape[0], dtype=torch.long, device=keys.device))
            for g in spec.goal_groups:
                gap = (provided[g] - obs[g]).abs().max().item()
                if gap > 1e-4:
                    raise ValueError(f"env.world_model_goals disagrees with the '{g}' observation by {gap:.2e}.")

        def width(group: str | None) -> int:
            return obs[group].shape[-1] if group else 0

        world_model = wm_class(
            state_dim=width(spec.state_group),
            action_dim=env.num_actions,
            context_dim=width(spec.context_group),
            goal_dim=sum(width(g) for g in spec.goal_groups),
            obs_dim=width(spec.obs_group),
            **wm_cfg,
        )
        imagined_obs = ImaginedObservation(spec, obs, model_groups)
        if world_model.observation is None and imagined_obs.reads_obs_head:
            raise ValueError("A model reads world-model observation terms, but the spec names no obs_group.")

        # provided goals are recomputed from their key, so only the key is stored
        stored = [spec.state_group, *imagined_obs.groups, spec.context_group, spec.obs_group]
        if goal_provider is not None:
            stored = [g for g in stored if g not in spec.goal_groups] + [spec.goal_key_group]
        else:
            stored += spec.goal_groups
        replay = SequenceReplayBuffer(
            env.num_envs,
            capacity_per_env,
            obs,
            [g for g in stored if g],
            env.num_actions,
            device=storage_device,
            obs_dtypes=obs_dtypes,
            extras=row_extras,
        )
        return WorldModelLearner(
            world_model, replay, spec, imagined_obs, goal_provider=goal_provider, device=device, **learner_kwargs
        )
