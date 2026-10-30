"""Actor-critic trained through a learned, differentiable world model (SuperMimic-style BPTT)."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections.abc import Callable
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.imagination import ImaginedObservation
from robot_rl.env import VecEnv, WorldModelSpec
from robot_rl.models import MLPModel, WorldModel
from robot_rl.modules import TargetNetwork
from robot_rl.storage import SequenceReplayBuffer
from robot_rl.utils import resolve_callable, resolve_obs_groups

GoalProvider = Callable[[torch.Tensor, torch.Tensor], TensorDict]
"""``(keys (N, K), steps (N,)) -> goal groups (N, ...)``: the goals ``steps`` env steps after each key."""


@dataclass
class Rollout:
    """An imagined rollout, time-major, still attached to the actor's graph."""

    obs: list[TensorDict]
    """Every model's groups at steps ``0 .. H``."""

    rewards: torch.Tensor
    """Predicted rewards, ``(H, B)``."""

    dones: torch.Tensor
    """Predicted termination probabilities after each step, ``(H, B)``."""

    survival: torch.Tensor
    """Soft survival weights ``c_0 .. c_H``, ``(H + 1, B)``."""


class WorldModelAC:
    """Deterministic actor improved by backpropagation through imagined world-model rollouts.

    Each update fits the world model to ``wm_window``-step autoregressive replay windows, then rolls the actor
    ``imagination_horizon`` steps through the frozen model from replay start states and ascends the imagined return
    ``sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)`` with the soft survival weight ``c_{t+1} = c_t (1 - d_{t+1})``.
    The value critic is fit either on real replay windows (``critic_mode="replay"``: TD(lambda) with a target network
    on logged actions, so it estimates the behaviour policy's value) or on the imagined rollout (``"imagination"``:
    survival-weighted lambda-returns without a target network, as in SuperMimic).

    Reference:
        - Liao et al. "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model."
          arXiv preprint arXiv:2610.28480 (2026).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        world_model: WorldModel,
        replay_buffer: SequenceReplayBuffer,
        spec: WorldModelSpec,
        imagined_obs: ImaginedObservation,
        critic_groups: list[str],
        goal_provider: GoalProvider | None = None,
        learning_rate: float = 2e-4,
        actor_betas: tuple[float, float] = (0.9, 0.95),
        lr_decay_updates: int = 0,
        lr_final_ratio: float = 0.03,
        wm_window: int = 8,
        wm_batch_size: int = 2048,
        wm_mini_batches: int = 10,
        reward_loss_coef: float = 1.0,
        obs_loss_coef: float = 1.0,
        termination_loss_coef: float = 1.0,
        imagination_horizon: int = 32,
        num_imagined: int = 4096,
        gamma: float = 0.99,
        lam: float = 0.95,
        action_noise_std: float = 0.1,
        updates_per_iteration: int = 1,
        actor_start_update: int = 0,
        critic_mode: str = "replay",
        critic_window: int = 16,
        critic_batch_size: int = 4096,
        critic_epochs: int = 5,
        critic_mini_batches: int = 4,
        critic_tau: float = 0.005,
        max_grad_norm: float = 1.0,
        actor_grad_clip_value: float | None = 0.1,
        noise_bank_size: int = 65536,
        tf32: bool = True,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            actor: Deterministic policy over the deployable groups.
            critic: Value function over groups the world model can rebuild.
            world_model: The dynamics model and its heads.
            replay_buffer: Logged trajectories.
            spec: The env's world-model spec.
            imagined_obs: Rebuilds the actor's and critic's groups in imagination.
            critic_groups: The critic's groups.
            goal_provider: The env's goal lookup by key; None reads goals along the logged trajectory.
            learning_rate: Learning rate of every network.
            actor_betas: Adam betas of the actor.
            lr_decay_updates: Updates of cosine decay to ``lr_final_ratio``; 0 keeps the rate constant.
            lr_final_ratio: Learning-rate floor relative to ``learning_rate``.
            wm_window: Steps per autoregressive world-model fit window.
            wm_batch_size: Windows per world-model minibatch.
            wm_mini_batches: World-model gradient steps per update.
            reward_loss_coef: Weight of the reward-head loss.
            obs_loss_coef: Weight of the observation-head loss.
            termination_loss_coef: Weight of the termination-head loss.
            imagination_horizon: Imagined steps the actor backpropagates through.
            num_imagined: Imagined rollouts per actor update.
            gamma: Discount.
            lam: Lambda of the critic targets.
            action_noise_std: Std of the Gaussian exploration noise during collection.
            updates_per_iteration: Updates per runner iteration (sets the update-to-data ratio).
            actor_start_update: Updates that fit only the world model before the actor and critic start.
            critic_mode: ``"replay"`` or ``"imagination"`` (see the class docstring).
            critic_window: Steps per replay-critic window (``"replay"`` only).
            critic_batch_size: Windows per replay-critic update (``"replay"`` only).
            critic_epochs: Passes over the critic targets per update.
            critic_mini_batches: Minibatches per critic pass.
            critic_tau: Polyak rate of the replay critic's target (``"replay"`` only).
            max_grad_norm: Global gradient-norm clip of every network.
            actor_grad_clip_value: Element-wise actor gradient clip after the norm clip; None disables it.
            noise_bank_size: Logged rows drawn per update for the sensor-noise bank.
            tf32: Allow TF32 matmuls (a process-wide torch setting).
            device: Training device.
            multi_gpu_cfg: Multi-GPU settings, or None.
            **kwargs: Runner-level cfg entries the algorithm does not use.
        """
        if critic_mode not in ("replay", "imagination"):
            raise ValueError(f"critic_mode must be 'replay' or 'imagination', got '{critic_mode}'.")
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1
        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.world_model = world_model.to(device)
        self.critic_target = TargetNetwork(self.critic, critic_tau).to(device) if critic_mode == "replay" else None
        self.replay_buffer = replay_buffer
        self.spec = spec
        self.imagined_obs = imagined_obs
        self.critic_groups = critic_groups
        self.goal_provider = goal_provider

        self.learning_rate = learning_rate
        self.lr_decay_updates = lr_decay_updates
        self.lr_final_ratio = lr_final_ratio
        self.wm_window = wm_window
        self.wm_batch_size = wm_batch_size
        self.wm_mini_batches = wm_mini_batches
        self.reward_loss_coef = reward_loss_coef
        self.obs_loss_coef = obs_loss_coef
        self.termination_loss_coef = termination_loss_coef
        self.horizon = imagination_horizon
        self.num_imagined = num_imagined
        self.gamma = gamma
        self.lam = lam
        self.action_noise_std = action_noise_std
        self.updates_per_iteration = updates_per_iteration
        self.actor_start_update = actor_start_update
        self.critic_mode = critic_mode
        self.critic_window = critic_window
        self.critic_batch_size = critic_batch_size
        self.critic_epochs = critic_epochs
        self.critic_mini_batches = critic_mini_batches
        self.max_grad_norm = max_grad_norm
        self.actor_grad_clip_value = actor_grad_clip_value
        self.noise_bank_size = noise_bank_size

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, betas=tuple(actor_betas))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.optimizers = [self.wm_optimizer, self.actor_optimizer, self.critic_optimizer]

        self.update_step = 0
        self._obs: TensorDict | None = None
        self._actions: torch.Tensor | None = None

    """
    Collection.
    """

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Act with the deterministic actor plus Gaussian exploration noise (the executed action is logged)."""
        self._obs = obs
        actions = self.actor(obs)
        if self.action_noise_std > 0:
            actions = actions + self.action_noise_std * torch.randn_like(actions)
        self._actions = actions
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Log the step and update the observation normalizers."""
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).view(-1).bool()
        terminated = dones.view(-1).bool() & ~time_outs
        self.replay_buffer.add(self._obs, self._actions, rewards, terminated, dones)  # type: ignore[arg-type]
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs.select(*self.critic_groups))
        self.world_model.update_normalization(*self._wm_inputs(obs))

    """
    Updates.
    """

    def update(self) -> dict[str, float]:
        """Run ``updates_per_iteration`` rounds of world-model fit, actor update and critic fit."""
        logs: dict[str, list[float]] = {}
        for _ in range(self.updates_per_iteration):
            self._set_learning_rate()
            out = self._update_world_model()
            if self.update_step >= self.actor_start_update:
                rollout = self._imagine_from_replay()
                if rollout is None:
                    out["actor/skipped"] = 1.0
                else:
                    out.update(self._update_actor(rollout))
                    if self.critic_mode == "imagination":
                        out.update(self._update_imagined_critic(rollout))
                if self.critic_mode == "replay":
                    out.update(self._update_replay_critic())
            self.update_step += 1
            for k, v in out.items():
                logs.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in logs.items()}

    def _update_world_model(self) -> dict[str, float]:
        """Fit the world model on autoregressive replay windows rolled on the logged actions."""
        sums = {"state": 0.0, "reward": 0.0, "obs": 0.0, "termination": 0.0, "k_step_error": 0.0}
        k, done = self.wm_window, 0
        wm = self.world_model
        for _ in range(self.wm_mini_batches):
            win = self.replay_buffer.sample_windows(self.wm_batch_size, k, device=self.device)
            if win is None:
                continue
            obs = self._with_goals(win.obs, self._window_goals(win))
            state_raw, ctx_raw, goal_raw, obs_raw = self._wm_inputs(obs)
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
            self.wm_optimizer.zero_grad()
            loss.backward()
            self._reduce_and_clip(self.world_model.parameters())
            self.wm_optimizer.step()
            sums["state"] += state_loss.item() / k
            sums["reward"] += reward_loss.item() / k
            sums["obs"] += obs_loss.item() / k
            sums["termination"] += term_loss.item() / k
            sums["k_step_error"] += ((err * last_mask).sum() / last_mask.sum().clamp(min=1)).item()
            done += 1
        if done == 0:
            return {"world_model/skipped": 1.0}
        return {f"world_model/{key}": v / done for key, v in sums.items()}

    def _imagine_from_replay(self) -> Rollout | None:
        """Draw start states from replay and imagine ``H`` steps (None when no start is available yet).

        With a goal provider any logged row can start a rollout; otherwise a start needs ``H`` logged steps of goals
        ahead of it in the same episode.
        """
        H = self.horizon
        if self.goal_provider is not None:
            starts, _ = self.replay_buffer.sample_rows(self.num_imagined, device=self.device)
            goals = self._provided_goals(starts[self.spec.goal_key_group], H)
        else:
            win = self.replay_buffer.sample_windows(self.num_imagined, H, device=self.device, require_next=True)
            if win is None:
                return None
            starts = win.obs[0]
            goals = win.obs.select(*self.spec.goal_groups) if self.spec.goal_groups else None
        bank_rows, _ = self.replay_buffer.sample_rows(self.noise_bank_size, device=self.device)
        return self.imagine(starts, goals, self.imagined_obs.noise_bank(bank_rows))

    def imagine(self, starts: TensorDict, goals: TensorDict | None, noise_bank: dict[str, torch.Tensor]) -> Rollout:
        """Roll the actor ``H`` steps through the world model (its parameters frozen; the graph reaches the actor).

        Args:
            starts: Logged rows to start from, ``(B, ...)``.
            goals: The goal groups at steps ``0 .. H``, ``(H + 1, B, ...)`` (None without goals).
            noise_bank: Sensor-noise rows per group, from :meth:`ImaginedObservation.noise_bank`.

        Returns:
            The rollout, attached to the actor's graph.
        """
        wm, H = self.world_model, self.horizon
        B = starts.batch_size[0]
        if goals is not None:
            starts = self._with_goals(starts, goals[0])
        state_raw, ctx_raw, _, _ = self._wm_inputs(starts)
        goal_all = torch.cat([goals[g] for g in self.spec.goal_groups], dim=-1) if goals is not None else None
        s, ctx_n, _ = wm.normalize(state_raw, ctx_raw, None)
        goal_n = wm.goal_normalizer(goal_all) if goal_all is not None else s.new_zeros(H + 1, B, 0)
        bank_size = next(iter(noise_bank.values())).shape[0] if noise_bank else 1

        wm.requires_grad_(False)
        obs = starts.select(*self.imagined_obs.groups)
        obs_seq, rewards, dones = [obs], [], []
        survival = [torch.ones(B, device=self.device)]
        for t in range(H):
            action = self.actor(obs)
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
            rewards.append(reward if reward is not None else torch.zeros_like(done))
            dones.append(done)
            survival.append(survival[-1] * (1.0 - done))
        wm.requires_grad_(True)
        return Rollout(obs_seq, torch.stack(rewards), torch.stack(dones), torch.stack(survival))

    def _update_actor(self, rollout: Rollout) -> dict[str, float]:
        """Ascend the imagined return, bootstrapped by the critic at the horizon."""
        H = self.horizon
        discounts = self.gamma ** torch.arange(H, device=self.device, dtype=torch.float32)
        ret = (discounts[:, None] * rollout.survival[:H] * rollout.rewards).sum(0)
        value_h = self.critic(rollout.obs[H].select(*self.critic_groups)).squeeze(-1)
        ret = ret + (self.gamma**H) * rollout.survival[H] * value_h
        self.actor_optimizer.zero_grad()
        (-ret.mean() / H).backward()
        self._reduce_and_clip(self.actor.parameters(), clip_value=self.actor_grad_clip_value)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()  # the bootstrap value left gradients on the critic
        return {
            "actor/imagined_return": ret.mean().item(),
            "actor/imagined_reward": rollout.rewards.mean().item(),
            "actor/survival": rollout.survival[H].mean().item(),
        }

    def _update_imagined_critic(self, rollout: Rollout) -> dict[str, float]:
        """Fit V to the rollout's lambda-returns, weighted by the survival weight of each step."""
        H = self.horizon
        with torch.no_grad():
            critic_obs = torch.stack([o.select(*self.critic_groups) for o in rollout.obs]).detach()
            values = self.critic(critic_obs.reshape(-1)).view(H + 1, -1)
            targets = self._lambda_returns(rollout.rewards.detach(), rollout.dones.detach(), values)
            weights = rollout.survival[:H].detach()
        return self._fit_critic(critic_obs[:H].reshape(-1), targets.reshape(-1), weights.reshape(-1))

    def _update_replay_critic(self) -> dict[str, float]:
        """Fit V on real replay windows to TD(lambda) targets from the target network."""
        L = self.critic_window
        win = self.replay_buffer.sample_windows(self.critic_batch_size, L, device=self.device)
        if win is None:
            return {"critic/skipped": 1.0}
        critic_obs = self._with_goals(win.obs, self._window_goals(win)).select(*self.critic_groups)
        with torch.no_grad():
            values = self.critic_target(critic_obs.reshape(-1)).view(L + 1, -1)  # type: ignore[misc]
            targets = self._lambda_returns(win.rewards, win.terminated, values)
        out = self._fit_critic(critic_obs[:L].reshape(-1), targets.reshape(-1))
        self.critic_target.update()  # type: ignore[union-attr]
        return out

    def _fit_critic(
        self, critic_obs: TensorDict, targets: torch.Tensor, weights: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Regress V onto fixed targets for ``critic_epochs`` passes over shuffled minibatches."""
        n = targets.shape[0]
        size = max(n // self.critic_mini_batches, 1)
        weights = torch.ones_like(targets) if weights is None else weights
        total = 0.0
        for _ in range(self.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.critic_mini_batches):
                idx = perm[i * size : (i + 1) * size]
                err = (self.critic(critic_obs[idx]).squeeze(-1) - targets[idx]).pow(2)
                loss = (weights[idx] * err).sum() / weights[idx].sum().clamp(min=1e-6)
                self.critic_optimizer.zero_grad()
                loss.backward()
                self._reduce_and_clip(self.critic.parameters())
                self.critic_optimizer.step()
                total += loss.item()
        return {
            "critic/loss": total / (self.critic_epochs * self.critic_mini_batches),
            "critic/target": targets.mean().item(),
        }

    def _lambda_returns(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Lambda-returns backward over a window, bootstrapped from ``values[-1]``.

        Args:
            rewards: ``(L, B)``.
            dones: Termination probabilities or flags after each step, ``(L, B)``.
            values: ``(L + 1, B)``.

        Returns:
            ``(L, B)`` targets.
        """
        L = rewards.shape[0]
        targets = torch.empty_like(rewards)
        nxt = values[-1]
        for t in reversed(range(L)):
            nxt = rewards[t] + self.gamma * (1.0 - dones[t]) * ((1.0 - self.lam) * values[t + 1] + self.lam * nxt)
            targets[t] = nxt
        return targets

    """
    Helpers.
    """

    def _wm_inputs(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Split observations into the world model's raw state, context, goal and noise-free observation."""
        spec = self.spec
        ctx = obs[spec.context_group] if spec.context_group else None
        goal = torch.cat([obs[g] for g in spec.goal_groups], dim=-1) if spec.goal_groups else None
        obs_bar = obs[spec.obs_group] if spec.obs_group else None
        return obs[spec.state_group], ctx, goal, obs_bar

    def _provided_goals(self, keys: torch.Tensor, steps: int) -> TensorDict:
        """Return the goal groups at ``0 .. steps`` env steps after each key, ``(steps + 1, B, ...)``."""
        B = keys.shape[0]
        offsets = torch.arange(steps + 1, device=keys.device).repeat_interleave(B)
        goals = self.goal_provider(keys.repeat(steps + 1, 1), offsets)  # type: ignore[misc]
        return goals.reshape(steps + 1, B)

    def _window_goals(self, win: SequenceReplayBuffer.Window) -> TensorDict | None:
        """Return the goal groups along a replay window: from the provider when there is one, else as logged."""
        if not self.spec.goal_groups:
            return None
        if self.goal_provider is not None:
            return self._provided_goals(win.obs[self.spec.goal_key_group][0], win.actions.shape[0])
        return win.obs.select(*self.spec.goal_groups)

    @staticmethod
    def _with_goals(obs: TensorDict, goals: TensorDict | None) -> TensorDict:
        """Observations with their goal groups set (they are not stored when a provider supplies them)."""
        if goals is None:
            return obs
        obs = obs.clone(recurse=False)
        obs.update(goals)
        return obs

    def _reduce_and_clip(self, params: Any, clip_value: float | None = None) -> None:
        """Average gradients across ranks, clip the global norm, then optionally clip element-wise."""
        params = [p for p in params if p.grad is not None]
        if self.is_multi_gpu:
            grads = torch.cat([p.grad.view(-1) for p in params])
            torch.distributed.all_reduce(grads, op=torch.distributed.ReduceOp.SUM)
            grads /= self.gpu_world_size
            offset = 0
            for p in params:
                p.grad.copy_(grads[offset : offset + p.numel()].view_as(p.grad))
                offset += p.numel()
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        if clip_value is not None:
            nn.utils.clip_grad_value_(params, clip_value)

    def _set_learning_rate(self) -> None:
        """Cosine-decay every learning rate to ``lr_final_ratio`` over ``lr_decay_updates``, then hold."""
        if self.lr_decay_updates <= 0:
            return
        frac = min(self.update_step / self.lr_decay_updates, 1.0)
        scale = self.lr_final_ratio + (1.0 - self.lr_final_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = self.learning_rate * scale

    """
    Runner interface.
    """

    def train_mode(self) -> None:
        """Set every model to training mode."""
        self.actor.train()
        self.critic.train()
        self.world_model.train()

    def eval_mode(self) -> None:
        """Set every model to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.world_model.eval()

    def eval(self, env: VecEnv, max_steps: int = 200, stochastic: bool = False, action_repeat: int = 1) -> list[dict]:
        """Roll out the policy for ``max_steps`` env steps (each ``env.step`` renders a frame for a video wrapper).

        Args:
            env: Vectorized environment to roll out in.
            max_steps: Number of environment steps to run.
            stochastic: Add the collection noise to the deterministic actor, as during training.
            action_repeat: Hold each queried action for this many ``env.step`` calls before re-querying the actor.

        Returns:
            An empty list (this rollout collects no eval metrics).
        """
        was_training = self.actor.training
        self.eval_mode()
        if hasattr(env, "eval_mode"):
            env.eval_mode()
        obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()[0]
        action_repeat = max(1, action_repeat)

        def query_actions() -> torch.Tensor:
            actions = self.actor(obs)
            if stochastic and self.action_noise_std > 0:
                actions = actions + self.action_noise_std * torch.randn_like(actions)
            return actions

        with torch.inference_mode():
            actions = query_actions()
            for step in range(max_steps):
                if step > 0 and step % action_repeat == 0:
                    actions = query_actions()
                obs, _, _, _ = env.step(actions)
        if was_training:
            self.train_mode()
            if hasattr(env, "train_mode"):
                env.train_mode()
        return []

    def get_policy(self) -> MLPModel:
        """Return the actor."""
        return self.actor

    def log_info(self) -> dict:
        """Extra per-iteration values for the runner's log call."""
        lr = self.actor_optimizer.param_groups[0]["lr"]
        return {"learning_rate": lr, "action_std": torch.full((1,), self.action_noise_std)}

    def reset_rollout_state(self) -> None:
        """Start a new logged episode: the env was reset outside a step (e.g. after an eval)."""
        self.replay_buffer.mark_episode_start()

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """State-dict keys sufficient to run, evaluate and export the policy (the rest is resume-only)."""
        return ("actor_state_dict",)

    def save(self) -> dict:
        """Return the model and optimizer states."""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            "wm_optimizer_state_dict": self.wm_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load model and optimizer states; returns whether to resume the iteration count."""
        if load_cfg is None:
            full = "world_model_state_dict" in loaded_dict
            load_cfg = {"actor": True, "critic": full, "world_model": full, "optimizer": full, "iteration": full}
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            if self.critic_target is not None:
                self.critic_target.hard_sync()
        if load_cfg.get("world_model"):
            self.world_model.load_state_dict(loaded_dict["world_model_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.wm_optimizer.load_state_dict(loaded_dict["wm_optimizer_state_dict"])
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.update_step = loaded_dict.get("update_step", 0)
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        """Broadcast every model from rank 0."""
        params = [self.actor.state_dict(), self.critic.state_dict(), self.world_model.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self.actor.load_state_dict(params[0])
        self.critic.load_state_dict(params[1])
        self.world_model.load_state_dict(params[2])
        if self.critic_target is not None:
            self.critic_target.hard_sync()

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str, inference: bool = False
    ) -> WorldModelAC:
        """Build the algorithm from a runner cfg dict and the env's world-model spec (``env.world_model_spec``).

        With a ``goal_key_group`` in the spec, goals come from ``env.world_model_goals`` and are checked against the
        env's own goal observations once here.
        """
        cfg = copy.deepcopy(cfg)
        alg_cfg = cfg["algorithm"]
        alg_class: type[WorldModelAC] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore
        spec: WorldModelSpec | None = getattr(env, "world_model_spec", None)
        if spec is None:
            raise ValueError("WorldModelAC needs the env's world_model_spec (the env cfg's world_model field).")
        wm_cfg = dict(alg_cfg.pop("world_model"))
        wm_class: type[WorldModel] = resolve_callable(wm_cfg.pop("class_name", "WorldModel"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

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

        num_actions = env.num_actions
        actor = actor_class(obs, obs_groups, "actor", num_actions, **cfg["actor"])
        critic = critic_class(obs, obs_groups, "critic", 1, **cfg["critic"])
        world_model = wm_class(
            state_dim=width(spec.state_group),
            action_dim=num_actions,
            context_dim=width(spec.context_group),
            goal_dim=sum(width(g) for g in spec.goal_groups),
            obs_dim=width(spec.obs_group),
            **wm_cfg,
        )
        imagined_obs = ImaginedObservation(spec, obs, [*obs_groups["actor"], *obs_groups["critic"]])
        if world_model.observation is None and imagined_obs.reads_obs_head:
            raise ValueError("A model reads world-model observation terms, but the spec names no obs_group.")

        # provided goals are recomputed from their key, so only the key is stored
        stored = [spec.state_group, *imagined_obs.groups, spec.context_group, spec.obs_group]
        if goal_provider is not None:
            stored = [g for g in stored if g not in spec.goal_groups] + [spec.goal_key_group]
        else:
            stored += spec.goal_groups
        buffer_size = int(alg_cfg.pop("replay_buffer_size", 1_000_000))
        replay = SequenceReplayBuffer(
            env.num_envs,
            1 if inference else max(buffer_size // env.num_envs, 1),
            obs,
            [g for g in stored if g],
            num_actions,
            device=cfg.get("storage_device") or device,
            obs_dtypes=alg_cfg.pop("replay_obs_dtypes", None),
        )
        return alg_class(
            actor,
            critic,
            world_model,
            replay,
            spec,
            imagined_obs,
            obs_groups["critic"],
            goal_provider=goal_provider,
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
