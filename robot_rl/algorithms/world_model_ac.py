"""Actor-critic trained through a learned, differentiable world model (SuperMimic-style BPTT)."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.world_model_learner import Rollout, WorldModelLearner
from robot_rl.env import VecEnv
from robot_rl.models import MLPModel
from robot_rl.modules import TargetNetwork
from robot_rl.utils import resolve_callable, resolve_obs_groups

LEARNER_KEYS = (
    "wm_window",
    "wm_batch_size",
    "wm_mini_batches",
    "reward_loss_coef",
    "obs_loss_coef",
    "termination_loss_coef",
    "imagination_horizon",
    "noise_bank_size",
)
"""Algorithm cfg entries that configure the :class:`WorldModelLearner` rather than the algorithm."""


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
        learner: WorldModelLearner,
        critic_groups: list[str],
        learning_rate: float = 2e-4,
        actor_betas: tuple[float, float] = (0.9, 0.95),
        lr_decay_updates: int = 0,
        lr_final_ratio: float = 0.03,
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
        tf32: bool = True,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            actor: Deterministic policy over the deployable groups.
            critic: Value function over groups the world model can rebuild.
            learner: The world model, its replay and its imagination (fit at ``learning_rate`` too).
            critic_groups: The critic's groups.
            learning_rate: Learning rate of every network.
            actor_betas: Adam betas of the actor.
            lr_decay_updates: Updates of cosine decay to ``lr_final_ratio``; 0 keeps the rate constant.
            lr_final_ratio: Learning-rate floor relative to ``learning_rate``.
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
        self.learner = learner
        self.world_model = learner.world_model
        self.replay_buffer = learner.replay_buffer
        self.critic_target = TargetNetwork(self.critic, critic_tau).to(device) if critic_mode == "replay" else None
        self.critic_groups = critic_groups

        self.learning_rate = learning_rate
        self.lr_decay_updates = lr_decay_updates
        self.lr_final_ratio = lr_final_ratio
        self.horizon = learner.horizon
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

        self.wm_optimizer = learner.optimizer
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
        self.learner.add(self._obs, self._actions, rewards, dones, extras)  # type: ignore[arg-type]
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs.select(*self.critic_groups))

    """
    Updates.
    """

    def update(self) -> dict[str, float]:
        """Run ``updates_per_iteration`` rounds of world-model fit, actor update and critic fit."""
        logs: dict[str, list[float]] = {}
        for _ in range(self.updates_per_iteration):
            self._set_learning_rate()
            out = self.learner.fit(self._reduce_and_clip)
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

    def _imagine_from_replay(self) -> Rollout | None:
        """Draw start states from replay and imagine ``H`` steps under the actor (None when no start is available)."""
        drawn = self.learner.sample_starts(self.num_imagined)
        if drawn is None:
            return None
        starts, goals, noise_bank, _ = drawn
        return self.learner.imagine(self.actor, starts, goals, noise_bank)

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
        critic_obs = self.learner.with_goals(win.obs, self.learner.window_goals(win)).select(*self.critic_groups)
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
            **self.learner.state_dict(),
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
            self.learner.load_state_dict(loaded_dict, optimizer=bool(load_cfg.get("optimizer")), strict=strict)
        if load_cfg.get("optimizer"):
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
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        num_actions = env.num_actions
        actor = actor_class(obs, obs_groups, "actor", num_actions, **cfg["actor"])
        critic = critic_class(obs, obs_groups, "critic", 1, **cfg["critic"])
        buffer_size = int(alg_cfg.pop("replay_buffer_size", 1_000_000))
        learner = WorldModelLearner.build(
            obs,
            env,
            alg_cfg.pop("world_model"),
            [*obs_groups["actor"], *obs_groups["critic"]],
            1 if inference else max(buffer_size // env.num_envs, 1),
            cfg.get("storage_device") or device,
            device,
            obs_dtypes=alg_cfg.pop("replay_obs_dtypes", None),
            learning_rate=alg_cfg.get("learning_rate", 2e-4),
            **{k: alg_cfg.pop(k) for k in LEARNER_KEYS if k in alg_cfg},
        )
        return alg_class(
            actor,
            critic,
            learner,
            obs_groups["critic"],
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
