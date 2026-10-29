"""Actor-critic trained through a learned, differentiable world model (SuperMimic-style BPTT)."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.imagination import ImaginedObservation
from robot_rl.env import VecEnv, WorldModelSpec
from robot_rl.models import MLPModel, WorldModel
from robot_rl.modules import TargetNetwork
from robot_rl.storage import SequenceReplayBuffer
from robot_rl.utils import resolve_callable, resolve_obs_groups


class WorldModelAC:
    """Deterministic actor improved by backpropagation through imagined world-model rollouts.

    Each update fits the world model to ``wm_window``-step autoregressive replay windows, then rolls the actor
    ``imagination_horizon`` steps through the frozen model from replay start states and ascends the imagined return
    ``sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)`` with the soft survival weight ``c_{t+1} = c_t (1 - d_{t+1})``.
    The value critic is fit either on real replay windows (``critic_mode="replay"``, TD(lambda) with a target
    network) or on the imagined rollout (``"imagination"``, lambda-returns without a target, as in SuperMimic).

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
        actor_grad_clip_value: float = 0.1,
        noise_bank_size: int = 65536,
        tf32: bool = True,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm. See the class docstring for the update."""
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
        self.critic.update_normalization(self._critic_obs(obs))
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
                out.update(self._update_actor_and_imagined_critic())
                if self.critic_mode == "replay":
                    out.update(self._update_replay_critic())
            self.update_step += 1
            for k, v in out.items():
                logs.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in logs.items()}

    def _update_world_model(self) -> dict[str, float]:
        """Fit the world model on autoregressive replay windows rolled on the logged actions."""
        sums = {"state": 0.0, "reward": 0.0, "obs": 0.0, "termination": 0.0, "k_step_error": 0.0}
        k = self.wm_window
        for _ in range(self.wm_mini_batches):
            win = self.replay_buffer.sample_windows(self.wm_batch_size, k, device=self.device)
            state_raw, ctx_raw, goal_raw, obs_raw = self._wm_inputs(win.obs)
            wm = self.world_model
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
        return {f"world_model/{key}": v / self.wm_mini_batches for key, v in sums.items()}

    def _update_actor_and_imagined_critic(self) -> dict[str, float]:
        """BPTT through the frozen world model; in ``"imagination"`` mode also fit the critic on the rollout."""
        wm, H = self.world_model, self.horizon
        win = self.replay_buffer.sample_windows(self.num_imagined, H, device=self.device, require_next=True)
        bank_rows = self.replay_buffer.sample_rows(self.noise_bank_size, device=self.device)
        bank = self.imagined_obs.noise_bank(bank_rows)
        state_raw, ctx_raw, goal_raw, _ = self._wm_inputs(win.obs)
        state_n, ctx_n, goal_n = wm.normalize(state_raw, ctx_raw, goal_raw)
        ctx_n = ctx_n[0]
        goals = win.obs.select(*self.spec.goal_groups) if self.spec.goal_groups else None

        wm.requires_grad_(False)
        s = state_n[0]
        actor_obs = win.obs[0].select(*self.imagined_obs.groups)
        survival = torch.ones(self.num_imagined, device=self.device)
        ret = torch.zeros(self.num_imagined, device=self.device)
        states, rewards, dones = [s], [], []
        for t in range(H):
            action = self.actor(actor_obs)
            s_next, reward, term_logit = wm.step(s, action, ctx_n, goal_n[t])
            done = torch.sigmoid(term_logit) if term_logit is not None else torch.zeros_like(survival)
            ret = ret + (self.gamma**t) * survival * reward  # type: ignore[operator]
            survival = survival * (1.0 - done)
            idx = torch.randint(0, self.noise_bank_size, (self.num_imagined,), device=self.device)
            noise = {g: b[idx] for g, b in bank.items()}
            obs_bar = wm.observe(s_next, ctx_n, goal_n[t + 1]) if wm.observation is not None else None
            actor_obs = self.imagined_obs.step(
                actor_obs,
                obs_bar,
                action,
                goals[t + 1] if goals is not None else None,
                noise,  # type: ignore[arg-type]
            )
            s = s_next
            states.append(s)
            rewards.append(reward)
            dones.append(done)
        value_h = self.critic(self._imagined_critic_obs(s, ctx_raw[0], win.obs, H)).squeeze(-1)
        ret = ret + (self.gamma**H) * survival * value_h
        actor_loss = -ret.mean() / H
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._reduce_and_clip(self.actor.parameters(), clip_value=self.actor_grad_clip_value)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()  # the bootstrap value left gradients on the critic
        wm.requires_grad_(True)

        out = {
            "actor/imagined_return": ret.mean().item(),
            "actor/imagined_reward": torch.stack(rewards).mean().item(),  # type: ignore[arg-type]
            "actor/survival": survival.mean().item(),
        }
        if self.critic_mode == "imagination":
            with torch.no_grad():
                states_t = torch.stack(states).detach()
                critic_obs = self._imagined_critic_obs(states_t, ctx_raw[0], win.obs, slice(0, H + 1))
                values = self.critic(critic_obs.reshape(-1)).view(H + 1, -1)
                targets = self._lambda_returns(
                    torch.stack(rewards).detach(),
                    torch.stack(dones).detach(),
                    values,  # type: ignore[arg-type]
                )
            out.update(self._fit_critic(critic_obs[:H].reshape(-1), targets.reshape(-1)))
        return out

    def _update_replay_critic(self) -> dict[str, float]:
        """Fit V on real replay windows to TD(lambda) targets from the target network."""
        L = self.critic_window
        win = self.replay_buffer.sample_windows(self.critic_batch_size, L, device=self.device)
        critic_obs = self._critic_obs(win.obs)
        with torch.no_grad():
            values = self.critic_target(critic_obs.reshape(-1)).view(L + 1, -1)  # type: ignore[misc]
            # a time-out at the window's end leaves no next row to bootstrap from: drop those windows
            keep = win.next_valid | (win.terminated[-1] > 0)
            targets = self._lambda_returns(win.rewards, win.terminated, values)
        flat_obs = critic_obs[:L][:, keep].reshape(-1)
        out = self._fit_critic(flat_obs, targets[:, keep].reshape(-1))
        self.critic_target.update()  # type: ignore[union-attr]
        return out

    def _fit_critic(self, critic_obs: TensorDict, targets: torch.Tensor) -> dict[str, float]:
        """Regress V onto fixed targets for ``critic_epochs`` passes over shuffled minibatches."""
        n = targets.shape[0]
        size = max(n // self.critic_mini_batches, 1)
        total = 0.0
        for _ in range(self.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.critic_mini_batches):
                idx = perm[i * size : (i + 1) * size]
                loss = nn.functional.mse_loss(self.critic(critic_obs[idx]).squeeze(-1), targets[idx])
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

    def _critic_obs(self, obs: TensorDict) -> TensorDict:
        """Select the critic's groups from logged observations."""
        return obs.select(*self.critic_groups)

    def _imagined_critic_obs(
        self, state_n: torch.Tensor, ctx_raw: torch.Tensor | None, logged: TensorDict, t: int | slice
    ) -> TensorDict:
        """Critic input at imagined state(s): the de-normalized state plus the logged context and goal at ``t``."""
        groups = {self.spec.state_group: self.world_model.denormalize_state(state_n)}
        shape = state_n.shape[:-1]
        if self.spec.context_group in self.critic_groups:
            groups[self.spec.context_group] = ctx_raw.expand(*shape, -1)  # type: ignore[union-attr]
        for g in self.spec.goal_groups:
            if g in self.critic_groups:
                groups[g] = logged[g][t]
        return TensorDict(groups, batch_size=list(shape), device=self.device)

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

    def get_policy(self) -> MLPModel:
        """Return the actor."""
        return self.actor

    def log_info(self) -> dict:
        """Extra per-iteration values for the runner's log call."""
        lr = self.actor_optimizer.param_groups[0]["lr"]
        return {"learning_rate": lr, "action_std": torch.full((1,), self.action_noise_std)}

    def reset_rollout_state(self) -> None:
        """Nothing to clear: the actor is stateless."""

    def save(self) -> dict:
        """Return the model and optimizer states."""
        saved = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            "wm_optimizer_state_dict": self.wm_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }
        return saved

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
        """Build the algorithm from a runner cfg dict and the env's world-model spec.

        The spec comes from ``env.world_model_spec`` (the env side owns the spaces); a ``world_model_spec`` entry in
        the algorithm cfg is a fallback for envs without one.
        """
        cfg = copy.deepcopy(cfg)
        alg_cfg = cfg["algorithm"]
        alg_class: type[WorldModelAC] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore
        spec = getattr(env, "world_model_spec", None)
        spec_cfg = alg_cfg.pop("world_model_spec", None)
        if spec is None:
            if spec_cfg is None:
                raise ValueError("WorldModelAC needs the env's world_model_spec (or algorithm.world_model_spec).")
            spec = WorldModelSpec.from_dict(spec_cfg)
        wm_cfg = dict(alg_cfg.pop("world_model"))
        wm_class: type[WorldModel] = resolve_callable(wm_cfg.pop("class_name", "WorldModel"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        allowed = {spec.state_group, spec.context_group, *spec.goal_groups}
        if not set(obs_groups["critic"]) <= allowed:
            raise ValueError(f"Critic groups {obs_groups['critic']} must come from the spec's state/context/goals.")

        num_actions = env.num_actions
        actor = actor_class(obs, obs_groups, "actor", num_actions, **cfg["actor"])
        critic = critic_class(obs, obs_groups, "critic", 1, **cfg["critic"])
        width = lambda g: obs[g].shape[-1] if g else 0  # noqa: E731
        world_model = wm_class(
            state_dim=width(spec.state_group),
            action_dim=num_actions,
            context_dim=width(spec.context_group),
            goal_dim=sum(width(g) for g in spec.goal_groups),
            obs_dim=width(spec.obs_group),
            **wm_cfg,
        )
        imagined_obs = ImaginedObservation(spec, obs, obs_groups["actor"])
        if world_model.observation is None and any(imagined_obs.noise_width.values()):
            raise ValueError("The actor reads world-model observation terms, but the spec names no obs_group.")

        groups = [spec.state_group, *obs_groups["actor"], *obs_groups["critic"], *spec.goal_groups]
        groups += [g for g in (spec.context_group, spec.obs_group) if g]
        buffer_size = int(alg_cfg.pop("replay_buffer_size", 1_000_000))
        capacity = 1 if inference else max(buffer_size // env.num_envs, 1)
        replay = SequenceReplayBuffer(
            env.num_envs,
            capacity,
            obs,
            groups,
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
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
