# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from itertools import chain
from tensordict import TensorDict
from typing import Any

from robot_rl.env import VecEnv
from robot_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from robot_rl.models import FuseModel, MLPModel
from robot_rl.modules import TargetNetwork
from robot_rl.modules.rnn import HiddenState
from robot_rl.storage import ReplayBuffer
from robot_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


def _first_slot(state: HiddenState) -> torch.Tensor:
    """Return the ``h`` tensor of a GRU state or an LSTM ``(h, c)`` pair."""
    return state[0] if isinstance(state, tuple) else state  # type: ignore[return-value]


def _param_grad_norm(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return the total L2 gradient norm over ``params`` (zero if none has a gradient)."""
    grads = [p.grad.norm() for p in params if p.grad is not None]
    return torch.stack(grads).norm() if grads else torch.zeros(())


class SAC:
    """Soft Actor-Critic (SAC) algorithm.

    Reference:
        - Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
          Stochastic Actor." arXiv preprint arXiv:1801.01290 (2018).
        - Sabatini et al. "Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged
          Locomotion." arXiv preprint arXiv:2605.24975 (2026).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic_1: FuseModel,
        critic_2: FuseModel,
        replay_buffer: ReplayBuffer,
        num_actions: int,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        alpha_learning_rate: float = 1e-3,
        actor_optimizer: str = "adam",
        critic_optimizer: str = "adam",
        auto_alpha: bool = True,
        alpha: float = 0.05,
        tau: float = 0.005,
        gamma: float = 0.99,
        target_entropy_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        policy_frequency: int = 1,
        n_steps: int = 1,
        seq_len: int = 16,
        burn_in: int = 8,
        aux_obs_group: str | None = None,
        aux_loss_weight: float = 1.0,
        attn_target_group: str | None = None,
        attn_loss_weight: float = 1.0,
        perception_warmup_updates: int = 0,
        reference_bc_weight: float = 0.0,
        reference_bc_decay_updates: int = 0,
        freeze_actor_encoder: bool = False,
        compile_mode: str | None = None,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SAC algorithm. See the module docstring for the design."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND (intrinsic reward) -- optional, shared extension (owns its own predictor optimizer).
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry augmentation -- optional, shared extension.
        if symmetry_cfg is not None and (actor.is_recurrent or critic_1.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # Models
        self.actor = actor.to(device)
        self.critic_1 = critic_1.to(device)
        self.critic_2 = critic_2.to(device)
        self.critic_1_target = TargetNetwork(self.critic_1, tau).to(device)
        self.critic_2_target = TargetNetwork(self.critic_2, tau).to(device)

        # Replay buffer
        self.replay_buffer = replay_buffer
        self.transition = ReplayBuffer.Transition()

        # Hyperparameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.actor_learning_rate = actor_learning_rate
        self.policy_frequency = policy_frequency
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        # Recurrent actors train on contiguous windows: `burn_in` steps re-derive the stale stored
        # state without gradients, then `seq_len` steps carry the loss.
        self.recurrent = actor.is_recurrent
        self.seq_len = seq_len
        self.burn_in = burn_in
        # supervised target for the recurrent actor's aux head (e.g. privileged ball position)
        self.aux_obs_group = aux_obs_group
        self.aux_loss_weight = aux_loss_weight
        # (u, v, visible) target cell for an attention actor's cross-modal attention
        self.attn_target_group = attn_target_group
        self.attn_loss_weight = attn_loss_weight
        # updates during which the actor trains on its supervised targets only (RL term and alpha frozen),
        # so the head keeps its initial exploration until the encoder can see
        self.perception_warmup_updates = perception_warmup_updates
        # fine-tuning a cloned student: pull the actor mean toward a frozen copy of it, fading out over
        # reference_bc_decay_updates, so the first RL steps cannot exploit critic error off the student's data
        self.reference_bc_weight = reference_bc_weight
        self.reference_bc_decay_updates = reference_bc_decay_updates
        self.reference_actor: nn.Module | None = None
        # mean |h| over rollout steps since the last update, to compare against the training-side state
        self._rollout_hidden_abs = torch.zeros((), device=device)
        self._rollout_hidden_steps = 0
        self._last_critic_losses = (0.0, 0.0)
        self.update_step = 0
        self.intrinsic_rewards: torch.Tensor | None = None

        self.target_entropy = -target_entropy_scale * num_actions

        # Temperature (log-space); learned against the target entropy when auto_alpha.
        self.log_alpha = torch.log(torch.tensor(self.alpha, device=self.device)).detach().clone()
        self.log_alpha.requires_grad_(auto_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate) if auto_alpha else None

        # head-only fine-tuning: the distilled encoder (CNN, RNN, normalizers) stays, RL moves the policy head alone
        self.freeze_actor_encoder = freeze_actor_encoder
        if freeze_actor_encoder:
            for name, p in self.actor.named_parameters():
                if not name.startswith(("mlp.", "distribution.", "aux_head.")):
                    p.requires_grad_(False)
        # Optimizers over the trainable (online) parameters. Target params are frozen and excluded.
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = [
            p for p in chain(self.critic_1.parameters(), self.critic_2.parameters()) if p.requires_grad
        ]
        self.actor_optimizer = resolve_optimizer(actor_optimizer)(self.actor_parameters, lr=actor_learning_rate)
        self.critic_optimizer = resolve_optimizer(critic_optimizer)(self.critic_parameters, lr=critic_learning_rate)

        # Apply torch.compile to the forward+loss+backward paths.
        if compile_mode not in (None, "eager", "none") and not self.recurrent:
            self._critic_losses_and_backward = torch.compile(self._critic_losses_and_backward, mode=compile_mode)
            self._actor_loss_and_backward = torch.compile(self._actor_loss_and_backward, mode=compile_mode)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample a stochastic action and record the transition's observation/action."""
        if self.recurrent:
            # the state that PRODUCES this action, captured before the forward advances it
            hs = self.actor.get_hidden_state(batch_size=obs.batch_size[0], device=self.device)
            self.transition.hidden_state = tuple(h.clone() for h in hs) if isinstance(hs, tuple) else hs.clone()
            self._rollout_hidden_abs += _first_slot(hs).detach().abs().mean()
            self._rollout_hidden_steps += 1
        with torch.no_grad():
            action = self.actor(obs, stochastic_output=True)
        self.transition.observations = obs
        self.transition.actions = action
        return action

    def process_env_step(self, next_obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record a step and insert the transition into the replay buffer.

        Handles the off-policy timeout distinction: on a *timeout* the true (pre-reset) next observation comes
        from ``extras['time_outs_obs']`` and the bootstrap continues (``next_terminated=0``); on a true
        *termination* the bootstrap is masked out. Degrades gracefully (uses ``next_obs`` as-is) when the env
        does not provide ``time_outs``/``time_outs_obs``.
        """
        num_envs = dones.shape[0]
        dones_bool = dones.view(-1).bool().to(self.device)

        if "time_outs" in extras and extras["time_outs"] is not None:
            time_outs = extras["time_outs"].view(-1).bool().to(self.device)
            if extras.get("time_outs_obs") is not None:
                # the mask must carry one trailing singleton per feature dim of each group: (N, 1) for a
                # flat obs, (N, 1, 1, 1) for an image. A fixed (N, 1) right-aligns against a 4D group and
                # collides with its height instead of broadcasting.
                def _mask_for(value: torch.Tensor) -> torch.Tensor:
                    return time_outs.view(-1, *([1] * (value.dim() - 1)))

                true_next_obs = TensorDict(
                    {
                        key: torch.where(
                            _mask_for(next_obs[key]),
                            extras["time_outs_obs"][key].to(self.device),
                            next_obs[key],
                        )
                        for key in next_obs.keys()  # noqa: SIM118 -- TensorDict iterates its batch dim, not keys
                    },
                    batch_size=next_obs.batch_size,
                )
            else:
                true_next_obs = next_obs
            next_terminated = dones_bool & ~time_outs
        else:
            true_next_obs = next_obs
            next_terminated = dones_bool

        # Update normalizers on the observed next states.
        # a fine-tuned student keeps its normalizer: adapting it alone moves the actor off the frozen reference
        if self.reference_actor is None:
            self.actor.update_normalization(true_next_obs)
        self.critic_1.update_normalization(true_next_obs)
        self.critic_2.update_normalization(true_next_obs)
        if self.rnd:
            self.rnd.update_normalization(true_next_obs)

        rew = rewards.clone().to(self.device)
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(true_next_obs)
            rew = rew + self.intrinsic_rewards

        self.transition.rewards = rew
        self.transition.next_observations = true_next_obs
        self.transition.dones = dones_bool
        self.transition.next_terminated = next_terminated.byte()
        # SAC carries no latent context; provide an empty (z_dim=0) tensor for the shared buffer.
        self.transition.context = torch.zeros(num_envs, 0, device=self.device)

        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()
        if self.recurrent:
            self.actor.reset(dones_bool)

    def update(self) -> dict:
        """Run the off-policy SAC updates over sampled mini-batches; returns mean losses."""
        if self.recurrent:
            return self._update_recurrent()
        mean_critic_1_loss = 0.0
        mean_critic_2_loss = 0.0
        mean_actor_loss = 0.0
        mean_alpha_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        num_actor_updates = 0

        n_updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(n_updates):
            batch = self.replay_buffer.sample_mini_batch(self.device)
            obs_b = batch.observations
            next_obs_b = batch.next_observations
            actions_b = batch.actions
            rewards_b = batch.rewards.view(-1)
            not_terminated = 1.0 - batch.next_terminated.view(-1).float()

            # Critic update -- bootstrapped target with entropy and n-step discount.
            critic_1_loss, critic_2_loss = self._update_critics(
                batch, obs_b, next_obs_b, actions_b, rewards_b, not_terminated
            )

            # Alpha and actor updates, delayed by policy_frequency.
            new_actions, logp = self.actor.act_and_log_prob(obs_b)

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.is_multi_gpu and self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad /= self.gpu_world_size
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                mean_alpha_loss += alpha_loss.item()

            if self.update_step % self.policy_frequency == 0:
                for p in self.critic_parameters:
                    p.requires_grad_(False)
                actor_loss = self._update_actor(obs_b, new_actions, logp)
                for p in self.critic_parameters:
                    p.requires_grad_(True)
                mean_actor_loss += actor_loss.item()
                num_actor_updates += 1

            # Soft-update the target critics.
            self.critic_1_target.update()
            self.critic_2_target.update()

            # RND predictor update (RND owns its optimizer).
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(obs_b)
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(list(self.rnd.predictor.parameters()))
                self.rnd.optimizer.step()
                mean_rnd_loss += rnd_loss.item()

            mean_critic_1_loss += critic_1_loss.item()
            mean_critic_2_loss += critic_2_loss.item()
            self.update_step += 1

        loss_dict = {
            "critic_1": mean_critic_1_loss / n_updates,
            "critic_2": mean_critic_2_loss / n_updates,
            "actor": mean_actor_loss / max(num_actor_updates, 1),
            "alpha": mean_alpha_loss / n_updates if self.auto_alpha else 0.0,
            "alpha_value": self.alpha,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / n_updates
        return loss_dict

    def _update_recurrent(self) -> tuple[dict, dict]:
        """SAC updates for a recurrent actor; returns mean losses and diagnostics.

        The critics are feedforward, so they train on independent transitions exactly as in the plain
        update; their bootstrap action comes from the actor holding the state stepped from the stored
        one. The actor trains on contiguous windows replayed from their stored starting state: the
        burn-in prefix under no-grad, then the loss steps, with the state zeroed at every episode start.
        """
        sums = {"critic_1": 0.0, "critic_2": 0.0, "actor": 0.0, "alpha": 0.0}
        diag: dict[str, float] = {}
        counts: dict[str, int] = {}

        def _acc(key: str, value: torch.Tensor | float) -> None:
            diag[key] = diag.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

        num_updates = 0
        num_actor_updates = 0
        for _ in range(self.num_learning_epochs * self.num_mini_batches):
            batch = self.replay_buffer.sample_sequences(self.seq_len, self.burn_in, self.device)
            if batch is None:
                break
            self._update_critics_recurrent(_acc)

            burn = batch.burn_in
            obs, resets = batch.observations, batch.resets
            train_obs, train_resets = obs[burn:], resets[burn:]
            seq_len, batch_size = train_obs.batch_size
            flat = seq_len * batch_size
            flat_obs = train_obs.reshape(flat)
            actions_b = batch.actions[burn:].reshape(flat, -1)
            with torch.no_grad():
                h0 = batch.init_hidden
                if burn > 0:
                    _, h0 = self.actor.encode_sequence(obs[:burn], h0, resets[:burn])
                self._recurrent_state_diagnostics(batch, h0, _acc)

            # Alpha and actor on the window; gradients stop at the burn-in boundary.
            latent, _ = self.actor.encode_sequence(train_obs, h0, train_resets)
            latent = latent.reshape(flat, -1)
            new_actions, logp = self.actor.act_and_log_prob_from_latent(latent)
            logp = logp.reshape(-1)
            ref_bc = None
            if self.reference_actor is not None:
                with torch.no_grad():
                    ref = self.reference_actor
                    h0_ref = batch.init_hidden
                    if burn > 0:
                        _, h0_ref = ref.encode_sequence(obs[:burn], h0_ref, resets[:burn])
                    ref_latent, _ = ref.encode_sequence(train_obs, h0_ref, train_resets)
                    ref.distribution.update(ref.mlp(ref_latent.reshape(flat, -1)))  # type: ignore[attr-defined]
                    ref_mean = ref.distribution.mean  # type: ignore[attr-defined]
                # summed over action dims (TD3+BC scale): a per-dim mean was 256x too weak against the Q term
                ref_bc = (self.actor.distribution.mean - ref_mean).pow(2).sum(-1).mean()  # type: ignore[attr-defined]
            with torch.no_grad():
                _acc("Actor/logp", logp.mean())
                attn_entropy = getattr(self.actor, "attention_entropy", lambda: None)()
                if attn_entropy is not None:
                    _acc("Actor/attn_entropy", attn_entropy)
                # sign of the auto-alpha drive: > 0 means entropy is below target and alpha will rise
                _acc("Actor/entropy_gap", (logp + self.target_entropy).mean())
            warming_up = self.update_step < self.perception_warmup_updates
            if self.auto_alpha and not warming_up:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.is_multi_gpu and self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad /= self.gpu_world_size
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                sums["alpha"] += alpha_loss.item()
            if self.update_step % self.policy_frequency == 0:
                for p in self.critic_parameters:
                    p.requires_grad_(False)
                q1_pi = self.critic_1(flat_obs, new_actions).view(-1)
                q2_pi = self.critic_2(flat_obs, new_actions).view(-1)
                min_q_pi = torch.min(q1_pi, q2_pi)
                rl_term = (self.log_alpha.exp().detach() * logp - min_q_pi).mean()
                actor_loss = rl_term * 0.0 if warming_up else rl_term
                _acc("Actor/warming_up", float(warming_up))
                if ref_bc is not None:
                    w = self._reference_bc_weight()
                    actor_loss = actor_loss + w * ref_bc
                    _acc("Actor/ref_bc_loss", ref_bc.detach())
                    _acc("Actor/ref_bc_weight", w)
                if self.aux_obs_group is not None:
                    target = train_obs[self.aux_obs_group].reshape(flat, -1)
                    pred = self.actor.aux_prediction(latent)  # type: ignore[attr-defined]
                    aux_loss = nn.functional.mse_loss(pred, target)
                    actor_loss = actor_loss + self.aux_loss_weight * aux_loss
                    _acc("Actor/aux_loss", aux_loss.detach())
                    _acc("Actor/aux_abs_err", (pred - target).detach().abs().mean())
                if self.attn_target_group is not None:
                    uvv = train_obs[self.attn_target_group].reshape(flat, -1)
                    visible = uvv[:, 2]
                    log_p = self.actor.attention_target_log_prob(uvv[:, :2])  # type: ignore[attr-defined]
                    attn_loss = -(log_p * visible).sum() / visible.sum().clamp(min=1.0)
                    actor_loss = actor_loss + self.attn_loss_weight * attn_loss
                    _acc("Actor/attn_loss", attn_loss.detach())
                    _acc(
                        "Actor/attn_target_prob", (log_p.detach().exp() * visible).sum() / visible.sum().clamp(min=1.0)
                    )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(self.actor_parameters)
                _acc("Actor/grad_norm", nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm))
                _acc("Actor/grad_norm_rnn", _param_grad_norm(self.actor.rnn.parameters()))  # type: ignore[attr-defined]
                _acc("Actor/grad_norm_cnn", _param_grad_norm(self.actor.cnns.parameters()))  # type: ignore[attr-defined]
                self.actor_optimizer.step()
                for p in self.critic_parameters:
                    p.requires_grad_(True)
                with torch.no_grad():
                    # policy-action Q minus data-action Q: a growing gap is the actor exploiting critic error
                    q_data = torch.min(self.critic_1(flat_obs, actions_b), self.critic_2(flat_obs, actions_b)).view(-1)
                    _acc("Critic/q_pi_gap", (min_q_pi - q_data).mean())
                sums["actor"] += actor_loss.item()
                num_actor_updates += 1
            self.critic_1_target.update()
            self.critic_2_target.update()
            sums["critic_1"] += self._last_critic_losses[0]
            sums["critic_2"] += self._last_critic_losses[1]
            self.update_step += 1
            num_updates += 1
        if self._rollout_hidden_steps > 0:
            _acc("Recurrent/hidden_abs_rollout", self._rollout_hidden_abs / self._rollout_hidden_steps)
            self._rollout_hidden_abs.zero_()
            self._rollout_hidden_steps = 0
        n = max(num_updates, 1)
        losses = {
            "critic_1": sums["critic_1"] / n,
            "critic_2": sums["critic_2"] / n,
            "actor": sums["actor"] / max(num_actor_updates, 1),
            "alpha": sums["alpha"] / n if self.auto_alpha else 0.0,
            "alpha_value": self.alpha,
        }
        return losses, {k: v / counts[k] for k, v in diag.items()}

    def _update_critics_recurrent(self, acc: Callable[[str, torch.Tensor | float], None]) -> None:
        """One twin-critic step on independent transitions, bootstrapping through the recurrent actor.

        The next-state action is sampled with the state the actor reaches from the stored pre-step
        state through the step's observation, which is the rollout state up to staleness.
        """
        batch = self.replay_buffer.sample_mini_batch(self.device)
        obs_b, next_obs_b, actions_b = batch.observations, batch.next_observations, batch.actions
        rewards_b = batch.rewards.view(-1)
        not_terminated = 1.0 - batch.next_terminated.view(-1).float()
        with torch.no_grad():
            h_next = self.actor.step_state(obs_b, batch.hidden)
            next_latent = self.actor.encode_step(next_obs_b, h_next)
            next_actions, next_logp = self.actor.act_and_log_prob_from_latent(next_latent)
            q1_t = self.critic_1_target(next_obs_b, next_actions).view(-1)
            q2_t = self.critic_2_target(next_obs_b, next_actions).view(-1)
            min_q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
            target_q = rewards_b + self.gamma * not_terminated * min_q_t
        q1 = self.critic_1(obs_b, actions_b).view(-1)
        q2 = self.critic_2(obs_b, actions_b).view(-1)
        critic_1_loss = nn.functional.mse_loss(q1, target_q)
        critic_2_loss = nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        (critic_1_loss + critic_2_loss).backward()
        if self.is_multi_gpu:
            self.reduce_parameters(self.critic_parameters)
        acc("Critic/grad_norm", nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm))
        self.critic_optimizer.step()
        self._last_critic_losses = (critic_1_loss.item(), critic_2_loss.item())
        with torch.no_grad():
            acc("Critic/q_data", q1.mean())
            acc("Critic/q_target", target_q.mean())
            acc("Critic/td_abs", (q1 - target_q).abs().mean())
            acc("Critic/reward", rewards_b.mean())

    def _recurrent_state_diagnostics(
        self, batch: ReplayBuffer.SequenceBatch, h0: HiddenState, acc: Callable[[str, torch.Tensor | float], None]
    ) -> None:
        """Hidden-state health of one window batch, from the re-derived state at the loss boundary.

        ``state_staleness`` compares that state with the one the rollout actor actually held there, over
        windows whose burn-in stayed inside one episode; ``init_carryover`` is how much of it still
        depends on the stored start state after burn-in.
        """
        burn = batch.burn_in
        num_windows = batch.resets.shape[1]
        h_now = _first_slot(h0).transpose(0, 1).reshape(num_windows, -1)
        same_episode = 1.0 - (batch.resets[: burn + 1].sum(dim=0) > 0).float()
        same_sum = same_episode.sum().clamp(min=1.0)
        acc("Recurrent/episode_starts_per_window", batch.resets[burn:].sum(dim=0).mean())
        acc("Recurrent/burn_in_crosses_episode", 1.0 - same_episode.mean())
        acc("Recurrent/hidden_abs_train", h_now.abs().mean())
        acc("Recurrent/hidden_saturation", (h_now.abs() > 0.95).float().mean())
        if batch.ages is not None:
            acc("Recurrent/sample_age", batch.ages.mean())
        if batch.init_hidden is not None:
            h_init = _first_slot(batch.init_hidden).transpose(0, 1).reshape(num_windows, -1)
            acc("Recurrent/zero_init_frac", (h_init.abs().sum(dim=1) == 0).float().mean())
        if batch.burn_hidden is not None:
            h_stored = _first_slot(batch.burn_hidden).transpose(0, 1).reshape(num_windows, -1)
            rel = (h_now - h_stored).norm(dim=1) / h_stored.norm(dim=1).clamp(min=1e-6)
            acc("Recurrent/state_staleness", (rel * same_episode).sum() / same_sum)
        if burn > 0:
            _, h_zero = self.actor.encode_sequence(batch.observations[:burn], None, batch.resets[:burn])
            h_zero = _first_slot(h_zero).transpose(0, 1).reshape(num_windows, -1)
            rel = (h_now - h_zero).norm(dim=1) / h_now.norm(dim=1).clamp(min=1e-6)
            acc("Recurrent/init_carryover", (rel * same_episode).sum() / same_sum)

    def _update_critics(
        self,
        batch: ReplayBuffer.Batch,
        obs_b: TensorDict,
        next_obs_b: TensorDict,
        actions_b: torch.Tensor,
        rewards_b: torch.Tensor,
        not_terminated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One twin-critic gradient step; returns the two critic losses."""
        self.critic_optimizer.zero_grad()
        critic_1_loss, critic_2_loss = self._critic_losses_and_backward(
            batch, obs_b, next_obs_b, actions_b, rewards_b, not_terminated
        )
        # clone in eager context: compiled outputs live in the cudagraph pool and are invalidated by the
        # next replay of any graph (the in-graph clone does not escape the pool)
        critic_1_loss, critic_2_loss = critic_1_loss.clone(), critic_2_loss.clone()
        if self.is_multi_gpu:
            self.reduce_parameters(self.critic_parameters)
        nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
        self.critic_optimizer.step()
        return critic_1_loss, critic_2_loss

    def _critic_losses_and_backward(
        self,
        batch: ReplayBuffer.Batch,
        obs_b: TensorDict,
        next_obs_b: TensorDict,
        actions_b: torch.Tensor,
        rewards_b: torch.Tensor,
        not_terminated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled region: twin-critic losses + backward (no optimizer ops, no graph breaks)."""
        with torch.no_grad():
            next_actions, next_logp = self.actor.act_and_log_prob(next_obs_b)
            q1_t = self.critic_1_target(next_obs_b, next_actions).view(-1)
            q2_t = self.critic_2_target(next_obs_b, next_actions).view(-1)
            min_q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
            # n-step: discount by the per-sample horizon actually aggregated (capped at episode ends);
            # single-step: gamma^1. The buffer sets effective_n_steps only when n_steps > 1.
            if batch.effective_n_steps is not None:
                discount = self.gamma ** batch.effective_n_steps.view(-1).float()
            else:
                discount = self.gamma**self.n_steps
            target_q = rewards_b + discount * not_terminated * min_q_t

        q1 = self.critic_1(obs_b, actions_b).view(-1)
        q2 = self.critic_2(obs_b, actions_b).view(-1)
        critic_1_loss = nn.functional.mse_loss(q1, target_q)
        critic_2_loss = nn.functional.mse_loss(q2, target_q)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        return critic_1_loss.detach(), critic_2_loss.detach()

    def _reference_bc_weight(self) -> float:
        """Return the current weight of the pull toward the frozen student (linear decay to zero, or constant)."""
        if self.reference_bc_decay_updates <= 0:
            return self.reference_bc_weight
        return self.reference_bc_weight * max(0.0, 1.0 - self.update_step / self.reference_bc_decay_updates)

    def _update_actor(self, obs_b: TensorDict, new_actions: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """One actor gradient step against the frozen critics; returns the actor loss."""
        self.actor_optimizer.zero_grad()
        actor_loss = self._actor_loss_and_backward(obs_b, new_actions, logp).clone()
        if self.reference_actor is not None:
            with torch.no_grad():
                ref_mean = self.reference_actor(obs_b)
            ref_bc = self._reference_bc_weight() * (self.actor.distribution.mean - ref_mean).pow(2).sum(-1).mean()  # type: ignore[attr-defined]
            ref_bc.backward()
            actor_loss = actor_loss + ref_bc.detach()
        if self.is_multi_gpu:
            self.reduce_parameters(self.actor_parameters)
        nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
        self.actor_optimizer.step()
        return actor_loss

    def _actor_loss_and_backward(
        self, obs_b: TensorDict, new_actions: torch.Tensor, logp: torch.Tensor
    ) -> torch.Tensor:
        """Compiled region: actor loss + backward (no optimizer ops, no graph breaks)."""
        q1_pi = self.critic_1(obs_b, new_actions).view(-1)
        q2_pi = self.critic_2(obs_b, new_actions).view(-1)
        actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1_pi, q2_pi)).mean()
        actor_loss.backward()
        return actor_loss.detach()

    def train_mode(self) -> None:
        """Set the actor and critics to training mode."""
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set the actor and critics to evaluation mode."""
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        if self.rnd:
            self.rnd.eval()

    def get_policy(self) -> MLPModel:
        """Return the actor (policy) model."""
        return self.actor

    def log_info(self) -> dict:
        """Extra per-iteration values for the runner's log call (learning rate, action std, RND weight)."""
        return {
            "learning_rate": self.actor_learning_rate,
            "action_std": self.get_policy().output_std,
            "rnd_weight": self.rnd.weight if self.rnd else None,
        }

    def eval(self, env: VecEnv, max_steps: int = 200, stochastic: bool = False, action_repeat: int = 1) -> list[dict]:
        """Roll out the policy for ``max_steps`` env steps (each ``env.step`` renders a frame for a video wrapper).

        Args:
            env: Vectorized environment to roll out in.
            max_steps: Number of environment steps to run.
            stochastic: Act with the deterministic squashed mean when False; sample the action when True.
            action_repeat: Hold each queried action for this many ``env.step`` calls before re-querying the actor.

        Returns:
            An empty list (this rollout collects no eval metrics); present for parity with the other algorithms
            so the out-of-process video recorder can drive SAC the same way.
        """
        was_training = self.actor.training
        self.eval_mode()
        if hasattr(env, "eval_mode"):
            env.eval_mode()

        obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()[0]
        if self.recurrent:
            self.actor.reset()
        action_repeat = max(1, action_repeat)

        with torch.inference_mode():
            actions = self.actor(obs, stochastic_output=stochastic)
            for step in range(max_steps):
                if step > 0 and step % action_repeat == 0:
                    actions = self.actor(obs, stochastic_output=stochastic)
                obs, _, _, _ = env.step(actions)

        if self.recurrent:
            self.actor.reset()
        if was_training:
            self.train_mode()
            if hasattr(env, "train_mode"):
                env.train_mode()
        return []

    def reset_rollout_state(self) -> None:
        """Clear rollout-only state after the env is reset underneath the agent."""
        if self.recurrent:
            self.actor.reset()

    def save(self) -> dict:
        """Return a dict of model/optimizer/temperature states for checkpointing."""
        saved = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.auto_alpha and self.alpha_optimizer is not None:
            saved["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        if self.rnd:
            saved["rnd_state_dict"] = self.rnd.state_dict()
        if self.reference_actor is not None:
            saved["reference_actor_state_dict"] = self.reference_actor.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load model/optimizer/temperature states; targets are re-synced from the loaded critics."""
        from_student = "actor_state_dict" not in loaded_dict and "student_state_dict" in loaded_dict
        if from_student:
            # a distillation checkpoint: its student becomes the actor, everything else starts fresh
            loaded_dict = {**loaded_dict, "actor_state_dict": loaded_dict["student_state_dict"]}
        if load_cfg is None:
            full = "critic_1_state_dict" in loaded_dict
            load_cfg = {"actor": True, "critic": full, "optimizer": full, "iteration": full, "rnd": full}
        if load_cfg.get("actor") and from_student:
            # the BC student has no aux head, so that stays fresh; any other mismatch is a wrong architecture
            result = self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=False)
            bad = [k for k in result.missing_keys if not k.startswith("aux_head")] + list(result.unexpected_keys)
            if bad:
                raise RuntimeError(f"student checkpoint does not match the actor: {bad}")
        elif load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("actor") and self.reference_bc_weight > 0.0:
            # the pull's target survives a resume; a checkpoint without one falls back to the actor as loaded
            self.reference_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
            if "reference_actor_state_dict" in loaded_dict:
                self.reference_actor.load_state_dict(loaded_dict["reference_actor_state_dict"], strict=False)
        if load_cfg.get("critic"):
            self.critic_1.load_state_dict(loaded_dict["critic_1_state_dict"], strict=strict)
            self.critic_2.load_state_dict(loaded_dict["critic_2_state_dict"], strict=strict)
            self.critic_1_target.hard_sync()
            self.critic_2_target.hard_sync()
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.auto_alpha and "alpha_optimizer_state_dict" in loaded_dict:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            if "log_alpha" in loaded_dict:
                self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp().item()
        if load_cfg.get("rnd") and self.rnd and "rnd_state_dict" in loaded_dict:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters from rank 0 to all GPUs."""
        params = [self.actor.state_dict(), self.critic_1.state_dict(), self.critic_2.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self.actor.load_state_dict(params[0])
        self.critic_1.load_state_dict(params[1])
        self.critic_2.load_state_dict(params[2])
        self.critic_1_target.hard_sync()
        self.critic_2_target.hard_sync()

    def reduce_parameters(self, parameters: list) -> None:
        """Average gradients across GPUs for the given parameters (in place)."""
        grads = [p.grad.view(-1) for p in parameters if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in parameters:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str, inference: bool = False) -> SAC:
        """Build the SAC algorithm from a config dict.

        ``inference=True`` builds a minimal replay buffer for play/eval.
        """
        alg_class: type[SAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FuseModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        num_actions = env.num_actions
        aux_group = cfg["algorithm"].get("aux_obs_group")
        if aux_group is not None:
            cfg["actor"]["aux_target_dim"] = obs[aux_group].shape[-1]
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", num_actions, **cfg["actor"]).to(device)
        # Twin Q-critics: generic FuseModel fusing obs + action -> scalar Q.
        critic_1: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)
        critic_2: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)

        buffer_size = int(cfg["algorithm"].get("replay_buffer_size", 1_000_000))
        capacity_per_env = 1 if inference else max(buffer_size // env.num_envs, 1)
        storage_device = cfg.get("storage_device") or device
        hidden_kwargs: dict = {}
        if actor.is_recurrent:
            rnn = actor.rnn.rnn  # type: ignore[attr-defined]
            hidden_kwargs = {
                "hidden_dim": rnn.hidden_size,
                "hidden_layers": rnn.num_layers,
                "hidden_is_lstm": actor.rnn.is_lstm,  # type: ignore[attr-defined]
            }
        replay_buffer = ReplayBuffer(
            env.num_envs,
            capacity_per_env,
            obs,
            [num_actions],
            z_dim=0,
            batch_size=cfg["algorithm"].get("mini_batch_size", 256),
            device=storage_device,
            keep_terminal=True,
            n_steps=cfg["algorithm"].get("n_steps", 1),
            gamma=cfg["algorithm"].get("gamma", 0.99),
            **hidden_kwargs,
        )

        return alg_class(
            actor,
            critic_1,
            critic_2,
            replay_buffer,
            num_actions,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
