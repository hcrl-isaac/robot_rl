"""FB-CPR whose actor is improved by backpropagation through imagined world-model rollouts."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.fb_cpr import FbCpr
from robot_rl.algorithms.world_model_ac import LEARNER_KEYS
from robot_rl.algorithms.world_model_learner import Rollout, WorldModelLearner
from robot_rl.env import URLVecEnv
from robot_rl.storage import ReplayBuffer, SequenceReplayBuffer
from robot_rl.utils import compute_td_targets, eval_mode


@dataclass
class CriticExpansion:
    """k-step critic targets from a world-model rollout started at each transition's real next state."""

    obs: TensorDict
    """Normalized observations at the bootstrap state ``s_{k-1}``."""

    gammas: torch.Tensor
    """Bootstrap discount ``gamma^k`` (zero past a real termination), ``(B, 1)``."""

    credit: torch.Tensor
    """``sum_{i=1}^{k-1} gamma^i B(s_i)`` over the imagined states, detached, ``(B, z_dim)``."""

    disc_rewards: torch.Tensor
    """``sum_{i=1}^{k-1} gamma^i r_D(s_{i-1})``, ``(B, 1)``."""

    aux_rewards: torch.Tensor
    """``sum_{i=1}^{k-1} gamma^i r_{i-1}`` (the world model's reward head), at the aux critic's scale, ``(B, 1)``."""


class WorldModelFbCpr(FbCpr):
    """FB-CPR with the actor trained on ``H``-step imagined returns instead of one-step critic values.

    The forward-backward maps, the discriminator and both critics train on replay exactly as in :class:`FbCpr`
    (replay is a :class:`SequenceReplayBuffer` that also logs each step's ``z``). Each update also fits a world model
    on replay windows, then rolls the actor ``H`` steps through it from the update's replay states, each with its
    batch ``z``, and ascends

        ``sum_t gamma^t c_t [r_fb(s_{t+1}) + w (a_D r_D(s_t) + a_aux r_aux_t)]
        + gamma^H c_H [Q_fb + w (a_D Q_D + a_aux Q_aux)](s_H, a_H)``

    where ``r_fb = B(s')^T Cov(B)^-1 z`` is FB's implicit reward, ``r_D`` the discriminator's log-odds, ``r_aux`` the
    world model's reward head (the env's auxiliary reward, scaled as the aux critic's targets are), ``c`` the soft
    survival weight, ``w = |Q_fb|`` and ``a_D, a_aux`` FB-CPR's regularization coefficients. With ``H = 0`` this is
    FB-CPR's own actor loss.
    """

    def __init__(
        self,
        *args: Any,
        learner: WorldModelLearner,
        num_imagined: int | None = None,
        imagined_actor_interval: int = 1,
        actor_start_update: int = 0,
        actor_max_grad_norm: float | None = 1.0,
        actor_grad_clip_value: float | None = 0.1,
        tf32: bool = True,
        imagination_compile_mode: str | None = "max-autotune",
        mve_k: int = 1,
        mve_critics: tuple[str, ...] = ("forward", "disc", "aux"),
        mage_beta: float = 0.0,
        mage_q_beta: float | None = None,
        mage_batch_size: int | None = None,
        mage_max_grad_norm: float | None = 1.0,
        mage_directions: int | None = None,
        mage_fd_eps: float = 0.01,
        mage_action: str = "policy",
        critic_model_start_update: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            *args: :class:`FbCpr` positional arguments.
            learner: The world model, its replay (also passed as ``replay_buffer``) and its imagination.
            num_imagined: Rollouts per imagined update, drawn from the update's batch; None imagines from all of it.
            imagined_actor_interval: Updates per imagined actor update; the others take FB-CPR's one-step loss.
            actor_start_update: Updates that train the actor on FB-CPR's one-step loss while the model is fit.
            actor_max_grad_norm: Global gradient-norm clip of the imagined actor update; None disables it.
            actor_grad_clip_value: Element-wise clip of the imagined actor gradients after the norm clip.
            tf32: Allow TF32 matmuls (a process-wide torch setting).
            imagination_compile_mode: ``torch.compile`` mode of the imagined loss; None runs it eagerly.
            mve_k: Steps of the critics' targets: the real next state plus ``k - 1`` imagined ones (1 = FB-CPR's).
            mve_critics: Which critics take the k-step targets: ``"forward"`` (F and B), ``"disc"``, ``"aux"``.
            mage_beta: Weight of F's action-gradient (MAGE) loss through the world model (0 = off).
            mage_q_beta: Weight of the discriminator and aux critics' MAGE losses; None uses ``mage_beta``.
            mage_batch_size: Rows of the update's batch MAGE differentiates on; None uses all of them.
            mage_max_grad_norm: Norm clip of each critic's MAGE parameter gradient (FB-CPR's critics run unclipped,
                and a second-order gradient can be large); None disables it.
            mage_directions: Random unit directions of a central finite-difference estimate of ||∂δ/∂a||²
                (first-order in the critic's parameters); None differentiates exactly by double backprop.
            mage_fd_eps: Finite-difference step in action units.
            mage_action: Where MAGE differentiates: ``"policy"`` (the actor's action) or ``"buffer"`` (the logged one).
            critic_model_start_update: Updates before MVE and MAGE start (the world model's warm-up).
            **kwargs: :class:`FbCpr` keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if mve_k < 1:
            raise ValueError(f"mve_k must be at least 1, got {mve_k}.")
        if mve_k > 1 and self.value_loss_coef > 0:
            raise ValueError("mve_k > 1 does not expand the F value regularizer's target; set value_loss_coef = 0.")
        if mage_action not in ("policy", "buffer"):
            raise ValueError(f"mage_action must be 'policy' or 'buffer', got '{mage_action}'.")
        unknown = set(mve_critics) - {"forward", "disc", "aux"}
        if unknown:
            raise ValueError(f"mve_critics may name 'forward', 'disc' and 'aux', got {sorted(unknown)}.")
        self.mve_k = mve_k
        self.mve_critics = tuple(mve_critics)
        self.mage_beta = mage_beta
        self.mage_q_beta = mage_beta if mage_q_beta is None else mage_q_beta
        self.mage_batch_size = mage_batch_size
        self.mage_max_grad_norm = mage_max_grad_norm
        self.mage_directions = mage_directions
        self.mage_fd_eps = mage_fd_eps
        self.mage_action = mage_action
        self.critic_model_start_update = critic_model_start_update
        self.learner = learner
        self.world_model = learner.world_model
        self.num_imagined = num_imagined
        self.imagined_actor_interval = imagined_actor_interval
        self.actor_start_update = actor_start_update
        self.actor_max_grad_norm = actor_max_grad_norm
        self.actor_grad_clip_value = actor_grad_clip_value
        self.update_step = 0
        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._compiled_imagined_loss = (
            torch.compile(self._imagined_loss, mode=imagination_compile_mode)
            if imagination_compile_mode is not None
            else self._imagined_loss
        )

    """
    Collection.
    """

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Log the step (with its ``z``) to the sequence replay and advance the rollout state."""
        t = self.transition
        self.learner.add(t.observations, t.actions, rewards, dones, extras, {"z": t.context})  # type: ignore[arg-type]
        t.clear()
        for model in self.models:
            model.reset(dones)
        if self._cur_episode_length is not None:
            self._cur_episode_length += 1
            self._cur_episode_length[(dones > 0).nonzero(as_tuple=False)] = 0

    def reset_rollout_state(self) -> None:
        """Reset the rollout bookkeeping after an external env reset; the next logged row starts an episode."""
        super().reset_rollout_state()
        self.learner.replay_buffer.mark_episode_start()

    def compute_gammas(self) -> None:
        """Do nothing: each sampled batch derives its discounts from the logged terminations."""

    """
    Updates.
    """

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Run FB-CPR's replay updates, fit the world model, then update the actor through imagination."""
        with torch.profiler.record_function("wm_fbcpr/sample"):
            win = self.learner.replay_buffer.sample_transitions(self.batch_size, device=self.device)
        if win is None:
            return {"Actor_Loss/skipped": torch.ones(())}, {}
        starts = win.obs[0]
        batch = self._as_batch(win)
        expert_obs, expert_next_obs = self.expert_buffer.sample(
            self.batch_size, self.device, seq_length=self.expert_sequence_length
        )
        self.obs_normalizer(batch.observations)
        self.obs_normalizer(batch.next_observations)
        with torch.no_grad(), eval_mode(self.obs_normalizer):
            batch.observations = self.obs_normalizer(batch.observations)
            batch.next_observations = self.obs_normalizer(batch.next_observations)
            expert_obs = self.obs_normalizer(expert_obs)
            expert_next_obs = self.obs_normalizer(expert_next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        with torch.profiler.record_function("wm_fbcpr/fb_cpr_updates"):
            expert_z = self._encode_expert(expert_next_obs)
            disc_loss, disc_extras = self._update_discriminator(batch, expert_obs, expert_z)
            z = self._sample_mixed_z(batch.next_observations, expert_z)
            self.z_buffer.add(z)
            relabel_mask = torch.rand(self.batch_size, 1, device=self.device) < self.z_relabel_ratio
            batch.context = torch.where(relabel_mask, z, batch.context)
            batch.rewards = self.aux_reward_normalizer(batch.rewards)
            fb_kwargs, disc_kwargs, aux_kwargs, model_extras = self._model_critic_inputs(win, batch)
            if "forward" in model_extras:
                fb_kwargs["keep_grads"] = True
            fb_loss, fb_extras = self._update_forward_backward(batch, **fb_kwargs)
            if self._mage_on() and self.mage_q_beta > 0:
                model_extras.update(self._mage_backward("disc", win, batch))
                disc_kwargs["keep_grads"] = True
            disc_critic_loss, disc_critic_extras = self._update_disc_critic(batch, **disc_kwargs)
            if self._mage_on() and self.mage_q_beta > 0:
                model_extras.update(self._mage_backward("aux", win, batch))
                aux_kwargs["keep_grads"] = True
            aux_critic_loss, aux_critic_extras = self._update_aux_critic(batch, **aux_kwargs)
            model_extras.pop("forward", None)

        with torch.profiler.record_function("wm_fbcpr/world_model_fit"):
            wm_logs = self.learner.fit(self._reduce_and_clip_wm)
        imagined = self.update_step >= self.actor_start_update and self.update_step % self.imagined_actor_interval == 0
        horizon = self.learner.horizon if imagined else 0
        z = batch.context.clone()
        if horizon > 0 and self.num_imagined is not None and self.num_imagined < self.batch_size:
            rows = torch.randperm(self.batch_size, device=self.device)[: self.num_imagined]
            starts, z = starts[rows], z[rows]
        with torch.profiler.record_function("wm_fbcpr/imagined_actor"):
            actor_loss, actor_extras = self._update_imagined_actor(starts, z, batch.next_observations, horizon)
        self.update_step += 1

        loss_dict = {**disc_loss, **fb_loss, **disc_critic_loss, **aux_critic_loss, **actor_loss}
        extras = {**disc_extras, **fb_extras, **disc_critic_extras, **aux_critic_extras, **actor_extras, **model_extras}
        with torch.no_grad():
            self._soft_update_targets()
            loss_dict = {k: v.clone() for k, v in loss_dict.items()}
            extras = {k: v.clone() for k, v in extras.items()}
        loss_dict.update(wm_logs)
        return loss_dict, extras

    """
    World-model critics.
    """

    def _mage_on(self) -> bool:
        return (self.mage_beta > 0 or self.mage_q_beta > 0) and self.update_step >= self.critic_model_start_update

    def _model_critic_inputs(
        self, win: SequenceReplayBuffer.Window, batch: ReplayBuffer.Batch
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
        """Return the FB, discriminator-critic and aux-critic update arguments for MVE, after MAGE's F gradient if on.

        Args:
            win: The update's one-step replay windows (raw observations).
            batch: The same transitions, normalized, with their latents and normalized aux rewards.

        Returns:
            The three updates' keyword arguments and the logged diagnostics (``"forward"`` marks a MAGE gradient
            already accumulated on F).
        """
        fb_kwargs: dict[str, Any] = {}
        disc_kwargs: dict[str, Any] = {}
        aux_kwargs: dict[str, Any] = {}
        extras: dict[str, torch.Tensor] = {}
        if self.update_step < self.critic_model_start_update:
            return fb_kwargs, disc_kwargs, aux_kwargs, extras
        if self.mve_k > 1:
            exp = self._critic_expansion(win.obs[1], batch)
            if "forward" in self.mve_critics:
                fb_kwargs.update(bootstrap_obs=exp.obs, bootstrap_gammas=exp.gammas, credit_backward=exp.credit)
            if "disc" in self.mve_critics:
                disc_kwargs.update(bootstrap_obs=exp.obs, bootstrap_gammas=exp.gammas, extra_rewards=exp.disc_rewards)
            if "aux" in self.mve_critics:
                aux_kwargs.update(bootstrap_obs=exp.obs, bootstrap_gammas=exp.gammas, extra_rewards=exp.aux_rewards)
            extras["mve_fb_credit_norm"] = exp.credit.norm(dim=-1).mean()
            extras["mve_disc_reward"] = exp.disc_rewards.mean()
            extras["mve_aux_reward"] = exp.aux_rewards.mean()
        if self._mage_on() and self.mage_beta > 0:
            extras.update(self._mage_backward("forward", win, batch))
            extras["forward"] = torch.ones(())
        return fb_kwargs, disc_kwargs, aux_kwargs, extras

    @torch.no_grad()
    def _critic_expansion(self, next_raw: TensorDict, batch: ReplayBuffer.Batch) -> CriticExpansion:
        """Roll ``k - 1`` steps from each real next state under the current policy and sum the critics' rewards.

        Args:
            next_raw: The transitions' real next states, raw, ``(B, ...)``.
            batch: The same transitions, normalized, with their latents.

        Returns:
            The expanded targets' pieces (rewards at imagined states; ``B`` there detached).
        """
        k, n, z = self.mve_k, self.batch_size, batch.context
        alive = (batch.gammas > 0).float()

        def policy(raw: TensorDict) -> torch.Tensor:
            return self.actor(self._normalize(raw), z, stochastic_output=True, std_clip=self.clip_actor_std)

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            goals = self.learner.start_goals(next_raw, k - 1)
            rollout = self.learner.imagine(policy, next_raw, goals, self.learner.noise_bank(), horizon=k - 1)
            norm = self._normalize(torch.stack(rollout.obs).reshape(-1)).reshape(k, n)
            weights = self.gamma ** torch.arange(1, k, device=self.device, dtype=torch.float32)
            B_imag = self.backward_map(norm[1:].reshape(-1)).float().view(k - 1, n, -1)
            credit = (weights[:, None, None] * B_imag).sum(dim=0) * alive
            eps = self.discriminator_reward_eps
            logits = self.discriminator(norm[: k - 1].reshape(-1), z.repeat(k - 1, 1)).view(k - 1, n)
            logits = logits.clamp(eps, 1 - eps)
            r_disc = torch.log(logits) - torch.log(1 - logits)
            r_aux = rollout.rewards * self.aux_reward_normalizer.S.sqrt().reciprocal()
            return CriticExpansion(
                obs=norm[k - 1],
                gammas=(self.gamma**k) * alive,
                credit=credit,
                disc_rewards=(weights[:, None] * r_disc).sum(dim=0, keepdim=True).T * alive,
                aux_rewards=(weights[:, None] * r_aux).sum(dim=0, keepdim=True).T * alive,
            )

    def _mage_backward(
        self, critic: str, win: SequenceReplayBuffer.Window, batch: ReplayBuffer.Batch
    ) -> dict[str, torch.Tensor]:
        """Write the gradient of ``beta`` E||∂δ/∂a||² into one critic (its optimizer's gradients reset first).

        δ is the critic's one-step TD residual with the next state from one world-model step, so the model's action
        Jacobian shapes the critic's action slope. The squared slope comes exactly (double backprop) or from central
        finite differences along ``mage_directions`` random unit directions. Only the trained critic gets gradient,
        norm-clipped, and a non-finite one is dropped.

        Args:
            critic: ``"forward"`` (F, through ``F^T z`` and B's implicit reward), ``"disc"`` or ``"aux"``.
            win: The update's one-step replay windows (raw observations).
            batch: The same transitions, normalized, with their latents and normalized aux rewards.

        Returns:
            The action-slope and parameter-gradient norms and whether the gradient was dropped, for logging.
        """
        trained, optimizer = {
            "forward": (self.forward_map, self.forward_optimizer),
            "disc": (self.disc_critic, self.disc_critic_optimizer),
            "aux": (self.aux_critic, self.aux_critic_optimizer),
        }[critic]
        frozen = [self.actor, self.backward_map, self.discriminator, self.world_model]
        frozen += [m for m in (self.forward_map, self.disc_critic, self.aux_critic) if m is not trained]
        for net in frozen:
            net.requires_grad_(False)
        z, obs, alive = batch.context, batch.observations, (batch.gammas > 0).float().squeeze(-1)
        actions, starts, next_real = batch.actions, win.obs[0], win.obs[1]
        if self.mage_batch_size is not None and self.mage_batch_size < self.batch_size:
            rows = torch.randperm(self.batch_size, device=self.device)[: self.mage_batch_size]
            z, obs, alive, actions = z[rows], obs[rows], alive[rows], actions[rows]
            starts, next_real = starts[rows], next_real[rows]
        with torch.no_grad():
            reward_z = self._implicit_reward_weights(batch.next_observations, z) if critic == "forward" else None
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            if self.mage_action == "policy":
                with torch.no_grad():
                    actions = self.actor(obs, z, stochastic_output=True, std_clip=self.clip_actor_std)
            actions = actions.detach()
            # the noise does not depend on the action: a noise-free step keeps paired differences exact
            bank = {g: torch.zeros_like(b[:1]) for g, b in self.learner.noise_bank().items()}
            base = None
            if self.mage_action == "buffer":
                with torch.no_grad():
                    base = self._model_step(starts, actions, bank)[0]
            if self.mage_directions is None:
                action = actions.clone().requires_grad_(True)
                delta = self._mage_residual(critic, obs, z, alive, action, starts, next_real, base, bank, reward_z)
                sq = torch.stack([
                    torch.autograd.grad(delta[p].sum(), action, create_graph=True)[0].pow(2).sum(dim=-1)
                    for p in range(delta.shape[0])
                ]).mean()
            else:
                k, n, eps = self.mage_directions, actions.shape[0], self.mage_fd_eps
                u = nn.functional.normalize(torch.randn(k, n, actions.shape[-1], device=self.device), dim=-1)
                shifted = torch.cat([actions + eps * u, actions - eps * u]).reshape(2 * k * n, -1)

                def rep(x: Any) -> Any:
                    return None if x is None else torch.cat([x] * (2 * k), dim=0)

                delta = self._mage_residual(
                    critic,
                    rep(obs),
                    rep(z),
                    rep(alive),
                    shifted,
                    rep(starts),
                    rep(next_real),
                    rep(base),
                    bank,
                    rep(reward_z),
                ).view(-1, 2, k, n)
                slope = (delta[:, 0] - delta[:, 1]) / (2 * eps)
                # a random unit direction u has E[(u . g)^2] = |g|^2 / d
                sq = actions.shape[-1] * slope.pow(2).mean()
            loss = (self.mage_beta if critic == "forward" else self.mage_q_beta) * sq
        params = [p for p in trained.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads if g is not None]))
        finite = torch.isfinite(norm)
        scale = torch.ones_like(norm)
        if self.mage_max_grad_norm is not None:
            scale = torch.clamp(self.mage_max_grad_norm / (norm + 1e-6), max=1.0)
        # a non-finite gradient is dropped (scaled to zero) rather than written into the critic
        scale = torch.where(finite, scale, torch.zeros_like(scale))
        optimizer.zero_grad()
        for p, g in zip(params, grads, strict=True):
            if g is not None:
                # in the parameter's own layout, as the fused optimizer step requires
                p.grad = torch.zeros_like(p).copy_(torch.nan_to_num(g) * scale)
        for net in frozen:
            net.requires_grad_(True)
        return {
            f"mage_{critic}_action_grad": sq.detach().sqrt(),
            f"mage_{critic}_param_grad_norm": norm.detach(),
            f"mage_{critic}_dropped": (~finite).float(),
        }

    def _model_step(
        self, starts: TensorDict, actions: torch.Tensor, bank: dict[str, torch.Tensor]
    ) -> tuple[TensorDict, torch.Tensor]:
        """One world-model step from raw ``starts`` under ``actions``: the next raw groups and the reward."""
        goals = self.learner.start_goals(starts, 1)
        step = self.learner.imagine(lambda _: actions, starts, goals, bank, freeze_model=False, horizon=1)
        return step.obs[1], step.rewards[0]

    def _mage_residual(
        self,
        critic: str,
        obs: TensorDict,
        z: torch.Tensor,
        alive: torch.Tensor,
        action: torch.Tensor,
        starts: TensorDict,
        next_real: TensorDict,
        base: TensorDict | None,
        bank: dict[str, torch.Tensor],
        reward_z: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one critic's TD residual per member, ``(P, N)``, with the next state from a model step at ``action``.

        With ``base`` (buffer mode) the next state is the real one moved by the model's change from ``base``.
        """
        next_raw, reward_model = self._model_step(starts, action, bank)
        if base is not None:
            next_raw = next_real.select(*next_raw.keys()) + (next_raw - base)
        next_obs = self._normalize(next_raw)
        next_action = self.actor(next_obs, z)
        if critic == "forward":
            q = (self.forward_map(obs, z, action) * z).sum(dim=-1)
            reward = (self.backward_map(next_obs).float() * reward_z).sum(dim=-1)
            q_next = (self.target_forward_map.target(next_obs, z, next_action) * z).sum(dim=-1).mean(dim=0)
        elif critic == "disc":
            q = self.disc_critic(obs, z, action).squeeze(-1)
            reward = torch.zeros_like(alive)
            q_next = self.target_disc_critic.target(next_obs, z, next_action).squeeze(-1).mean(dim=0)
        else:
            q = self.aux_critic(obs, z, action).squeeze(-1)
            reward = reward_model * self.aux_reward_normalizer.S.sqrt().reciprocal()
            q_next = self.target_aux_critic.target(next_obs, z, next_action).squeeze(-1).mean(dim=0)
        return q - (reward + self.gamma * alive * q_next)

    def _update_imagined_actor(
        self, starts: TensorDict, z: torch.Tensor, cov_obs: TensorDict, horizon: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Ascend the imagined return from ``starts`` under their ``z`` (see the class docstring).

        Args:
            starts: Raw replay rows to start from, ``(B, ...)``.
            z: Each rollout's latent, ``(B, z_dim)``.
            cov_obs: Normalized observations whose ``B`` sets the implicit reward's covariance.
            horizon: Imagined steps; 0 is FB-CPR's one-step loss.

        Returns:
            The loss and extras dicts.
        """
        frozen = (self.forward_map, self.backward_map, self.discriminator, self.disc_critic, self.aux_critic)
        frozen += (self.world_model,)
        for net in frozen:
            net.requires_grad_(False)
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            if horizon > 0:
                goals, bank = self.learner.start_goals(starts), self.learner.noise_bank()
                reward_z = self._implicit_reward_weights(cov_obs, z)
                # an input, not the normalizer's buffers, whose storage it reallocates each update
                aux_scale = self.aux_reward_normalizer.S.sqrt().reciprocal().detach()
                actor_loss, extras = self._compiled_imagined_loss(starts, z, reward_z, aux_scale, goals, bank, horizon)
            else:
                actor_loss, extras = self._imagined_loss(starts, z, z, z.new_ones(1), None, {}, 0)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters(self.actor)
        if horizon > 0:
            if self.actor_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad_norm)
            if self.actor_grad_clip_value is not None:
                nn.utils.clip_grad_value_(self.actor.parameters(), self.actor_grad_clip_value)
        elif self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        for net in frozen:
            net.requires_grad_(True)
        loss_dict = {"Actor_Loss/total_loss": actor_loss.detach().clone(), "Actor_Loss/horizon": torch.tensor(horizon)}
        return loss_dict, {k: v.clone() for k, v in extras.items()}

    def _imagined_loss(
        self,
        starts: TensorDict,
        z: torch.Tensor,
        reward_z: torch.Tensor,
        aux_scale: torch.Tensor,
        goals: TensorDict | None,
        noise_bank: dict[str, torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Roll out and score the actor's imagined return (the compiled region: no module state changes inside).

        Args:
            starts: Raw replay rows to start from, ``(B, ...)``.
            z: Each rollout's latent, ``(B, z_dim)``.
            reward_z: ``Cov(B)^-1 z``, so FB's implicit reward is ``B(s')^T reward_z``.
            aux_scale: Scale of the world model's aux reward, as the aux critic's targets are normalized.
            goals: The goal groups at steps ``0 .. H`` (None without goals).
            noise_bank: Sensor-noise rows per group.
            horizon: Imagined steps; 0 is FB-CPR's one-step loss.

        Returns:
            The actor loss and the mean reward terms for logging.
        """

        def policy(raw: TensorDict) -> torch.Tensor:
            return self.actor(self._normalize(raw), z, stochastic_output=True, std_clip=self.clip_actor_std)

        if horizon > 0:
            rollout = self.learner.imagine(policy, starts, goals, noise_bank, freeze_model=False)
        else:
            rollout = self._empty_rollout(starts)
        ret, terms = self._imagined_return(rollout, policy, z, reward_z, aux_scale, horizon)
        extras = {f"imagined_{k}": v.detach().mean() for k, v in terms.items()}
        extras["imagined_survival"] = rollout.survival[-1].detach().mean()
        return -ret.mean(), extras

    def _imagined_return(
        self,
        rollout: Rollout,
        policy: Any,
        z: torch.Tensor,
        reward_z: torch.Tensor,
        aux_scale: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return each rollout's survival-weighted imagined return and its per-step reward terms (for logging)."""
        # every step normalized and scored in one batched call per network, steps flattened into the batch
        n = z.shape[0]
        norm = self._normalize(torch.stack(rollout.obs).reshape(-1)).reshape(horizon + 1, n)
        norm_h = norm[horizon]
        action_h = policy(rollout.obs[horizon])
        Qs_fb = (self.forward_map(norm_h, z, action_h) * z).sum(dim=-1)
        Q_fb = compute_td_targets(Qs_fb, self.actor_pessimism)
        Q_disc = compute_td_targets(self.disc_critic(norm_h, z, action_h), self.actor_pessimism).squeeze(-1)
        Q_aux = compute_td_targets(self.aux_critic(norm_h, z, action_h), self.actor_pessimism).squeeze(-1)
        reg_weight = Q_fb.abs().mean().detach()
        value_h = Q_fb + reg_weight * (self.discriminator_reg_coef * Q_disc + self.aux_reg_coef * Q_aux)
        ret = (self.gamma**horizon) * rollout.survival[horizon] * value_h
        terms = {"fb_value": Q_fb, "disc_value": Q_disc, "aux_value": Q_aux}
        if horizon == 0:
            return ret, terms

        B_next = self.backward_map(norm[1:].reshape(-1)).float().view(horizon, n, -1)
        r_fb = (B_next * reward_z).sum(dim=-1)
        eps = self.discriminator_reward_eps
        logits = self.discriminator(norm[:horizon].reshape(-1), z.repeat(horizon, 1)).view(horizon, n)
        logits = logits.clamp(eps, 1 - eps)
        r_disc = torch.log(logits) - torch.log(1 - logits)
        r_aux = rollout.rewards * aux_scale
        rewards = r_fb + reg_weight * (self.discriminator_reg_coef * r_disc + self.aux_reg_coef * r_aux)
        discounts = self.gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32)
        ret = ret + (discounts[:, None] * rollout.survival[:horizon] * rewards).sum(dim=0)
        terms.update({"fb_reward": r_fb, "disc_reward": r_disc, "aux_reward": r_aux})
        return ret, terms

    """
    Helpers.
    """

    @torch.no_grad()
    def _implicit_reward_weights(self, cov_obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        """Return ``Cov(B)^-1 z``, with the covariance of ``B`` over ``cov_obs`` (FB's implicit reward weights).

        Args:
            cov_obs: Normalized observations whose ``B`` sets the covariance.
            z: Latents, ``(N, z_dim)``.

        Returns:
            The weights, ``(N, z_dim)``, in float32.
        """
        B = self.backward_map(cov_obs)
        # disable autocast so the covariance solve runs in float32, as FB-CPR's value regularizer does
        with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=False):
            cov = torch.matmul(B.T.float(), B.float()) / B.shape[0]
            # a small ridge keeps the solve defined while B spans fewer than z_dim directions (early training)
            ridge = 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0], device=cov.device)
            weights, info = torch.linalg.solve_ex(cov + ridge, z.float().T)
            if int(info) != 0 or not bool(torch.isfinite(weights).all()):
                bad = {k: int((~torch.isfinite(v)).any(dim=-1).sum()) for k, v in cov_obs.items()}
                bad_b = int((~torch.isfinite(B)).any(dim=-1).sum())
                raise FloatingPointError(
                    f"Cov(B) solve failed (info {int(info)}): non-finite B rows {bad_b}/{B.shape[0]}, non-finite obs"
                    f" rows per group {bad}, cov diag mean {float(cov.diagonal().mean())}"
                )
            return weights.T

    def _normalize(self, raw: TensorDict) -> TensorDict:
        """FB-CPR's observation normalization at its running statistics, differentiable in the observations.

        The eval-mode BatchNorm formula written out, so no module switches mode inside the compiled region.
        """
        out = raw.clone(recurse=False)
        for key, bn in self.obs_normalizer.modules_dict.items():
            if key in out:
                out[key] = (out[key] - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
        return out

    def _as_batch(self, win: SequenceReplayBuffer.Window) -> ReplayBuffer.Batch:
        """One-step replay windows as FB-CPR's transition batch (raw observations)."""
        gammas = (self.gamma * (1.0 - win.terminated[0])).unsqueeze(-1)
        return ReplayBuffer.Batch(
            win.obs[0].clone(), win.obs[1].clone(), win.actions[0], win.rewards[0], gammas, win.extras["z"][0]
        )

    def _empty_rollout(self, starts: TensorDict) -> Rollout:
        """Return a zero-step rollout at ``starts``."""
        B = starts.batch_size[0]
        empty = torch.zeros(0, B, device=self.device)
        obs = starts.select(*self.learner.imagined_obs.groups)
        return Rollout([obs], [], empty, empty, torch.ones(1, B, device=self.device))

    def _reduce_and_clip_wm(self, params: Any) -> None:
        """Average the world model's gradients across ranks and clip their global norm."""
        params = [p for p in params if p.grad is not None]
        if self.is_multi_gpu:
            self.reduce_parameters(self.world_model)
        nn.utils.clip_grad_norm_(params, 1.0)

    """
    Runner interface.
    """

    def train_mode(self) -> None:
        """Set every model to training mode."""
        super().train_mode()
        self.world_model.train()

    def eval_mode(self) -> None:
        """Set every model to evaluation mode."""
        super().eval_mode()
        self.world_model.eval()

    def save(self) -> dict:
        """Return FB-CPR's states plus the world model's."""
        return {**super().save(), **self.learner.state_dict(), "update_step": self.update_step}

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load FB-CPR's states, and the world model's when critics are loaded and the checkpoint has one."""
        resume = super().load(loaded_dict, load_cfg, strict)
        if (load_cfg is None or load_cfg.get("critic")) and "world_model_state_dict" in loaded_dict:
            optimizer = load_cfg is None or bool(load_cfg.get("optimizer"))
            self.learner.load_state_dict(loaded_dict, optimizer=optimizer, strict=strict)
            self.update_step = loaded_dict.get("update_step", 0)
        return resume

    def broadcast_parameters(self) -> None:
        """Broadcast every model, the world model included, from rank 0."""
        super().broadcast_parameters()
        params = [self.world_model.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self.world_model.load_state_dict(params[0])

    @staticmethod
    def _build_storage(obs: TensorDict, env: URLVecEnv, cfg: dict, device: str, inference: bool) -> dict[str, Any]:
        """Build the world-model learner over every model's groups; its sequence replay logs each step's ``z``.

        Args:
            obs: An observation sample.
            env: The env, carrying ``world_model_spec``.
            cfg: The resolved runner cfg; the world-model entries are popped from ``cfg["algorithm"]``.
            device: Training device.
            inference: Keep one row per env, as nothing samples the storage.

        Returns:
            The ``learner`` and ``replay_buffer`` constructor kwargs.
        """
        alg_cfg = cfg["algorithm"]
        max_episode_length = env.max_episode_length
        if isinstance(max_episode_length, torch.Tensor):
            max_episode_length = int(max_episode_length.max().item())
        groups = [g for name, gs in cfg["obs_groups"].items() if name != "expert" for g in gs]
        learner = WorldModelLearner.build(
            obs,
            env,
            alg_cfg.pop("world_model"),
            list(dict.fromkeys(groups)),
            alg_cfg["storage_scale"] * max_episode_length if not inference else 1,
            cfg["storage_device"],
            device,
            obs_dtypes=alg_cfg.get("replay_obs_dtypes"),
            row_extras={"z": alg_cfg["z_dim"]},
            learning_rate=alg_cfg.pop("wm_learning_rate", 2e-4),
            **{k: alg_cfg.pop(k) for k in LEARNER_KEYS if k in alg_cfg},
        )
        return {"learner": learner, "replay_buffer": learner.replay_buffer}
