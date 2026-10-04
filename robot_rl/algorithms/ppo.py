# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from typing import Any

from robot_rl.env import VecEnv
from robot_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from robot_rl.models import EncoderInferencePolicy, MLPModel, SharedMemoryInferencePolicy
from robot_rl.storage import RolloutStorage
from robot_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_learning_rate: float = 1e-2,
        min_learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        adaptive_lr_once_per_iteration: bool = False,
        device: str = "cpu",
        # Optional shared memory module (consumed by both actor and critic as heads)
        memory: nn.Module | None = None,
        # Optional shared observation encoder (its latent is an extra input to actor and critic)
        encoder: nn.Module | None = None,
        encoder_cfg: dict | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Meta-RL parameters
        meta_rl_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent or memory is not None):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies (including shared memory).")
        if symmetry_cfg is not None and encoder is not None:
            raise ValueError("Symmetry augmentation is not supported with a shared observation encoder.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # Meta RL components
        self.meta_rl = meta_rl_cfg is not None
        self.detach_critic_memory = False
        if meta_rl_cfg is not None:
            self.num_episodes_per_trial: int = meta_rl_cfg["num_episodes_per_trial"]
            self.detach_critic_memory = meta_rl_cfg.get("detach_critic_memory", False)
            # Wait to initialize episode counter since we use data shape to get num_envs
            self.ep_counter: torch.Tensor | None = None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        # Shared memory (optional). When set, actor/critic must be MLP heads on top of the memory's latent.
        if memory is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError(
                "Shared memory is not supported with recurrent actor/critic models. "
                "When `meta_rl_cfg.memory` is set, actor and critic must be plain MLP heads."
            )
        self.memory: nn.Module | None = memory.to(self.device) if memory is not None else None
        # Shared observation encoder (optional), trained by the joint PPO loss
        if encoder is not None and memory is not None:
            raise ValueError("A shared encoder cannot be combined with a shared memory module.")
        if encoder is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("A shared encoder requires plain MLP actor/critic models.")
        self.encoder: nn.Module | None = encoder.to(self.device) if encoder is not None else None
        self.encoder_detach_actor = bool((encoder_cfg or {}).get("detach_actor_gradients", False))
        # L2 penalty on the encoder latent, added to the joint PPO loss
        self.encoder_l2_coef = float((encoder_cfg or {}).get("l2_coef", 0.0))

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic`` / ``self.memory`` / ``self.encoder``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._raw_memory = self.memory
        self._raw_encoder = self.encoder

        # Create the optimizer
        params: Any = chain(self.actor.parameters(), self.critic.parameters())
        if self.memory is not None:
            params = chain(params, self.memory.parameters())
        if self.encoder is not None:
            params = chain(params, self.encoder.parameters())
        self.optimizer = resolve_optimizer(optimizer)(params, lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.adaptive_lr_once_per_iteration = adaptive_lr_once_per_iteration
        self.learning_rate = learning_rate
        # Bounds for the adaptive LR schedule (both the per-minibatch and once-per-iteration paths).
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def _encoder_args(self, obs: TensorDict, detach: bool = False) -> tuple[torch.Tensor, ...]:
        """Return the encoder latent as an extra model input tuple; empty when no encoder is configured."""
        if self.encoder is None:
            return ()
        latent = self.encoder(obs)
        return (latent.detach(),) if detach else (latent,)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Pre-step batch info so the RNNModel can lazy-init its hidden states; without it the first
        # rollout's step-0 snapshot is None and PPO update() crashes on a hidden-size mismatch (MLP/TXL ignore).
        batch_size = obs.batch_size[0]
        device = obs.device
        if self.memory is not None:
            try:
                self.transition.memory_hidden_state = self.memory.get_hidden_state(batch_size=batch_size, device=device)
            except TypeError:
                self.transition.memory_hidden_state = self.memory.get_hidden_state()
            self.transition.hidden_states = (None, None)
            latent = self.memory(obs).detach()
            self.transition.actions = self.actor.forward_from_latent(latent, stochastic_output=True).detach()
            # Include additional critic obs for asymmetric actor-critic
            self.transition.values = self.critic.forward_from_latent(latent, obs=obs).detach()
        else:
            try:
                actor_hs = self.actor.get_hidden_state(batch_size=batch_size, device=device)
                critic_hs = self.critic.get_hidden_state(batch_size=batch_size, device=device)
            except TypeError:
                actor_hs = self.actor.get_hidden_state()
                critic_hs = self.critic.get_hidden_state()
            self.transition.hidden_states = (actor_hs, critic_hs)
            enc_args = self._encoder_args(obs, detach=True)
            self.transition.actions = self.actor(obs, *enc_args, stochastic_output=True).detach()
            self.transition.values = self.critic(obs, *enc_args).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        if self.memory is not None:
            self.memory.update_normalization(obs)
        if self.encoder is not None:
            self.encoder.update_normalization(obs)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Record trial done status for meta-RL
        if self.meta_rl:
            # Initialize episode counter if necessary
            if self.ep_counter is None:
                self.ep_counter = torch.zeros(dones.shape[0], dtype=torch.long, device=self.device)
            # Increment episode counter
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.ep_counter[new_ids] += 1
            # Compute whether trial is finished
            trial_dones = (dones.bool() & (self.ep_counter % self.num_episodes_per_trial == 0)).byte()
            self.transition.meta_dones = trial_dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()

        # For meta-RL environments, we only want to reset hidden states at end of trial
        # Otherwise we reset at end of episode
        do_reset = trial_dones if self.meta_rl else dones
        if self.memory is not None:
            self.memory.reset(do_reset)
        self.actor.reset(do_reset)
        self.critic.reset(do_reset)
        return trial_dones.nonzero(as_tuple=False).squeeze(1) if self.meta_rl else None

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        if self.memory is not None:
            # Save the shared memory's hidden state before the extra forward pass
            memory_hidden_state = self.memory.get_hidden_state()
            latent = self.memory(obs).detach()
            last_values = self.critic.forward_from_latent(latent, obs=obs).detach()
            # Restore the memory's hidden state so the next rollout is not affected by the forward pass
            self.memory.reset(hidden_state=memory_hidden_state)
        else:
            critic_hidden_state = self.critic.get_hidden_state()
            last_values = self.critic(obs, *self._encoder_args(obs, detach=True)).detach()
            # Restore the critic's hidden state so the next rollout is not affected by the forward pass
            self.critic.reset(hidden_state=critic_hidden_state)
        # GAE runs over storage tensors; bring the bootstrap value to the storage device.
        last_values = last_values.to(st.device)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_log_prob = 0
        mean_clip_fraction = 0.0
        sum_kl = 0.0
        max_kl = 0.0
        kl_first_minibatch: float | None = None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Encoder L2 loss
        mean_encoder_l2_loss = 0 if (self.encoder is not None and self.encoder_l2_coef > 0.0) else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent or self.memory is not None:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, device=self.device
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                device=self.device,
            )

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            if self.memory is not None:
                # Run shared memory once per mini-batch; both heads consume the same unpadded latent.
                latent = self.memory(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.memory_hidden_state,
                )
                self.actor.forward_from_latent(latent, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                # Optionally stop the value loss from backpropagating into the shared memory
                critic_latent = latent.detach() if self.detach_critic_memory else latent
                values = self.critic.forward_from_latent(critic_latent, obs=batch.observations, masks=batch.masks)
            else:
                # Recompute the encoder latent with gradients; optionally stop the actor loss from training it
                enc_args = self._encoder_args(batch.observations)
                actor_enc_args = tuple(a.detach() for a in enc_args) if self.encoder_detach_actor else enc_args
                self.actor(
                    batch.observations,
                    *actor_enc_args,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(
                    batch.observations, *enc_args, masks=batch.masks, hidden_state=batch.hidden_states[1]
                )
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence (always, for logging) and adapt the learning rate if scheduled
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                kl_mean = torch.mean(kl)

                # Reduce the KL divergence across all GPUs
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

                kl_value = kl_mean.item()
                if kl_first_minibatch is None:
                    kl_first_minibatch = kl_value
                sum_kl += kl_value
                if kl_value > max_kl:
                    max_kl = kl_value

                # Per-minibatch adaptive LR
                if (
                    self.desired_kl is not None
                    and self.schedule == "adaptive"
                    and not self.adaptive_lr_once_per_iteration
                ):
                    if kl_value > self.desired_kl * 2.0:
                        self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                    elif self.desired_kl / 2.0 > kl_value > 0.0:
                        self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # Fraction of samples whose importance ratio fell outside the clip band (PPO trust-region diagnostic).
            clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # grad-carrying ``enc_args``, not ``actor_enc_args``, so the penalty survives ``detach_actor_gradients``
            if self.encoder is not None and self.encoder_l2_coef > 0.0:
                encoder_l2_loss = enc_args[0].pow(2).mean()
                loss = loss + self.encoder_l2_coef * encoder_l2_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if self.memory is not None:
                nn.utils.clip_grad_norm_(self.memory.parameters(), self.max_grad_norm)
            if self.encoder is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_log_prob += actions_log_prob.mean().item()
            mean_clip_fraction += clip_fraction.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # Encoder L2 loss
            if mean_encoder_l2_loss is not None:
                mean_encoder_l2_loss += encoder_l2_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches

        # Adapt the LR once per iteration
        if self.desired_kl is not None and self.schedule == "adaptive" and self.adaptive_lr_once_per_iteration:
            mean_kl_iter = sum_kl / num_updates
            if self.gpu_global_rank == 0:
                if mean_kl_iter > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < mean_kl_iter < self.desired_kl / 2.0:
                    self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_log_prob /= num_updates
        mean_clip_fraction /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_encoder_l2_loss is not None:
            mean_encoder_l2_loss /= num_updates

        # Construct the loss dictionary
        # Slash-prefixed keys are logged under that scalar group as-is (Train/...); bare keys go under Loss/.
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "Train/log_prob": mean_log_prob,
            "Train/clip_fraction": mean_clip_fraction,
        }
        loss_dict["Train/kl_mean"] = sum_kl / num_updates
        loss_dict["Train/kl_max"] = max_kl
        if kl_first_minibatch is not None:
            loss_dict["Train/kl_first_minibatch"] = kl_first_minibatch
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_encoder_l2_loss is not None:
            loss_dict["encoder_l2"] = mean_encoder_l2_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.memory is not None:
            self.memory.train()
        if self.encoder is not None:
            self.encoder.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.memory is not None:
            self.memory.eval()
        if self.encoder is not None:
            self.encoder.eval()
        if self.rnd:
            self.rnd.eval()

    def eval(
        self, env: VecEnv, max_steps: int = 200, stochastic: bool = False, action_repeat: int = 1
    ) -> list[dict[str, torch.Tensor]]:
        """Run an evaluation rollout for ``max_steps`` environment steps.

        The caller is responsible for any env-side video wrapping: each ``env.step()`` here produces a rendered
        frame for any active video-recording wrapper around the env.

        Args:
            env: Vectorized environment to roll out in.
            max_steps: Number of environment steps to run.
            stochastic: When ``False`` (default), act with the actor's deterministic mean. When ``True``,
                sample the action exactly as during training (same exploration noise). The stochastic rollout
                reflects what training actually experiences, which can differ markedly from the deterministic
                mean (e.g. a hierarchical policy whose mean is a trivial fixed point).
            action_repeat: Hold each queried action for this many ``env.step`` calls, re-querying the actor --
                and advancing the recurrent memory -- only every ``action_repeat`` steps. For a hierarchical
                task recorded at the low-level rate (env ``decimation`` lowered to the low-level value for
                smooth frames), set this to the high-level decimation so the high-level control rate matches
                training instead of running ``action_repeat``x too fast. Default 1 = re-query every step.

        Returns:
            A list of per-batch info dicts; empty, as this rollout collects none.
        """
        was_training = self.actor.training
        self.eval_mode()
        if hasattr(env, "eval_mode"):
            env.eval_mode()

        obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()[0]
        # Reset any recurrent / TXL state on actor + memory so video starts from a clean context.
        if hasattr(self.actor, "reset"):
            self.actor.reset()
        if self.memory is not None and hasattr(self.memory, "reset"):
            self.memory.reset()

        action_repeat = max(1, action_repeat)

        def query_actions() -> torch.Tensor:
            # query the actor (advancing the recurrent memory, if any) for one high-level decision
            if self.memory is not None:
                return self.actor.forward_from_latent(self.memory(obs), stochastic_output=stochastic)
            return self.actor(obs, *self._encoder_args(obs, detach=True), stochastic_output=stochastic)

        with torch.inference_mode():
            actions = query_actions()  # initial decision (step 0)
            for step in range(max_steps):
                # re-query only once per high-level decision; step 0 already used the initial query above
                if step > 0 and step % action_repeat == 0:
                    actions = query_actions()
                obs, _, _, _ = env.step(actions)

        if was_training:
            self.train_mode()
            if hasattr(env, "train_mode"):
                env.train_mode()
        return []

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """State-dict keys sufficient to run/eval/export the policy (all other keys are resume-only).

        A checkpoint keeping only these can be played, rendered, and exported, but NOT resumed for
        training (critic/optimizer/RND absent). The memory/encoder keys only exist on those runs.
        """
        return ("actor_state_dict", "memory_state_dict", "encoder_state_dict")

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.memory is not None:
            saved_dict["memory_state_dict"] = self._raw_memory.state_dict()
        if self._raw_encoder is not None:
            saved_dict["encoder_state_dict"] = self._raw_encoder.state_dict()
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "memory": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        # critic/optimizer are absent from a policy-only (slim) checkpoint -- skip rather than KeyError
        if load_cfg.get("critic") and "critic_state_dict" in loaded_dict:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("memory") and self.memory is not None and "memory_state_dict" in loaded_dict:
            self._raw_memory.load_state_dict(loaded_dict["memory_state_dict"], strict=strict)
        # not gated on load_cfg: the encoder is part of the actor's input, so inference-only loads need it too
        if self._raw_encoder is not None:
            if "encoder_state_dict" in loaded_dict:
                self._raw_encoder.load_state_dict(loaded_dict["encoder_state_dict"], strict=strict)
            elif strict:
                raise KeyError("Checkpoint has no encoder_state_dict for this encoder policy.")
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> nn.Module:
        """Get the policy model.

        Wraps the actor with the shared memory module when configured, since the actor is then a head over a
        precomputed latent and cannot consume raw obs. Likewise wraps it with the shared encoder, since the
        actor then expects the encoder latent as an extra input that raw obs alone cannot supply.
        """
        if self.memory is not None:
            return SharedMemoryInferencePolicy(self._raw_memory, self._raw_actor)  # type: ignore
        if self._raw_encoder is not None:
            return EncoderInferencePolicy(self._raw_encoder, self._raw_actor)
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and the shared memory/encoder modules (if any) with ``torch.compile``.

        See :func:`~robot_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        if self._raw_memory is not None:
            self.memory = compile_model(self._raw_memory, mode)  # type: ignore
        if self._raw_encoder is not None:
            self.encoder = compile_model(self._raw_encoder, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Optional shared memory config
        meta_rl_cfg = cfg["algorithm"].get("meta_rl_cfg")
        shared_memory_cfg: dict | None = meta_rl_cfg.get("memory") if isinstance(meta_rl_cfg, dict) else None
        # ``.get``, not ``.pop``: ``__init__`` also receives it through the ``**cfg["algorithm"]`` splat below
        encoder_cfg: dict | None = cfg["algorithm"].get("encoder_cfg")

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if encoder_cfg is not None:
            default_sets.append("encoder")
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Build the optional shared memory module first so we can size actor/critic heads from its latent_dim
        memory: nn.Module | None = None
        head_kwargs: dict = {}
        critic_head_kwargs: dict = {}
        if shared_memory_cfg is not None:
            mem_class: type[MLPModel] = resolve_callable(shared_memory_cfg.pop("class_name"))  # type: ignore
            memory = mem_class(obs, cfg["obs_groups"], "actor", 1, memory_only=True, **shared_memory_cfg).to(device)
            print(f"Shared Memory Model: {memory}")
            head_kwargs["input_dim_override"] = memory.latent_dim  # type: ignore[attr-defined]
            # Critic consumes privileged obs + memory latent
            critic_head_kwargs["input_dim_override"] = memory.latent_dim  # type: ignore[attr-defined]
            critic_head_kwargs["append_obs_groups"] = True

        # Build the optional shared encoder; its latent is appended to both heads' inputs
        encoder: nn.Module | None = None
        if encoder_cfg is not None:
            encoder_model_cfg = dict(encoder_cfg["model"])
            encoder_class: type[MLPModel] = resolve_callable(encoder_model_cfg.pop("class_name"))  # type: ignore
            encoder_dim = int(encoder_cfg["output_dim"])
            encoder = encoder_class(obs, cfg["obs_groups"], "encoder", encoder_dim, **encoder_model_cfg).to(device)
            print(f"Encoder Model: {encoder}")
            head_kwargs["other_input_dims"] = (encoder_dim,)
            critic_head_kwargs["other_input_dims"] = (encoder_dim,)

        # Initialize the policy
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **head_kwargs, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_head_kwargs, **cfg["critic"]).to(
            device
        )
        print(f"Critic Model: {critic}")

        # Initialize the storage. Use "meta_rl" when meta-RL is configured so the trajectory generator splits
        # at trial boundaries (where memory was reset) rather than episode boundaries.
        training_type = "meta_rl" if cfg["algorithm"].get("meta_rl_cfg") is not None else "rl"
        storage_device = cfg.get("storage_device")
        if storage_device is None:
            storage_device = device
        storage = RolloutStorage(
            training_type, env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], storage_device
        )

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            memory=memory,
            encoder=encoder,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.memory is not None:
            model_params.append(self._raw_memory.state_dict())
        if self._raw_encoder is not None:
            model_params.append(self._raw_encoder.state_dict())
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        idx = 2
        if self.memory is not None:
            self._raw_memory.load_state_dict(model_params[idx])
            idx += 1
        if self._raw_encoder is not None:
            self._raw_encoder.load_state_dict(model_params[idx])
            idx += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.memory is not None:
            all_params = chain(all_params, self.memory.parameters())
        if self.encoder is not None:
            all_params = chain(all_params, self.encoder.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
