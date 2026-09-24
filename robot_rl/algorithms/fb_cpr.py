from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.env import URLVecEnv
from robot_rl.models import FuseModel, MLPModel
from robot_rl.modules import DictModule, ExponentialMovingAverageNormalization, TargetNetwork
from robot_rl.storage import ReplayBuffer, TrajectoryBuffer, ZBuffer
from robot_rl.utils import (
    compute_emd,
    compute_td_targets,
    eval_mode,
    forward_sliding_mean,
    pad_to_size,
    pad_to_size_repeat,
    resolve_callable,
    resolve_dtype,
    resolve_obs_groups,
    resolve_optimizer,
)


def _up_axis(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Express the world z axis in the body frame of (w, x, y, z) quaternions: roll and pitch, blind to heading."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1)


class FbCpr:
    """Forward-Backward representations with Conditional Policy Regularization (FB-CPR) algorithm.

    Reference:
        - Tirinzoni et al. "Zero-shot whole-body humanoid control via behavioral foundation models." arXiv preprint
          arXiv:2504.11054 (2025).
    """

    actor: FuseModel
    """The actor model."""

    forward_map: FuseModel
    """The forward model."""

    backward_map: MLPModel
    """The backward model."""

    disc_critic: FuseModel
    """The discriminator critic model."""

    aux_critic: FuseModel
    """The auxiliary critic model."""

    discriminator: MLPModel
    """The discriminator model"""

    def __init__(
        self,
        actor: FuseModel,
        forward_map: FuseModel,
        backward_map: MLPModel,
        disc_critic: FuseModel,
        aux_critic: FuseModel,
        discriminator: MLPModel,
        obs_normalizer: DictModule[nn.BatchNorm1d],
        replay_buffer: ReplayBuffer,
        expert_buffer: TrajectoryBuffer | None,
        z_buffer: ZBuffer,
        z_dim: int,
        motion_path: str,
        expert_sequence_length: int,
        steps_per_z_update: int,
        actor_learning_rate: float = 1e-4,
        forward_learning_rate: float = 1e-4,
        backward_learning_rate: float = 1e-4,
        discriminator_learning_rate: float = 1e-4,
        disc_critic_learning_rate: float = 1e-4,
        aux_critic_learning_rate: float = 1e-4,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        clip_actor_std: float = 0.2,
        gamma: float = 0.99,
        train_goal_ratio: float = 0.5,
        expert_asm_ratio: float = 0.0,
        expert_rollout_ratio: float = 0.5,
        expert_rollout_length: int = 250,
        z_relabel_ratio: float = 0.8,
        forward_backward_pessimism: float = 0.0,
        actor_pessimism: float = 0.5,
        disc_critic_pessimism: float = 0.5,
        aux_critic_pessimism: float = 0.5,
        discriminator_reg_coef: float = 1.0,
        aux_reg_coef: float = 1.0,
        value_loss_coef: float = 1.0,
        ortho_loss_coef: float = 1.0,
        grad_loss_coef: float = 1.0,
        fb_tau: float = 0.01,
        critic_tau: float = 0.005,
        batch_size: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
        compile_mode: str | None = "reduce-overhead",
        clip_actions: float | None = None,
        num_seed_steps_per_env: int = 0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # FB-CPR components
        self.actor = actor.to(self.device)
        self.forward_map = forward_map.to(self.device)
        self.backward_map = backward_map.to(self.device)
        self.disc_critic = disc_critic.to(self.device)
        self.aux_critic = aux_critic.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.obs_normalizer = obs_normalizer.to(self.device)

        # Initialize model weights
        for model in self.models:
            model.init_weights()

        # Instantiate target (Polyak) networks
        self.target_forward_map = TargetNetwork(self.forward_map, tau=fb_tau).to(self.device)
        self.target_backward_map = TargetNetwork(self.backward_map, tau=fb_tau).to(self.device)
        self.target_disc_critic = TargetNetwork(self.disc_critic, tau=critic_tau).to(self.device)
        self.target_aux_critic = TargetNetwork(self.aux_critic, tau=critic_tau).to(self.device)

        # Create the optimizers. Adam/AdamW support a fused CUDA kernel that collapses the per-parameter
        # _foreach_add_/_foreach_mul_ ops into a single launch.
        optimizer_cls = resolve_optimizer(optimizer)
        optimizer_kwargs: dict[str, Any] = {"weight_decay": weight_decay}
        if optimizer.lower() in ("adam", "adamw"):
            optimizer_kwargs["fused"] = True
        self.actor_optimizer = optimizer_cls(self.actor.parameters(), lr=actor_learning_rate, **optimizer_kwargs)
        self.forward_optimizer = optimizer_cls(
            self.forward_map.parameters(), lr=forward_learning_rate, **optimizer_kwargs
        )
        self.backward_optimizer = optimizer_cls(
            self.backward_map.parameters(), lr=backward_learning_rate, **optimizer_kwargs
        )
        self.disc_critic_optimizer = optimizer_cls(
            self.disc_critic.parameters(), lr=disc_critic_learning_rate, **optimizer_kwargs
        )
        self.aux_critic_optimizer = optimizer_cls(
            self.aux_critic.parameters(), lr=aux_critic_learning_rate, **optimizer_kwargs
        )
        self.discriminator_optimizer = optimizer_cls(
            self.discriminator.parameters(), lr=discriminator_learning_rate, **optimizer_kwargs
        )

        # Add storage
        self.replay_buffer = replay_buffer
        self.expert_buffer = expert_buffer
        self.z_buffer = z_buffer
        self.transition = ReplayBuffer.Transition()

        # Add normalization for auxiliary rewards
        self.aux_reward_normalizer = ExponentialMovingAverageNormalization(scale=True).to(self.device)

        # State for expert rollout context
        self.expert_rollout_envs: torch.Tensor | None = None
        self.expert_rollout_z: torch.Tensor | None = None

        # Rollout state: latent z, last dones, per-env episode step, and act count for the seed phase.
        # Episode lengths are lazily allocated on the first act (num_envs known then).
        self.clip_actions = clip_actions
        self.num_seed_steps_per_env = num_seed_steps_per_env
        self._rollout_z: torch.Tensor | None = None
        self._cur_episode_length: torch.Tensor | None = None
        self._act_steps = 0

        # FB-CPR parameters
        self.dtype = resolve_dtype(dtype)
        self.discriminator_reward_eps = torch.finfo(self.dtype).resolution
        self.motion_path = os.path.abspath(motion_path)
        self.expert_sequence_length = expert_sequence_length
        self.steps_per_z_update = steps_per_z_update
        self.train_goal_ratio = train_goal_ratio
        self.expert_asm_ratio = expert_asm_ratio
        self.expert_rollout_ratio = expert_rollout_ratio
        self.expert_rollout_length = expert_rollout_length
        self.z_relabel_ratio = z_relabel_ratio
        self.forward_backward_pessimism = forward_backward_pessimism
        self.actor_pessimism = actor_pessimism
        self.disc_critic_pessimism = disc_critic_pessimism
        self.aux_critic_pessimism = aux_critic_pessimism
        self.discriminator_reg_coef = discriminator_reg_coef
        self.aux_reg_coef = aux_reg_coef
        self.value_loss_coef = value_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.grad_loss_coef = grad_loss_coef
        self.fb_tau = fb_tau
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.clip_actor_std = clip_actor_std
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.z_dim = z_dim

        # Precompute useful variables
        self._off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self._off_diag_sum = self._off_diag.sum()

        # Apply torch.compile to hot-path methods
        if compile_mode is not None:
            # _sample_mixed_z skipped: upstream Inductor joint_graph pattern-matcher bug
            # TODO: once bug is fixed, re-enable this
            self._encode_expert = torch.compile(self._encode_expert, mode=compile_mode, fullgraph=True)
            self._update_discriminator = torch.compile(self._update_discriminator, mode=compile_mode)
            self._update_forward_backward = torch.compile(self._update_forward_backward, mode=compile_mode)
            self._update_disc_critic = torch.compile(self._update_disc_critic, mode=compile_mode)
            self._update_aux_critic = torch.compile(self._update_aux_critic, mode=compile_mode)
            self._update_actor = torch.compile(self._update_actor, mode=compile_mode)

    @property
    def models(self) -> list[MLPModel]:
        """Return a list of the algorithm's trainable models."""
        return [
            self.actor,
            self.forward_map,
            self.backward_map,
            self.disc_critic,
            self.aux_critic,
            self.discriminator,
        ]

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data.

        Owns the per-env rollout state: the latent ``z`` (refreshed from episode progress), previous dones,
        seed-phase random sampling, and action clipping.
        """
        num_envs = obs.batch_size[0]
        if self._cur_episode_length is None:
            self._cur_episode_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Update latent z from episode progress, then sample
        z = self._rollout_z = self.update_rollout_z(self._rollout_z, self._cur_episode_length, num_envs)
        random_sample = self._act_steps <= self.num_seed_steps_per_env
        self._act_steps += 1
        # Normalize observations
        with eval_mode(self.obs_normalizer):
            norm_obs = self.obs_normalizer(obs)
        # compute the actions and values
        self.transition.actions = self.actor(norm_obs, z, stochastic_output=True).detach()
        # uniformly sample from action space during the seed phase
        if random_sample:
            clip_actions = 1.0 if self.clip_actions is None else self.clip_actions
            self.transition.actions.uniform_(-clip_actions, clip_actions).detach()
        # record obs before env.step(); dones is attached in process_env_step, once it is known
        self.transition.observations = obs
        self.transition.context = z
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step and store transition data."""
        # Record the rewards
        self.transition.rewards = rewards
        # Record the and next obs and next terminated (after env.step)
        # Terminated is all dones that are not time_outs
        self.transition.next_terminated = (dones * ~extras["time_outs"]).byte()
        self.transition.next_observations = obs
        # the drop filter needs the dones of the step just taken (they mark THIS transition's next_obs as post-reset)
        self.transition.dones = dones

        # Record the transition
        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()

        # Reset hidden states of all models
        for model in self.models:
            model.reset(dones)

        # Advance the rollout state consumed by the next act(): episode step counters
        if self._cur_episode_length is not None:
            self._cur_episode_length += 1
            self._cur_episode_length[(dones > 0).nonzero(as_tuple=False)] = 0

    def reset_rollout_state(self) -> None:
        """Reset the per-env rollout bookkeeping after an external env reset (e.g. post-eval); z is kept."""
        if self._cur_episode_length is not None:
            self._cur_episode_length[:] = 0

    def compute_gammas(self) -> None:
        """Compute gamma values from stored transitions."""
        st = self.replay_buffer
        # Write in place (not reassign): this runs under inference_mode, so reassigning would make
        # st.gammas an inference tensor, which CUDA-graph capture (reduce-overhead) forbids on a GPU buffer.
        st.gammas.copy_(self.gamma * (1 - st.next_terminated).float())

    def update_rollout_z(self, z: torch.Tensor | None, cur_episode_length: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Refresh the per-environment latent ``z`` used for rollout based on episode progress."""
        if not torch.all(cur_episode_length == cur_episode_length[0]):
            raise ValueError("Expected episode lengths to be uniform.")
        step = int(cur_episode_length[0].item())
        # Update from z buffer
        # the z buffer is empty until the first update, e.g. when an eval resets rollouts during the seed phase
        if z is None or (step % self.steps_per_z_update == 0 and len(self.z_buffer) == 0):
            z = self._sample_random_z(num_envs)
        elif step % self.steps_per_z_update == 0:
            z = self.z_buffer.sample(num_envs, device=self.device)

        # Update from expert buffer
        rollout_idx = step % self.expert_rollout_length
        if rollout_idx == 0 or self.expert_rollout_envs is None or self.expert_rollout_z is None:
            # Use constant num_expert_updates to avoid torch recompile
            num_expert_updates = int(num_envs * self.expert_rollout_ratio)
            self.expert_rollout_envs = torch.randperm(num_envs, device=self.device)[:num_expert_updates]

            _, expert_next_obs = self.expert_buffer.sample(
                num_expert_updates * self.expert_rollout_length,
                device=self.device,
                seq_length=self.expert_rollout_length,
            )
            with eval_mode(self.obs_normalizer):
                expert_next_obs = self.obs_normalizer(expert_next_obs)
            expert_z = self.backward_map(expert_next_obs).view(num_expert_updates, self.expert_rollout_length, -1)
            expert_z = forward_sliding_mean(expert_z, self.expert_sequence_length, dim=1)
            self.expert_rollout_z = self.project_z(expert_z)
        z[self.expert_rollout_envs] = self.expert_rollout_z[:, rollout_idx]

        return z

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Run optimization epochs over stored batches and return mean losses."""
        batch = self.replay_buffer.sample_mini_batch(self.device)
        expert_obs, expert_next_obs = self.expert_buffer.sample(
            self.batch_size, self.device, seq_length=self.expert_sequence_length
        )

        # Update normalizer running statistics from training data
        self.obs_normalizer(batch.observations)
        self.obs_normalizer(batch.next_observations)

        # Normalize all observations using running statistics
        with torch.no_grad(), eval_mode(self.obs_normalizer):
            batch.observations = self.obs_normalizer(batch.observations)
            batch.next_observations = self.obs_normalizer(batch.next_observations)
            expert_obs = self.obs_normalizer(expert_obs)
            expert_next_obs = self.obs_normalizer(expert_next_obs)

        torch.compiler.cudagraph_mark_step_begin()

        # Encode expert z
        expert_z = self._encode_expert(expert_next_obs)

        # Update discriminator
        disc_loss_dict, disc_extras = self._update_discriminator(batch, expert_obs, expert_z)

        # Sample and store mixed z
        z = self._sample_mixed_z(batch.next_observations, expert_z)
        self.z_buffer.add(z)
        # Replace some train zs with sampled zs
        relabel_mask = torch.rand(self.batch_size, 1, device=self.device) < self.z_relabel_ratio
        batch.context = torch.where(relabel_mask, z, batch.context)
        # Normalize aux rewards (TODO: move to reward manager?)
        batch.rewards = self.aux_reward_normalizer(batch.rewards)

        # Update other models
        fb_loss_dict, fb_extras = self._update_forward_backward(batch)
        disc_critic_loss_dict, disc_critic_extras = self._update_disc_critic(batch)
        aux_critic_loss_dict, aux_critic_extras = self._update_aux_critic(batch)
        actor_loss_dict, actor_extras = self._update_actor(batch)

        # Prepare logging dicts
        loss_dict = {}
        loss_dict.update(disc_loss_dict)
        loss_dict.update(fb_loss_dict)
        loss_dict.update(disc_critic_loss_dict)
        loss_dict.update(aux_critic_loss_dict)
        loss_dict.update(actor_loss_dict)

        extras = {}
        extras.update(disc_extras)
        extras.update(fb_extras)
        extras.update(disc_critic_extras)
        extras.update(aux_critic_extras)
        extras.update(actor_extras)

        with torch.no_grad():
            self._soft_update_targets()
            # clone to avoid issues with accessing memory outside cudagraphs when logging
            loss_dict = {k: v.clone() for k, v in loss_dict.items()}
            extras = {k: v.clone() for k, v in extras.items()}

        return loss_dict, extras

    def eval(self, env: URLVecEnv, max_steps: int | None = None, **kwargs: Any) -> list[dict[str, torch.Tensor]]:
        """Track every expert motion and re-weight the expert buffer by how badly each one was tracked.

        Args:
            env: Vectorized environment to replay the expert motions in.
            max_steps: Stop after this many ``env.step`` calls (a bounded video clip); priorities are then left
                untouched. ``None`` scores every motion.
            **kwargs: Accepted and ignored, so callers can use one eval signature across algorithms.

        Returns:
            One info dict holding the per-motion ``emd`` and ``joint_error``.
        """
        result = self.eval_motions(env, max_steps=max_steps)
        if max_steps is None:
            self.apply_eval_outputs({"motion_priorities": self.motion_priorities(result["emd"])})
        return [{"emd": result["emd"].cpu(), "joint_error": result["joint_error"].cpu()}]

    def eval_motions(self, env: URLVecEnv, max_steps: int | None = None) -> dict[str, torch.Tensor]:
        """Track every expert motion from its first frame and score it; the expert buffer is not modified.

        Each motion is driven by ``z_t = B(obs_{t+1})`` with no termination guard, in a random env assignment
        (per-env domain randomization is fixed), and results are returned in expert-buffer order.

        Args:
            env: Vectorized environment to replay the expert motions in.
            max_steps: Stop after this many ``env.step`` calls; unscored motions keep NaN.

        Returns:
            ``emd`` and ``joint_error`` of shape ``(num_motions,)``; per-step ``root_error`` (root displacement
            error from the clip's start [m]) and ``tilt_error`` (angle between the root's and the reference's
            gravity directions in their own frames [rad], blind to heading), each ``(num_motions, bucket_size - 1)``.
        """
        print("[INFO] Evaluating motions...")
        self.eval_mode()
        env.eval_mode()

        buffer = self.expert_buffer
        rollout_steps = buffer.bucket_size - 1
        emd = torch.full((buffer.num_motions,), float("nan"), device=self.device)
        joint_error = torch.full_like(emd, float("nan"))
        root_error = torch.full((buffer.num_motions, rollout_steps), float("nan"), device=self.device)
        tilt_error = torch.full_like(root_error, float("nan"))
        start = 0
        steps_done = 0
        for eval_obs in buffer.get_batch_motions(env.num_envs, device=self.device):
            batch = eval_obs.shape[0]
            ids = buffer.eval_order[start : start + batch]
            start += batch
            ref = buffer.get_expert_state(eval_obs)
            norm_eval_obs = self.obs_normalizer(eval_obs.view(-1))
            # z at rollout step t encodes the next desired state (frame t+1), matching how the actor is
            # trained on (obs_t, z=encode(next_obs)) pairs.
            eval_zs = self.backward_map(norm_eval_obs).view(batch, buffer.bucket_size, -1)[:, 1:, :]
            # spare envs replay the batch's first motion: a zero root quaternion is not a legal pose and
            # leaves NaN physics behind in an env that goes on training
            eval_zs = pad_to_size_repeat(eval_zs, env.num_envs, dim=0)
            first = {k: pad_to_size_repeat(v[:, 0, :], env.num_envs, dim=0) for k, v in ref.items()}
            obs, _ = env.reset_to({"articulation": {"robot": first}}, is_relative=True)
            qpos = torch.zeros((batch, rollout_steps, first["joint_position"].shape[1]), device=self.device)
            root = torch.zeros((batch, rollout_steps, 3), device=self.device)
            up = torch.zeros((batch, rollout_steps, 3), device=self.device)
            root_start = buffer.get_expert_state(obs)["root_pose"][:batch, :3].to(self.device)
            steps = rollout_steps
            for t in range(rollout_steps):
                actions = pad_to_size(self.actor(self.obs_normalizer(obs), eval_zs[:, t, :]), env.num_envs, dim=0)
                obs, _, _, _ = env.step(actions.to(env.device))
                state = buffer.get_expert_state(obs)
                qpos[:, t] = state["joint_position"][:batch].to(self.device)
                root[:, t] = state["root_pose"][:batch, :3].to(self.device)
                up[:, t] = _up_axis(state["root_pose"][:batch, 3:7].to(self.device))
                steps_done += 1
                if max_steps is not None and steps_done >= max_steps:
                    steps = t + 1
                    break
            # actual step t is the pose after targeting frame t+1
            ref_qpos = ref["joint_position"][:, 1 : steps + 1]
            ref_root = ref["root_pose"][:, : steps + 1, :3]
            act_disp = root[:, :steps] - root_start[:, None]
            ref_disp = ref_root[:, 1:] - ref_root[:, :1]
            root_error[ids, :steps] = (act_disp - ref_disp).norm(dim=-1)
            ref_up = _up_axis(ref["root_pose"][:, 1 : steps + 1, 3:7])
            cos = (up[:, :steps] * ref_up).sum(dim=-1).clamp(-1.0, 1.0)
            tilt_error[ids, :steps] = torch.acos(cos)
            joint_error[ids] = (qpos[:, :steps] - ref_qpos).norm(dim=-1).mean(dim=-1)
            emd[ids] = torch.stack([compute_emd(qpos[i, :steps], ref_qpos[i]) for i in range(batch)])
            if max_steps is not None and steps_done >= max_steps:
                break

        self.train_mode()
        env.train_mode()
        print("[INFO] Finished evaluating motions.")
        return {"emd": emd, "joint_error": joint_error, "root_error": root_error, "tilt_error": tilt_error}

    @staticmethod
    def motion_priorities(emd: torch.Tensor) -> torch.Tensor:
        """Expert sampling weight per motion: ``2^(2 * clamp(emd, 0.5, 2))``, so poorly tracked motions recur."""
        return torch.pow(2, emd.clamp(min=0.5, max=2.0) * 2)

    def apply_eval_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        """Consume an eval's outputs; ``motion_priorities`` (buffer order) replaces the expert sampling weights."""
        if "motion_priorities" in outputs:
            self.expert_buffer.set_priorities(outputs["motion_priorities"])

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        for model in self.models:
            model.train()
        self.obs_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        for model in self.models:
            model.eval()
        self.obs_normalizer.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "forward_map_state_dict": self.forward_map.state_dict(),
            "backward_map_state_dict": self.backward_map.state_dict(),
            "disc_critic_state_dict": self.disc_critic.state_dict(),
            "aux_critic_state_dict": self.aux_critic.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "obs_normalizer_state_dict": self.obs_normalizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "forward_optimizer_state_dict": self.forward_optimizer.state_dict(),
            "backward_optimizer_state_dict": self.backward_optimizer.state_dict(),
            "disc_critic_optimizer_state_dict": self.disc_critic_optimizer.state_dict(),
            "aux_critic_optimizer_state_dict": self.aux_critic_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "target_forward_map_state_dict": self.target_forward_map.target.state_dict(),
            "target_backward_map_state_dict": self.target_backward_map.target.state_dict(),
            "target_disc_critic_state_dict": self.target_disc_critic.target.state_dict(),
            "target_aux_critic_state_dict": self.target_aux_critic.target.state_dict(),
            "z_buffer_state": self.z_buffer.state_dict(),
            "expert_buffer_state": self.expert_buffer.state_dict(),
        }
        return saved_dict

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """State-dict keys sufficient to run/eval/export the policy (all other keys are resume-only).

        A checkpoint keeping only these can be played, video-rendered, and exported, but NOT resumed
        for training (critics/optimizers/buffers are absent).
        """
        return ("actor_state_dict", "backward_map_state_dict", "obs_normalizer_state_dict")

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict.

        Missing keys are skipped, so a policy-only (slim) checkpoint loads its actor/backward/normalizer
        and leaves the resume-only nets at their constructed init (fine for play/eval/export).
        """
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "backward": True,
                "critic": True,
                "discriminator": True,
                "target": True,
                "optimizer": True,
                "buffer": True,
                "iteration": True,
            }

        # Warn loudly if a resume was requested but this is a policy-only checkpoint (training state absent)
        wants_resume = any(load_cfg.get(k) for k in ("critic", "optimizer", "buffer"))
        if wants_resume and "forward_map_state_dict" not in loaded_dict:
            print(
                "[WARNING] FbCpr.load: policy-only (slim) checkpoint — critics/optimizers/buffers were NOT"
                " restored; this checkpoint is playable/exportable but not training-resumable."
            )

        def _load(module: torch.nn.Module, key: str) -> None:
            if key in loaded_dict:
                module.load_state_dict(loaded_dict[key], strict=strict)

        # Load the specified models (each guarded so slim checkpoints skip absent keys)
        if load_cfg.get("actor"):
            _load(self.actor, "actor_state_dict")
        if load_cfg.get("backward"):
            _load(self.backward_map, "backward_map_state_dict")
        if load_cfg.get("critic"):
            _load(self.forward_map, "forward_map_state_dict")
            _load(self.disc_critic, "disc_critic_state_dict")
            _load(self.aux_critic, "aux_critic_state_dict")
        if load_cfg.get("discriminator"):
            _load(self.discriminator, "discriminator_state_dict")
        if load_cfg.get("target"):
            for online, target, target_key in (
                (self.forward_map, self.target_forward_map, "target_forward_map_state_dict"),
                (self.backward_map, self.target_backward_map, "target_backward_map_state_dict"),
                (self.disc_critic, self.target_disc_critic, "target_disc_critic_state_dict"),
                (self.aux_critic, self.target_aux_critic, "target_aux_critic_state_dict"),
            ):
                target.target.load_state_dict(loaded_dict.get(target_key, online.state_dict()), strict=strict)
        _load(self.obs_normalizer, "obs_normalizer_state_dict")
        if load_cfg.get("optimizer"):
            for opt, key in (
                (self.actor_optimizer, "actor_optimizer_state_dict"),
                (self.forward_optimizer, "forward_optimizer_state_dict"),
                (self.backward_optimizer, "backward_optimizer_state_dict"),
                (self.disc_critic_optimizer, "disc_critic_optimizer_state_dict"),
                (self.aux_critic_optimizer, "aux_critic_optimizer_state_dict"),
                (self.discriminator_optimizer, "discriminator_optimizer_state_dict"),
            ):
                if key in loaded_dict:
                    opt.load_state_dict(loaded_dict[key])
        if load_cfg.get("buffer"):
            if "z_buffer_state" in loaded_dict:
                self.z_buffer.load_state_dict(loaded_dict["z_buffer_state"])
            if "expert_buffer_state" in loaded_dict:
                self.expert_buffer.load_state_dict(loaded_dict["expert_buffer_state"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: URLVecEnv, cfg: dict, device: str, inference: bool = False) -> FbCpr:
        """Construct the FB-CPR algorithm.

        Set ``inference=True`` to skip loading the expert motion ``TrajectoryBuffer`` (the
        ~GB-scale motion dataset at ``cfg["algorithm"]["motion_path"]``). Only training/eval touch it,
        so play/visualization paths (which just need the actor + obs normalizer) can avoid the disk load.
        """
        # Resolve class callables. The FB-CPR-specific models live on the algorithm cfg; pop them so they
        # aren't re-passed as kwargs to the algorithm constructor below.
        alg_class: type[FbCpr] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FuseModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        forward_map_cfg = cfg["algorithm"].pop("forward_map")
        backward_map_cfg = cfg["algorithm"].pop("backward_map")
        disc_critic_cfg = cfg["algorithm"].pop("disc_critic")
        aux_critic_cfg = cfg["algorithm"].pop("aux_critic")
        discriminator_cfg = cfg["algorithm"].pop("discriminator")
        forward_map_class: type[FuseModel] = resolve_callable(forward_map_cfg.pop("class_name"))  # type: ignore
        backward_map_class: type[MLPModel] = resolve_callable(backward_map_cfg.pop("class_name"))  # type: ignore
        disc_critic_class: type[FuseModel] = resolve_callable(disc_critic_cfg.pop("class_name"))  # type: ignore
        aux_critic_class: type[FuseModel] = resolve_callable(aux_critic_cfg.pop("class_name"))  # type: ignore
        discriminator_class: type[MLPModel] = resolve_callable(discriminator_cfg.pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic", "backward", "discriminator", "expert"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Match TruncatedGaussianDistribution bounds with clip_action bounds.
        actor_dist_cfg = cfg["actor"].get("distribution_cfg")
        if actor_dist_cfg is not None and actor_dist_cfg.get("class_name") == "TruncatedGaussianDistribution":
            clip_actions = cfg["clip_actions"]
            actor_dist_cfg["low"] = -clip_actions
            actor_dist_cfg["high"] = clip_actions

        # Initialize the policy
        z_dim = cfg["algorithm"]["z_dim"]
        actor: FuseModel = actor_class(obs, cfg["obs_groups"], "actor", (z_dim, 0), env.num_actions, **cfg["actor"]).to(
            device
        )
        print(f"Actor Model: {actor}")
        forward_map: FuseModel = forward_map_class(
            obs, cfg["obs_groups"], "critic", (z_dim, env.num_actions), z_dim, **forward_map_cfg
        ).to(device)
        print(f"Forward Map Model: {forward_map}")
        backward_map: MLPModel = backward_map_class(obs, cfg["obs_groups"], "backward", z_dim, **backward_map_cfg).to(
            device
        )
        print(f"Backward Map Model: {backward_map}")
        disc_critic: FuseModel = disc_critic_class(
            obs, cfg["obs_groups"], "critic", (z_dim, env.num_actions), 1, **disc_critic_cfg
        ).to(device)
        print(f"Discriminator Critic Model: {disc_critic}")
        aux_critic: FuseModel = aux_critic_class(
            obs, cfg["obs_groups"], "critic", (z_dim, env.num_actions), 1, **aux_critic_cfg
        ).to(device)
        print(f"Auxiliary Critic Model: {aux_critic}")
        discriminator: MLPModel = discriminator_class(
            obs, cfg["obs_groups"], "discriminator", 1, other_input_dims=(z_dim,), **discriminator_cfg
        ).to(device)
        print(f"Discriminator Model: {discriminator}")
        # Initialize shared observation normalizer across all obs keys used by any model
        all_obs_keys: list[str] = []
        for obs_set in cfg["obs_groups"].values():
            for key in obs_set:
                if key not in all_obs_keys and key in obs:
                    all_obs_keys.append(key)
        obs_normalizer: DictModule[nn.BatchNorm1d] = DictModule({
            key: nn.BatchNorm1d(obs[key].shape[-1], momentum=0.01, affine=False) for key in all_obs_keys
        }).to(device)
        print(f"Observation Normalizer: {obs_normalizer}")

        # Initialize the storage
        max_episode_length = env.max_episode_length
        if isinstance(max_episode_length, torch.Tensor):
            max_episode_length = int(max_episode_length.max().item())
        # inference never samples the storage: keep it at one row per env rather than
        # storage_scale * max_episode_length, which explodes on never-resetting play envs
        replay_buffer = ReplayBuffer(
            env.num_envs,
            cfg["algorithm"]["storage_scale"] * max_episode_length if not inference else 1,
            obs,
            [env.num_actions],
            z_dim,
            cfg["algorithm"]["batch_size"],
            cfg["storage_device"],
        )
        expert_buffer = (
            TrajectoryBuffer(cfg["algorithm"]["motion_path"], cfg["obs_groups"]["expert"], cfg["storage_device"])
            if not inference
            else None
        )
        z_buffer = ZBuffer(cfg["algorithm"]["z_buffer_capacity"], z_dim, cfg["storage_device"])

        # Initialize the algorithm
        alg: FbCpr = alg_class(
            actor,
            forward_map,
            backward_map,
            disc_critic,
            aux_critic,
            discriminator,
            obs_normalizer,
            replay_buffer,
            expert_buffer,
            z_buffer,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
            clip_actions=cfg.get("clip_actions"),
            num_seed_steps_per_env=cfg.get("num_seed_steps_per_env", 0),
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [model.state_dict() for model in self.models]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        for i, model in enumerate(self.models):
            model.load_state_dict(model_params[i])

    def reduce_parameters(self, m: nn.Module) -> None:
        """Average module ``m``'s gradients across all GPUs (call after ``m``'s backward pass).

        Scoped to a single module on purpose: FB-CPR trains several models with separate
        optimizers and backward/step cycles, so each model's gradients must be all-reduced
        independently right before its own ``optimizer.step()`` -- reducing every model's
        parameters here (as a global reduce would) is both incorrect (it would touch other
        models' stale grads) and wasteful (one collective per model instead of per step).
        """
        # Collect this module's populated gradients into one flat buffer for a single collective.
        params = [param for param in m.parameters() if param.grad is not None]
        if not params:
            return
        all_grads = torch.cat([param.grad.view(-1) for param in params])
        # Sum across ranks, then average.
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Scatter the reduced gradients back into the parameters.
        offset = 0
        for param in params:
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project ``z`` onto the sphere of radius ``sqrt(z_dim)``."""
        return math.sqrt(z.shape[-1]) * nn.functional.normalize(z, dim=-1)

    @torch.no_grad()
    def _sample_mixed_z(self, goal_obs: TensorDict, expert_z: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            z = self._sample_random_z(self.batch_size)

            mix_probs = torch.tensor(
                [self.train_goal_ratio, self.expert_asm_ratio, 1 - self.train_goal_ratio - self.expert_asm_ratio],
                device=self.device,
            )
            mix_indices = torch.multinomial(mix_probs, self.batch_size, replacement=True).view(-1, 1)

            # zs for encoded train goals
            perm = torch.randperm(self.batch_size, device=self.device)
            goal_z = self.project_z(self.backward_map(goal_obs[perm]))
            z = torch.where(mix_indices == 0, goal_z, z)

            # zs from expert trajectories
            perm = torch.randperm(self.batch_size, device=self.device)
            z = torch.where(mix_indices == 1, expert_z[perm], z)

        return z

    @torch.no_grad()
    def _encode_expert(self, obs: TensorDict) -> torch.Tensor:
        expert_z = self.backward_map(obs).view(-1, self.expert_sequence_length, self.z_dim).mean(dim=1)
        expert_z = self.project_z(expert_z)
        return torch.repeat_interleave(expert_z, self.expert_sequence_length, dim=0)

    def _sample_random_z(self, size: int) -> torch.Tensor:
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=self.device)
        z = self.project_z(z)
        return z

    def _soft_update_targets(self) -> None:
        """Update params of TD targets from main network params."""
        self.target_forward_map.update()
        self.target_backward_map.update()
        self.target_disc_critic.update()
        self.target_aux_critic.update()

    def _update_discriminator(
        self, batch: ReplayBuffer.Batch, expert_obs: TensorDict, expert_z: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            expert_logits = self.discriminator(expert_obs, expert_z, raw_output=True)
            unlabeled_logits = self.discriminator(batch.observations, batch.context, raw_output=True)
            # Compute loss with binary cross entropy
            expert_loss = -nn.functional.logsigmoid(expert_logits)
            unlabeled_loss = nn.functional.softplus(unlabeled_logits)
            loss = torch.mean(expert_loss + unlabeled_loss)

            # Compute gradient penalty loss
            grad_loss = self.grad_loss_coef * self._gradient_wgan_penalty(
                batch.observations, batch.context, expert_obs, expert_z
            )
            loss += grad_loss

        # Compute the gradients
        self.discriminator_optimizer.zero_grad()
        loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters(self.discriminator)

        # Apply the gradients
        self.discriminator_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Discriminator_Loss/total_loss": loss.mean().detach(),
                "Discriminator_Loss/train_loss": unlabeled_loss.mean().detach(),
                "Discriminator_Loss/expert_loss": expert_loss.mean().detach(),
                "Discriminator_Loss/gradient_loss": grad_loss.mean().detach(),
            }
            extras = {
                "disc_logit_expert": expert_logits.mean().detach(),
                "disc_logit_rollout": unlabeled_logits.mean().detach(),
                "disc_logit_gap": (expert_logits.mean() - unlabeled_logits.mean()).detach(),
            }

        return loss_dict, extras

    def _update_forward_backward(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            # Forward-Backward loss
            with torch.no_grad():
                # Compute successor measure from target networks
                next_actions = self.actor(
                    batch.next_observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
                )
                target_Fs = self.target_forward_map(batch.next_observations, batch.context, next_actions)
                target_B = self.target_backward_map(batch.next_observations)
                target_Ms = torch.matmul(target_Fs, target_B.T)
                target_M = compute_td_targets(target_Ms, self.forward_backward_pessimism)
            # Compute successor measure
            Fs = self.forward_map(batch.observations, batch.context, batch.actions)
            B = self.backward_map(batch.next_observations)
            Ms = torch.matmul(Fs, B.T)

            # FB loss
            diff = Ms - batch.gammas * target_M
            fb_offdiag = 0.5 * (diff * self._off_diag).pow(2).sum() / self._off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # Orthonormality loss
            Cov = torch.matmul(B, B.T)
            orth_offdiag = 0.5 * (Cov * self._off_diag).pow(2).sum() / self._off_diag_sum
            orth_diag = -Cov.diag().mean()
            orth_loss = self.ortho_loss_coef * (orth_offdiag + orth_diag)

            # Fz regularization loss
            q_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
            if self.value_loss_coef > 0.0:
                with torch.no_grad():
                    next_Qs = (target_Fs * batch.context).sum(dim=-1)  # batch_size
                    next_Q = compute_td_targets(next_Qs, self.forward_backward_pessimism)
                    # disable autocast to ensure that cov and B have the same type
                    with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_cov = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_cov * batch.context).sum(dim=-1)  # batch_size
                    target_Q = implicit_reward.detach() + batch.gammas.squeeze() * next_Q  # batch_size
                    target_Q = target_Q.expand(Fs.shape[0], -1)  # num_parallel x batch_size
                Qs = (Fs * batch.context).sum(dim=-1)  # num_parallel x batch_size
                q_loss = self.value_loss_coef * 0.5 * Fs.shape[0] * nn.functional.mse_loss(Qs, target_Q)

            loss = fb_loss + orth_loss + q_loss

        # Compute the gradients
        self.forward_optimizer.zero_grad()
        self.backward_optimizer.zero_grad()
        loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters(self.forward_map)
            self.reduce_parameters(self.backward_map)

        # Clip gradients if enabled
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.forward_map.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.backward_map.parameters(), self.max_grad_norm)

        # Apply the gradients
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Forward_Backward_Loss/total_loss": loss.mean().detach(),
                "Forward_Backward_Loss/fb_loss": fb_loss.mean().detach(),
                "Forward_Backward_Loss/ortho_loss": orth_loss.mean().detach(),
                "Forward_Backward_Loss/value_reg_loss": q_loss.mean().detach(),
            }

            extras = {
                "fb_target_successor_measure": target_M.mean().detach(),
                "fb_successor_measure": Ms[0].mean().detach(),
                "fb_forward": Fs[0].mean().detach(),
                "fb_backward": B.mean().detach(),
                "fb_forward_norm": Fs[0].norm(dim=-1).mean().detach(),
            }

        return loss_dict, extras

    def _update_disc_critic(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            with torch.no_grad():
                # Compute discriminator reward
                logits = self.discriminator(batch.observations, batch.context).clamp_(
                    self.discriminator_reward_eps, 1 - self.discriminator_reward_eps
                )
                discriminator_reward = torch.log(logits) - torch.log(1 - logits)
                # Compute target value
                next_actions = self.actor(
                    batch.next_observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
                )
                next_Qs = self.target_disc_critic(batch.next_observations, batch.context, next_actions)
                target_Q = discriminator_reward + batch.gammas * compute_td_targets(next_Qs, self.disc_critic_pessimism)
                target_Q = target_Q.expand(self.disc_critic.num_parallel, -1, -1)
            # Compute critic loss
            Qs = self.disc_critic(batch.observations, batch.context, batch.actions)
            loss = 0.5 * self.disc_critic.num_parallel * nn.functional.mse_loss(Qs, target_Q)

        # Compute the gradients
        self.disc_critic_optimizer.zero_grad()
        loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters(self.disc_critic)

        # Apply the gradients
        self.disc_critic_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Critic_Loss/discriminator_critic_loss": loss.mean().detach(),
            }

            extras_dict = {
                "discriminator_target_value": target_Q.mean().detach(),
                "discriminator_value": Qs.mean().detach(),
                "discriminator_reward": discriminator_reward.mean().detach(),
            }

        return loss_dict, extras_dict

    def _update_aux_critic(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            with torch.no_grad():
                # Compute target value
                next_actions = self.actor(
                    batch.next_observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
                )
                next_Qs = self.target_aux_critic(batch.next_observations, batch.context, next_actions)
                target_Q = batch.rewards.unsqueeze(1) + batch.gammas * compute_td_targets(
                    next_Qs, self.aux_critic_pessimism
                )
                target_Q = target_Q.expand(self.aux_critic.num_parallel, -1, -1)
            # Compute critic loss
            Qs = self.aux_critic(batch.observations, batch.context, batch.actions)
            loss = 0.5 * self.aux_critic.num_parallel * nn.functional.mse_loss(Qs, target_Q)

        # Compute the gradients
        self.aux_critic_optimizer.zero_grad()
        loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters(self.aux_critic)

        # Apply the gradients
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Critic_Loss/auxiliary_critic_loss": loss.mean().detach(),
            }

            extras_dict = {
                "aux_target_value": target_Q.mean().detach(),
                "aux_value": Qs.mean().detach(),
            }

        return loss_dict, extras_dict

    def _update_actor(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            actions = self.actor(
                batch.observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
            )
            # Compute discriminator value loss
            Qs_discriminator = self.disc_critic(batch.observations, batch.context, actions)
            Q_discriminator = (
                -self.discriminator_reg_coef * compute_td_targets(Qs_discriminator, self.actor_pessimism).mean()
            )
            # Compute auxiliary value loss
            Qs_aux = self.aux_critic(batch.observations, batch.context, actions)
            Q_aux = -self.aux_reg_coef * compute_td_targets(Qs_aux, self.actor_pessimism).mean()
            # Compute forward value loss
            Fs = self.forward_map(batch.observations, batch.context, actions)
            Qs_fb = (Fs * batch.context).sum(dim=-1)
            Q_fb = compute_td_targets(Qs_fb, self.actor_pessimism)
            # Weigh auxiliary and discriminator values by forward value
            reg_weight = Q_fb.abs().mean().detach()
            Q_fb = -Q_fb.mean()
            # Compute the actor loss
            actor_loss = Q_fb + Q_discriminator * reg_weight + Q_aux * reg_weight

        # Compute the gradients
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters(self.actor)

        # Clip gradients if enabled
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        # Apply the gradients
        self.actor_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Actor_Loss/total_loss": actor_loss.mean().detach(),
                "Actor_Loss/discriminator_loss": Q_discriminator.mean().detach(),
                "Actor_Loss/aux_loss": Q_aux.mean().detach(),
                "Actor_Loss/fb_loss": Q_fb.mean().detach(),
            }

            extras_dict = {
                "actor_reg_weight": reg_weight.mean().detach(),
                "actor_F1": Fs[0].mean().detach(),
            }

        return loss_dict, extras_dict

    @torch.compiler.disable
    def _gradient_wgan_penalty(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        expert_obs: TensorDict,
        expert_z: torch.Tensor,
    ) -> torch.Tensor:
        # Get concatenated input tensors
        latent = self.discriminator.get_latent(obs, z)
        expert_latent = self.discriminator.get_latent(expert_obs, expert_z)
        # Interpolate with uniformly sampled alpha
        alpha = torch.rand(self.batch_size, 1, device=self.device)
        interpolated_latent = alpha * latent + (1 - alpha) * expert_latent
        interpolated_latent.requires_grad_(True)
        # Compute interpolated logits
        interpolated_logits = self.discriminator.mlp(interpolated_latent)
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=interpolated_logits,
            inputs=interpolated_latent,
            grad_outputs=torch.ones_like(interpolated_logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        # Compute WGAN penalty from gradients
        return torch.mean(torch.square(gradients.norm(p=2, dim=1) - 1))
