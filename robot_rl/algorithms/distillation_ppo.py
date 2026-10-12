"""DAgger distillation with a PPO term on the task reward, in one update (Distillation-PPO)."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.distillation import Distillation
from robot_rl.env import VecEnv
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage
from robot_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class DistillationPPO(Distillation):
    """A student trained to copy a privileged teacher and to maximise the task reward at the same time.

    The student rolls out stochastically on its own observations. Each update replays the rollout in time order,
    as :class:`Distillation` does, and sums the behavior-cloning loss against the teacher's actions with a clipped
    PPO surrogate on a privileged critic's advantages. The behavior-cloning weight anneals over updates, so the
    teacher's actions dominate early and the reward takes over where the student's observations make copying
    impossible.
    """

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        critic_learning_rate: float | None = None,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        max_learning_rate: float = 1e-2,
        min_learning_rate: float = 1e-5,
        max_grad_norm: float | None = None,
        loss_type: str = "nll",
        aux_obs_group: str | None = None,
        aux_loss_weight: float = 1.0,
        optimizer: str = "adam",
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_param: float = 0.2,
        entropy_coef: float = 0.0,
        value_loss_coef: float = 1.0,
        use_clipped_value_loss: bool = True,
        bc_weight: float = 1.0,
        bc_weight_final: float = 0.0,
        bc_anneal_updates: int = 1,
        critic_warmup_updates: int = 0,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            student: The policy being trained, with an output distribution.
            teacher: The privileged policy whose actions label the student's rollouts.
            critic: State-value model over the privileged critic observation set.
            storage: Rollout storage in ``distillation_rl`` mode.
            num_learning_epochs: Replays of each rollout per update.
            gradient_length: Timesteps accumulated before each gradient step.
            learning_rate: Student learning rate; the starting value under the adaptive schedule.
            critic_learning_rate: Critic learning rate, or None to share ``learning_rate``.
            schedule: ``"adaptive"`` rescales the student's rate by 1.5 whenever the rollout-to-replay KL leaves
                ``[desired_kl / 2, 2 * desired_kl]``; ``"fixed"`` keeps it.
            desired_kl: KL target of the adaptive schedule.
            max_learning_rate: Ceiling of the adaptive schedule.
            min_learning_rate: Floor of the adaptive schedule.
            max_grad_norm: Gradient-norm clip, or None.
            loss_type: Behavior-cloning loss: ``"nll"`` is the teacher action's negative log-likelihood under
                the student's distribution, in the surrogate's units; ``"mse"``/``"huber"`` act on the mean.
            aux_obs_group: Observation group an auxiliary head on the student regresses, or None.
            aux_loss_weight: Weight of that auxiliary loss.
            optimizer: Optimizer name.
            gamma: Discount factor.
            lam: GAE lambda.
            clip_param: PPO clip range for the policy ratio and the value estimate.
            entropy_coef: Entropy bonus weight.
            value_loss_coef: Value loss weight.
            use_clipped_value_loss: Clip the value estimate against the rollout's, as in PPO.
            bc_weight: Behavior-cloning weight at the first update.
            bc_weight_final: Behavior-cloning weight reached after ``bc_anneal_updates`` updates and held.
            bc_anneal_updates: Updates over which the weight moves linearly from ``bc_weight`` to ``bc_weight_final``.
            critic_warmup_updates: Leading updates that train the critic and the cloning loss only, so the
                surrogate never acts on an untrained value estimate.
            device: Device for models and losses.
            multi_gpu_cfg: Distributed training configuration, or None.
            **kwargs: Unused configuration entries.
        """
        self.nll_cloning = loss_type == "nll"
        super().__init__(
            student,
            teacher,
            storage,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type="mse" if self.nll_cloning else loss_type,
            aux_obs_group=aux_obs_group,
            aux_loss_weight=aux_loss_weight,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.critic = critic.to(self.device)
        self._raw_critic = self.critic
        self.learning_rate = learning_rate
        self.critic_learning_rate = learning_rate if critic_learning_rate is None else critic_learning_rate
        self.optimizer = resolve_optimizer(optimizer)([
            {"params": list(self.student.parameters()), "lr": self.learning_rate},
            {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate},
        ])
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.bc_weight = bc_weight
        self.bc_weight_final = bc_weight_final
        self.bc_anneal_updates = max(1, bc_anneal_updates)
        self.critic_warmup_updates = critic_warmup_updates

    @property
    def current_bc_weight(self) -> float:
        """The behavior-cloning weight of the next update."""
        frac = min(1.0, self.num_updates / self.bc_anneal_updates)
        return self.bc_weight + frac * (self.bc_weight_final - self.bc_weight)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample a student action and record it with its log-probability, the teacher's action and the value."""
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.actions_log_prob = self.student.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.student.output_distribution_params)
        self.transition.privileged_actions = self.teacher(obs).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step, bootstrapping time-outs, and update the normalizers."""
        self.student.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,  # type: ignore
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE returns and normalized advantages over the stored rollout."""
        st = self.storage
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach().to(st.device)
        self.critic.reset(hidden_state=critic_hidden_state)
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Replay the rollout in time order and step on the summed cloning, surrogate, value and entropy losses."""
        bc_weight = self.current_bc_weight
        rl_on = self.num_updates >= self.critic_warmup_updates
        self.num_updates += 1
        sums = {
            "behavior": 0.0,
            "surrogate": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "kl": 0.0,
            "aux": 0.0,
        }
        loss = 0
        cnt = 0
        chunk_kl = 0.0

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                latent = self.student.get_latent(batch.observations)
                self.student.forward_from_latent(latent, stochastic_output=True)
                if self.nll_cloning:
                    behavior_loss = -self.student.get_output_log_prob(batch.privileged_actions).mean()
                else:
                    behavior_loss = self.loss_fn(self.student.output_mean, batch.privileged_actions)
                step_loss = bc_weight * behavior_loss

                with torch.inference_mode():
                    kl = self.student.get_kl_divergence(
                        batch.old_distribution_params, self.student.output_distribution_params
                    ).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= self.gpu_world_size
                    chunk_kl += kl.item()
                    sums["kl"] += kl.item()

                log_prob = self.student.get_output_log_prob(batch.actions)
                ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
                advantages = torch.squeeze(batch.advantages)
                surrogate = -advantages * ratio
                surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                entropy = self.student.output_entropy.mean()

                values = self.critic(batch.observations)
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_loss = torch.max(
                        (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)
                    ).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                step_loss = step_loss + self.value_loss_coef * value_loss
                if rl_on:
                    step_loss = step_loss + surrogate_loss - self.entropy_coef * entropy
                if self.aux_obs_group is not None:
                    aux_pred = self.student.aux_prediction(latent)  # type: ignore[attr-defined]
                    aux_loss = nn.functional.mse_loss(aux_pred, batch.observations[self.aux_obs_group])
                    step_loss = step_loss + self.aux_loss_weight * aux_loss
                    sums["aux"] += aux_loss.item()
                loss = loss + step_loss
                cnt += 1

                with torch.no_grad():
                    sums["behavior"] += behavior_loss.item()
                    sums["surrogate"] += surrogate_loss.item()
                    sums["value"] += value_loss.item()
                    sums["entropy"] += entropy.item()
                    sums["clip_fraction"] += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                if cnt % self.gradient_length == 0:
                    if rl_on:
                        self._adapt_learning_rate(chunk_kl / self.gradient_length)
                    chunk_kl = 0.0
                    self._step(loss)
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.critic.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        if cnt % self.gradient_length != 0:
            self._step(loss)

        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        loss_dict = {key: value / max(cnt, 1) for key, value in sums.items()}
        loss_dict["bc_weight"] = bc_weight
        loss_dict["learning_rate"] = self.learning_rate
        if self.aux_obs_group is None:
            loss_dict.pop("aux")
        return loss_dict

    def _adapt_learning_rate(self, kl: float) -> None:
        """Move the student's learning rate toward the KL target, as PPO's adaptive schedule does."""
        if self.schedule != "adaptive" or self.desired_kl is None:
            return
        if kl > self.desired_kl * 2.0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
        self.optimizer.param_groups[0]["lr"] = self.learning_rate

    def _step(self, loss: torch.Tensor) -> None:
        """Apply one gradient step of the summed loss to the student and the critic."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.student.detach_hidden_state()

    def train_mode(self) -> None:
        """Set train mode for the student and the critic; the teacher stays in eval mode."""
        super().train_mode()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for every model."""
        super().eval_mode()
        self.critic.eval()

    def save(self) -> dict:
        """Return the student, teacher, critic and optimizer states."""
        saved = super().save()
        saved["critic_state_dict"] = self._raw_critic.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the models a checkpoint holds; a teacher-only (RL) checkpoint leaves the student and critic fresh."""
        from_rl = load_cfg is None and any("actor_state_dict" in key for key in loaded_dict)
        resume = super().load(loaded_dict, load_cfg, strict)
        if not from_rl and (load_cfg is None or load_cfg.get("student")) and "critic_state_dict" in loaded_dict:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        return resume

    def compile(self, mode: str | None = None) -> None:
        """Compile the student, teacher and critic with ``torch.compile``."""
        super().compile(mode)
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DistillationPPO:
        """Build the algorithm from the runner config, which names the student, teacher and critic models."""
        cfg = copy.deepcopy(cfg)
        alg_class: type[DistillationPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher", "critic"])
        for ext in ("rnd_cfg", "symmetry_cfg"):
            if cfg["algorithm"].get(ext) is not None:
                raise ValueError(f"The {ext} extension is not compatible with DistillationPPO.")
            cfg["algorithm"][ext] = None

        aux_group = cfg["algorithm"].get("aux_obs_group")
        if aux_group is not None:
            cfg["student"]["aux_target_dim"] = obs[aux_group].shape[-1]
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"])
        if student.distribution is None:
            raise ValueError("DistillationPPO needs a student with an output distribution (distribution_cfg).")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"])
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"])
        print(f"Student Model: {student}\nTeacher Model: {teacher}\nCritic Model: {critic}")

        storage = RolloutStorage(
            "distillation_rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        alg: DistillationPPO = alg_class(
            student, teacher, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast the student, teacher and critic parameters from rank 0."""
        params = [self._raw_student.state_dict(), self._raw_teacher.state_dict(), self._raw_critic.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self._raw_student.load_state_dict(params[0])
        self._raw_teacher.load_state_dict(params[1])
        self._raw_critic.load_state_dict(params[2])

    def reduce_parameters(self) -> None:
        """Average the student's and critic's gradients across GPUs in place."""
        params = [p for p in list(self.student.parameters()) + list(self.critic.parameters()) if p.grad is not None]
        all_grads = torch.cat([p.grad.view(-1) for p in params])
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            n = p.numel()
            p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad.data))
            offset += n
