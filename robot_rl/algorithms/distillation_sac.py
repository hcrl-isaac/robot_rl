# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation-SAC: a recurrent student trained by SAC on the task reward plus cloning of a privileged teacher."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Callable
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.sac import SAC
from robot_rl.env import VecEnv
from robot_rl.models import FuseModel, MLPModel
from robot_rl.storage import ReplayBuffer
from robot_rl.utils import resolve_callable, resolve_obs_groups


class DistillationSAC(SAC):
    """SAC over a recurrent student whose actor loss also clones a frozen teacher, with an annealed weight.

    The student acts on its own observation set; the twin critics read the privileged ``critic`` set, so a
    teacher SAC checkpoint over the same set can initialize them. Teacher labels are computed on replayed
    steps from the stored ``teacher`` observations, so the cloning targets follow the student's own visits.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic_1: FuseModel,
        critic_2: FuseModel,
        replay_buffer: ReplayBuffer,
        num_actions: int,
        teacher: MLPModel | None = None,
        loss_type: str = "nll",
        bc_weight: float = 1.0,
        bc_weight_final: float = 0.0,
        bc_anneal_updates: int = 1,
        critic_warmup_updates: int = 0,
        normalize_q: bool = True,
        init_critic_from_teacher: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            actor: The recurrent student policy.
            critic_1: First twin Q-critic over the privileged ``critic`` set.
            critic_2: Second twin Q-critic.
            replay_buffer: Replay buffer storing every observation group, the teacher's included.
            num_actions: Action width.
            teacher: The frozen privileged policy whose deterministic action labels each replayed step.
            loss_type: Cloning loss: ``"nll"`` is the teacher action's negative log-likelihood under the student;
                ``"mse"`` acts on the student's mean.
            bc_weight: Cloning weight at the first update.
            bc_weight_final: Cloning weight reached after ``bc_anneal_updates`` updates and held.
            bc_anneal_updates: Updates (``update()`` calls) over which the weight moves linearly.
            critic_warmup_updates: Leading updates that train the actor by cloning alone while the critics fit.
            normalize_q: Divide the soft-Q term by the batch's mean ``|Q|``, so the cloning weight is unit-free.
            init_critic_from_teacher: Load the critics and temperature from a teacher SAC checkpoint.
            **kwargs: SAC arguments.
        """
        if not actor.is_recurrent:
            raise ValueError("DistillationSAC trains a recurrent student; use a recurrent actor model.")
        if kwargs.get("n_steps", 1) != 1:
            raise ValueError("DistillationSAC bootstraps one step through the recurrent actor; set n_steps=1.")
        if teacher is None:
            raise ValueError("DistillationSAC needs a teacher model.")
        if loss_type not in ("nll", "mse"):
            raise ValueError(f"Unknown cloning loss: {loss_type}")
        super().__init__(actor, critic_1, critic_2, replay_buffer, num_actions, **kwargs)
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_loaded = False
        self.loss_type = loss_type
        self.bc_weight = bc_weight
        self.bc_weight_final = bc_weight_final
        self.bc_anneal_updates = max(1, bc_anneal_updates)
        self.critic_warmup_updates = critic_warmup_updates
        self.normalize_q = normalize_q
        self.init_critic_from_teacher = init_critic_from_teacher
        self.num_updates = 0

    @property
    def current_bc_weight(self) -> float:
        """Return the cloning weight of the next update."""
        frac = min(1.0, self.num_updates / self.bc_anneal_updates)
        return self.bc_weight + frac * (self.bc_weight_final - self.bc_weight)

    def update(self) -> tuple[dict, dict]:
        """Run the recurrent SAC updates with the cloning term; returns mean losses and diagnostics."""
        if not self.teacher_loaded:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")
        self._bc_now = self.current_bc_weight
        self._rl_on = self.num_updates >= self.critic_warmup_updates
        losses, diag = self._update_recurrent()
        self.num_updates += 1
        losses["bc_weight"] = self._bc_now
        losses["rl_on"] = float(self._rl_on)
        return losses, diag

    def _actor_rl_scale(
        self, min_q_pi: torch.Tensor, acc: Callable[[str, torch.Tensor | float], None]
    ) -> torch.Tensor | float:
        """Return zero during critic warm-up, else ``1 / mean|Q|`` when normalizing, else 1."""
        if not self._rl_on:
            return 0.0
        if not self.normalize_q:
            return 1.0
        q_scale = min_q_pi.detach().abs().mean().clamp(min=1e-6)
        acc("Distill/q_scale", q_scale)
        return 1.0 / q_scale

    def _extra_actor_loss(
        self, obs: TensorDict, acc: Callable[[str, torch.Tensor | float], None]
    ) -> torch.Tensor | None:
        """Return the weighted cloning loss against the teacher's action on the same replayed steps."""
        with torch.no_grad():
            teacher_actions = self.teacher(obs)
        if self.loss_type == "nll":
            behavior = -self.actor.get_output_log_prob(teacher_actions).mean()
        else:
            behavior = nn.functional.mse_loss(self.actor.output_mean, teacher_actions)
        with torch.no_grad():
            acc("Distill/behavior", behavior)
            student_mean = self.actor.output_mean
            acc("Distill/teacher_cos", nn.functional.cosine_similarity(student_mean, teacher_actions, dim=-1).mean())
        return self._bc_now * behavior

    def train_mode(self) -> None:
        """Set the student and critics to training mode; the teacher stays in eval mode."""
        super().train_mode()
        self.teacher.eval()

    def save(self) -> dict:
        """Return the student, teacher, critic, optimizer and temperature states."""
        saved = super().save()
        saved["student_state_dict"] = saved.pop("actor_state_dict")
        saved["teacher_state_dict"] = self.teacher.state_dict()
        saved["num_updates"] = self.num_updates
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Resume from a Distillation-SAC checkpoint, or take the teacher (and optionally its critics) from an RL one.

        Args:
            loaded_dict: The checkpoint.
            load_cfg: For a Distillation-SAC checkpoint, which states to load (see :meth:`SAC.load`).
            strict: Whether state-dict loading is strict.

        Returns:
            Whether the training iteration should be restored.
        """
        if "student_state_dict" in loaded_dict:
            self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=strict)
            self.teacher_loaded = True
            self.num_updates = int(loaded_dict.get("num_updates", 0))
            sac_dict = dict(loaded_dict)
            sac_dict["actor_state_dict"] = sac_dict.pop("student_state_dict")
            return super().load(sac_dict, load_cfg, strict)
        if "actor_state_dict" not in loaded_dict:
            raise ValueError("The checkpoint has neither student nor teacher weights.")
        self.teacher.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.teacher_loaded = True
        if self.init_critic_from_teacher:
            if "critic_1_state_dict" not in loaded_dict:
                raise ValueError("init_critic_from_teacher is set, but the teacher checkpoint has no critics.")
            self.critic_1.load_state_dict(loaded_dict["critic_1_state_dict"], strict=strict)
            self.critic_2.load_state_dict(loaded_dict["critic_2_state_dict"], strict=strict)
            self.critic_1_target.hard_sync()
            self.critic_2_target.hard_sync()
            if "log_alpha" in loaded_dict:
                self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp().item()
        return False

    def broadcast_parameters(self) -> None:
        """Broadcast the student, critics and teacher from rank 0 to all GPUs."""
        super().broadcast_parameters()
        params = [self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self.teacher.load_state_dict(params[0])

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str, inference: bool = False
    ) -> DistillationSAC:
        """Build the student, teacher, twin critics and replay buffer from a config dict.

        ``inference=True`` builds a minimal replay buffer for play and eval.
        """
        cfg = copy.deepcopy(cfg)
        alg_class: type[DistillationSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore
        critic_class: type[FuseModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        for key in ("rnd_cfg", "symmetry_cfg", "estimator_obs_group"):
            if cfg["algorithm"].get(key) is not None:
                raise ValueError(f"DistillationSAC does not support {key}.")
            cfg["algorithm"].pop(key, None)
        cfg["algorithm"].pop("estimator_hidden_dim", None)
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher", "critic"])

        num_actions = env.num_actions
        aux_group = cfg["algorithm"].get("aux_obs_group")
        if aux_group is not None:
            cfg["student"]["aux_target_dim"] = obs[aux_group].shape[-1]
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", num_actions, **cfg["student"]).to(device)
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", num_actions, **cfg["teacher"]).to(device)
        critic_1: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)
        critic_2: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)
        replay_buffer = SAC.build_replay_buffer(obs, env, cfg, student, device, inference)
        return alg_class(
            student,
            critic_1,
            critic_2,
            replay_buffer,
            num_actions,
            teacher=teacher,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
