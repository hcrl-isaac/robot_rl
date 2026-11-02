# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.algorithms.distillation import Distillation
from robot_rl.models import CVAEModel, MLPModel
from robot_rl.storage import RolloutStorage


class CvaeDistillation(Distillation):
    """Masked online distillation into a CVAE (arXiv 2509.13780): DAgger with a KL term to the learned prior.

    The student acts from encoder samples (privileged), the teacher labels every visited state, and the update
    minimizes ``||a_teacher - decoder(z ~ q)||^2 + kl_weight * KL(q || prior)``.
    """

    student: CVAEModel

    def __init__(
        self,
        student: CVAEModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        kl_weight: float = 0.01,
        kl_weight_final: float = 0.001,
        kl_anneal_iterations: int = 0,
        prior_rollout_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm.

        Args:
            student: The CVAE model.
            teacher: The privileged teacher.
            storage: Rollout storage of type ``"distillation"``.
            kl_weight: Initial weight of the KL term.
            kl_weight_final: KL weight reached after ``kl_anneal_iterations`` updates.
            kl_anneal_iterations: Updates over which the KL weight anneals linearly; ``0`` keeps it constant.
            prior_rollout_ratio: Fraction of envs acting from the prior mean instead of an encoder sample.
            **kwargs: Forwarded to :class:`Distillation`.
        """
        super().__init__(student, teacher, storage, **kwargs)
        self.kl_weight_init = kl_weight
        self.kl_weight_final = kl_weight_final
        self.kl_anneal_iterations = kl_anneal_iterations
        self.kl_weight = kl_weight
        self.prior_rollout_ratio = prior_rollout_ratio

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample the student's action from an encoder draw and record the teacher's label."""
        self.transition.actions = self.student.act_train(obs, self.prior_rollout_ratio).detach()
        self.transition.privileged_actions = self.teacher(obs).detach()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def update(self) -> dict[str, float]:
        """Run optimization epochs over the stored batches and return mean losses."""
        self.num_updates += 1
        if self.kl_anneal_iterations > 0:
            frac = min(1.0, self.num_updates / self.kl_anneal_iterations)
            self.kl_weight = self.kl_weight_init + frac * (self.kl_weight_final - self.kl_weight_init)
        totals = {"behavior": 0.0, "kl": 0.0, "prior_behavior": 0.0}
        loss = 0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                obs = batch.observations
                mu_p, std_p = self.student.prior_params(obs)
                mu_q, std_q = self.student.posterior_params(obs, mu_p)
                z = mu_q + std_q * torch.randn_like(mu_q)
                behavior_loss = self.loss_fn(self.student.decode(obs, z), batch.privileged_actions)
                kl = self.student.kl_divergence(mu_q, std_q, mu_p, std_p).mean()
                loss = loss + behavior_loss + self.kl_weight * kl
                with torch.no_grad():
                    # how well the deployed (prior-mean) path imitates; diagnostic only
                    prior_loss = self.loss_fn(self.student.decode(obs, mu_p), batch.privileged_actions)
                totals["behavior"] += behavior_loss.item()
                totals["kl"] += kl.item()
                totals["prior_behavior"] += prior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0

        self.storage.clear()
        loss_dict = {key: value / cnt for key, value in totals.items()}
        loss_dict["kl_weight"] = self.kl_weight
        return loss_dict
