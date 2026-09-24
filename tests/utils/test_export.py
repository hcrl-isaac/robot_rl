# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for policy export: the ONNX helper and the env-free checkpoint rebuild."""

from __future__ import annotations

import torch
from pathlib import Path
from torch import nn
from typing import ClassVar

import onnx
import pytest

from robot_rl.models import EncoderInferencePolicy
from robot_rl.utils.export import rebuild_models, save_onnx
from tests.algorithms.test_ppo import (
    LATENT_DIM,
    NUM_ENVS,
    _build_ppo,
    _build_ppo_with_encoder,
)


class _Doubler(nn.Module):
    """Minimal module satisfying the export protocol (dummy inputs + input/output names)."""

    input_names: ClassVar[list[str]] = ["x"]
    output_names: ClassVar[list[str]] = ["y"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double the input."""
        return x * 2.0

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Tracing inputs for the export."""
        return (torch.zeros(1, 3),)


def _default_domain_opsets(path: str) -> list[int]:
    """Opset versions the saved model declares for the default ONNX domain."""
    model = onnx.load(path)
    return [i.version for i in model.opset_import if i.domain in ("", "ai.onnx")]


class TestSaveOnnx:
    """Tests for ``save_onnx``."""

    @pytest.mark.parametrize("opset", [18, 20])
    def test_writes_the_requested_opset(self, tmp_path: Path, opset: int) -> None:
        """A module needing a newer operator can raise the opset the export targets."""
        path = save_onnx(_Doubler(), str(tmp_path), "doubler.onnx", opset=opset)

        assert _default_domain_opsets(path) == [opset]

    def test_defaults_to_opset_18(self, tmp_path: Path) -> None:
        """Callers that do not care keep the historical default."""
        path = save_onnx(_Doubler(), str(tmp_path), "doubler.onnx")

        assert _default_domain_opsets(path) == [18]


_ACTOR_CFG = {
    "class_name": "MLPModel",
    "hidden_dims": [32, 32],
    "activation": "elu",
    "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
}
_CRITIC_CFG = {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu"}
_ENCODER_MODEL_CFG = {"class_name": "MLPModel", "hidden_dims": [16, 8], "activation": "elu"}


def _train_cfg(with_encoder: bool) -> dict:
    """Build the subset of a saved ``agent.yaml`` that the rebuild reads."""
    cfg: dict = {
        "actor": dict(_ACTOR_CFG),
        "critic": dict(_CRITIC_CFG),
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {},
    }
    if with_encoder:
        cfg["obs_groups"]["encoder"] = ["scan"]
        cfg["algorithm"]["encoder_cfg"] = {
            "model": dict(_ENCODER_MODEL_CFG),
            "output_dim": LATENT_DIM,
        }
    return cfg


def test_rebuild_without_encoder_matches_the_actor() -> None:
    """The plain rebuild path is unchanged: it reproduces the trained actor exactly."""
    ppo, obs = _build_ppo()
    ppo.eval_mode()

    policy = rebuild_models(_train_cfg(with_encoder=False), ppo.save())["policy"]
    with torch.inference_mode():
        expected = ppo.get_policy()(obs, stochastic_output=False)
        actual = policy(obs, stochastic_output=False)
    torch.testing.assert_close(expected, actual)


def test_rebuild_with_encoder_matches_the_encoder_policy() -> None:
    """On an encoder run the rebuild subtracts the latent width and restores the encoder."""
    ppo, obs = _build_ppo_with_encoder()
    ppo.eval_mode()

    policy = rebuild_models(_train_cfg(with_encoder=True), ppo.save())["policy"]
    assert isinstance(policy, EncoderInferencePolicy)

    with torch.inference_mode():
        expected = ppo.get_policy()(obs, stochastic_output=False)
        actual = policy(obs, stochastic_output=False)
    torch.testing.assert_close(expected, actual)


def test_rebuilt_encoder_policy_exports_to_jit() -> None:
    """The rebuilt encoder policy scripts, and the single-input export matches eager inference."""
    ppo, obs = _build_ppo_with_encoder()
    ppo.eval_mode()

    policy = rebuild_models(_train_cfg(with_encoder=True), ppo.save())["policy"]
    with torch.inference_mode():
        expected = policy(obs, stochastic_output=False)
        scripted = torch.jit.script(policy.as_jit())
        actual = scripted(torch.cat([obs["policy"], obs["scan"]], dim=-1))
    assert actual.shape[0] == NUM_ENVS
    torch.testing.assert_close(expected, actual)


def test_encoder_jit_exposes_the_gaussian_distribution() -> None:
    """``forward_dist`` returns the eager mean and the actor's exported std."""
    ppo, obs = _build_ppo_with_encoder()
    ppo.eval_mode()

    policy = rebuild_models(_train_cfg(with_encoder=True), ppo.save())["policy"]
    scripted = torch.jit.script(policy.as_jit())
    with torch.inference_mode():
        mean, std = scripted.forward_dist(torch.cat([obs["policy"], obs["scan"]], dim=-1))
        expected = policy(obs, stochastic_output=False)
    torch.testing.assert_close(mean, expected)
    torch.testing.assert_close(std, policy.actor.distribution.export_std().expand_as(mean))


def test_rebuilt_encoder_policy_exports_to_onnx(tmp_path: Path) -> None:
    """The ONNX export of an encoder policy matches eager inference on the concatenated input."""
    import onnxruntime as ort

    ppo, obs = _build_ppo_with_encoder()
    ppo.eval_mode()

    policy = rebuild_models(_train_cfg(with_encoder=True), ppo.save())["policy"]
    path = save_onnx(policy.as_onnx(), str(tmp_path), "policy.onnx")
    # the export traces a batch of one, like every ONNX export here
    x = torch.cat([obs["policy"], obs["scan"]], dim=-1)[:1]
    with torch.inference_mode():
        expected = policy(obs[:1], stochastic_output=False)
    session = ort.InferenceSession(path)
    (actual,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)


def test_rebuild_distillation_matches_the_student() -> None:
    """A distillation checkpoint exports its student; the teacher was the training signal, not a target."""
    from robot_rl.algorithms.distillation import Distillation
    from robot_rl.models import MLPModel
    from robot_rl.storage import RolloutStorage
    from tests.conftest import make_obs

    num_envs, obs_dim, num_actions = 4, 8, 4
    obs = make_obs(num_envs, obs_dim)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}
    student_cfg = {
        "class_name": "MLPModel",
        "hidden_dims": [32, 32],
        "distribution_cfg": dict(_ACTOR_CFG["distribution_cfg"]),
    }
    student = MLPModel(
        obs,
        obs_groups,
        "student",
        num_actions,
        hidden_dims=[32, 32],
        distribution_cfg=dict(_ACTOR_CFG["distribution_cfg"]),
    )
    teacher = MLPModel(obs, obs_groups, "teacher", num_actions, hidden_dims=[32, 32])
    alg = Distillation(student, teacher, RolloutStorage("distillation", num_envs, 4, obs, [num_actions]))
    alg.eval_mode()

    train_cfg = {"student": student_cfg, "obs_groups": obs_groups, "algorithm": {}}
    policy = rebuild_models(train_cfg, alg.save())["policy"]
    with torch.inference_mode():
        expected = alg.student(obs, stochastic_output=False)
        actual = policy(obs, stochastic_output=False)
    torch.testing.assert_close(expected, actual)
