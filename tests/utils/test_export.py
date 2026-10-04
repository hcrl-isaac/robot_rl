# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for policy export: the ONNX helper and the env-free checkpoint rebuild."""

from __future__ import annotations

import copy
import numpy as np
import torch
from pathlib import Path
from torch import nn
from typing import ClassVar

import onnx
import pytest

from robot_rl.models import EncoderInferencePolicy
from robot_rl.utils.export import rebuild_models, save_onnx
from robot_rl.utils.utils import resolve_callable
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


_STUDENT_DISTRIBUTIONS = {
    "Gaussian": dict(_ACTOR_CFG["distribution_cfg"]),
    "HeteroscedasticGaussian": {"class_name": "HeteroscedasticGaussianDistribution"},
    "TruncatedGaussian": {"class_name": "TruncatedGaussianDistribution"},
    "Beta": {"class_name": "BetaDistribution"},
    "VonMisesFisher": {"class_name": "VonMisesFisherDistribution", "init_std": 0.5},
    "SquashedTanhGaussian": {"class_name": "SquashedTanhGaussianDistribution"},
    "none": None,
}


def _distillation_checkpoint(student_cfg: dict, num_actions: int = 4) -> tuple[dict, dict, object, object]:
    """Train-cfg subset, checkpoint, observations and algorithm of a fresh distillation run."""
    from robot_rl.algorithms.distillation import Distillation
    from robot_rl.models import MLPModel
    from robot_rl.storage import RolloutStorage
    from tests.conftest import make_obs

    num_envs, obs_dim = 4, 8
    obs = make_obs(num_envs, obs_dim)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}
    model_cfg = {k: copy.deepcopy(v) for k, v in student_cfg.items() if k != "class_name"}
    student = resolve_callable(student_cfg["class_name"])(obs, obs_groups, "student", num_actions, **model_cfg)
    teacher = MLPModel(obs, obs_groups, "teacher", num_actions, hidden_dims=[32, 32])
    alg = Distillation(student, teacher, RolloutStorage("distillation", num_envs, 4, obs, [num_actions]))
    alg.eval_mode()
    train_cfg = {"student": student_cfg, "obs_groups": obs_groups, "algorithm": {}}
    return train_cfg, alg.save(), obs, alg


@pytest.mark.parametrize("distribution", list(_STUDENT_DISTRIBUTIONS))
def test_rebuild_distillation_matches_the_student(distribution: str) -> None:
    """Rebuild a distillation student from its checkpoint."""
    student_cfg = {"class_name": "MLPModel", "hidden_dims": [32, 32]}
    if _STUDENT_DISTRIBUTIONS[distribution] is not None:
        student_cfg["distribution_cfg"] = dict(_STUDENT_DISTRIBUTIONS[distribution])
    train_cfg, ckpt, obs, alg = _distillation_checkpoint(student_cfg)

    policy = rebuild_models(train_cfg, ckpt)["policy"]
    with torch.inference_mode():
        expected = alg.student(obs, stochastic_output=False)
        actual = policy(obs, stochastic_output=False)
    torch.testing.assert_close(expected, actual)


def _image_student(rnn_type: str) -> tuple[nn.Module, dict, dict]:
    """Build a CNN-RNN vMF student over a proprioceptive group and a 4-frame depth image, with fitted stats."""
    from tensordict import TensorDict

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(2, 7), "image": torch.zeros(2, 4, 12, 20)}, batch_size=[2])
    obs_groups = {"student": ["policy", "image"], "teacher": ["policy"]}
    student_cfg = {
        "class_name": "CNNRNNModel",
        "hidden_dims": [32, 32],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "VonMisesFisherDistribution", "init_std": 0.05},
        "cnn_cfg": {"output_channels": [8, 8], "kernel_size": 3, "stride": 2, "global_pool": "avg", "flatten": True},
        "rnn_type": rnn_type,
        "rnn_hidden_dim": 16,
    }
    model_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in student_cfg.items() if k != "class_name"}
    model = CNNRNNModel(obs, obs_groups, "student", 6, **model_cfg)
    model.obs_normalizer._mean.uniform_(-1.0, 1.0)
    model.obs_normalizer._std.uniform_(0.5, 2.0)
    train_cfg = {"student": student_cfg, "obs_groups": obs_groups, "algorithm": {}}
    return model.eval(), train_cfg, {"student_state_dict": model.state_dict()}


def _steps(seed: int = 0, count: int = 5) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return a short episode of random ``(proprioception, image)`` observations."""
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(1, 7, generator=generator), torch.rand(1, 4, 12, 20, generator=generator)) for _ in range(count)
    ]


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_an_image_student_exports_to_onnx_with_its_state_threaded_through(tmp_path: Path, rnn_type: str) -> None:
    """The exported student, fed back its own state each step, acts as the trained one does over an episode."""
    from tensordict import TensorDict

    import onnxruntime as ort

    model, train_cfg, ckpt = _image_student(rnn_type)
    policy = rebuild_models(train_cfg, ckpt, obs_shapes={"image": (4, 12, 20)})["policy"]
    path = save_onnx(policy.as_onnx(), str(tmp_path), "student.onnx")
    session = ort.InferenceSession(path)
    names = [i.name for i in session.get_inputs()]
    assert names == ["obs", "image", "h_in", *(["c_in"] if rnn_type == "lstm" else [])]

    state = {name: np.zeros((1, 1, 16), dtype=np.float32) for name in names[2:]}
    model.reset()
    with torch.inference_mode():
        for proprio, image in _steps():
            expected = model(TensorDict({"policy": proprio, "image": image}, batch_size=[1]), stochastic_output=False)
            outputs = session.run(None, {"obs": proprio.numpy(), "image": image.numpy(), **state})
            torch.testing.assert_close(torch.from_numpy(outputs[0]), expected, rtol=1e-4, atol=1e-5)
            state = dict(zip(names[2:], outputs[1:], strict=True))
    assert np.linalg.norm(outputs[0]) == pytest.approx(1.0, abs=1e-5)  # a vMF acts with its unit mean direction


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_an_image_student_exports_to_jit_carrying_its_state(tmp_path: Path, rnn_type: str) -> None:
    """The scripted student carries its own state, and a reset starts the next episode from zeros."""
    from tensordict import TensorDict

    from robot_rl.utils.export import save_jit

    model, train_cfg, ckpt = _image_student(rnn_type)
    policy = rebuild_models(train_cfg, ckpt, obs_shapes={"image": (4, 12, 20)})["policy"]
    scripted = torch.jit.load(save_jit(policy.as_jit(), str(tmp_path), "student.pt"))
    for _ in range(2):  # a reset starts the second episode from the same zero state as the first
        model.reset()
        scripted.reset()
        with torch.inference_mode():
            for proprio, image in _steps():
                expected = model(
                    TensorDict({"policy": proprio, "image": image}, batch_size=[1]), stochastic_output=False
                )
                torch.testing.assert_close(scripted(proprio, [image]), expected, rtol=1e-4, atol=1e-5)


def test_an_image_student_needs_its_image_shapes() -> None:
    """A checkpoint does not record image shapes, so rebuilding without them says what is missing."""
    _, train_cfg, ckpt = _image_student("gru")
    with pytest.raises(ValueError, match="obs_shapes"):
        rebuild_models(train_cfg, ckpt)


def _vmf_actor_checkpoint() -> tuple[nn.Module, dict, dict]:
    """Build a hyperspherical actor checkpoint as an off-policy run writes it."""
    from robot_rl.models import MLPModel
    from tests.conftest import make_obs

    obs = make_obs(4, 8)
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        6,
        hidden_dims=[32, 32],
        distribution_cfg={"class_name": "VonMisesFisherDistribution", "init_std": 0.05},
    ).eval()
    train_cfg = {
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "distribution_cfg": {"class_name": "VonMisesFisherDistribution", "init_std": 0.05},
        },
        "obs_groups": {"actor": ["policy"]},
        "algorithm": {},
    }
    return actor, train_cfg, {"actor_state_dict": actor.state_dict()}


def test_a_vmf_actor_rebuilds_with_its_unit_mean_direction() -> None:
    """The rebuilt actor acts with the unit mean direction the trained one does, not a Gaussian mean."""
    from tests.conftest import make_obs

    actor, train_cfg, ckpt = _vmf_actor_checkpoint()
    policy = rebuild_models(train_cfg, ckpt)["policy"]
    obs = make_obs(4, 8)
    with torch.inference_mode():
        expected = actor(obs, stochastic_output=False)
        actual = policy(obs, stochastic_output=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(4))


def test_all_models_says_so_for_a_checkpoint_without_a_single_critic() -> None:
    """An off-policy checkpoint has no single critic to rebuild, and asking for it names the limit."""
    _, train_cfg, ckpt = _vmf_actor_checkpoint()
    with pytest.raises(NotImplementedError, match="all_models"):
        rebuild_models(train_cfg, ckpt, all_models=True)


def test_rebuild_distillation_rejects_a_recurrent_student() -> None:
    """A recurrent 1D student raises instead of failing on a state-dict size mismatch."""
    student_cfg = {"class_name": "RNNModel", "hidden_dims": [32, 32], "rnn_hidden_dim": 16, "rnn_type": "gru"}
    train_cfg, ckpt, _, _ = _distillation_checkpoint(student_cfg)
    with pytest.raises(NotImplementedError, match="recurrent"):
        rebuild_models(train_cfg, ckpt)


def test_an_image_student_without_a_normalizer_takes_its_1d_shapes() -> None:
    """Without normalizer statistics the 1D input size comes from ``obs_shapes``, and its absence is named."""
    from tensordict import TensorDict

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(2, 7), "image": torch.zeros(2, 4, 12, 20)}, batch_size=[2])
    obs_groups = {"student": ["policy", "image"]}
    student_cfg = {
        "class_name": "CNNRNNModel",
        "hidden_dims": [32, 32],
        "obs_normalization": False,
        "cnn_cfg": {"output_channels": [8, 8], "kernel_size": 3, "stride": 2, "global_pool": "avg", "flatten": True},
        "rnn_hidden_dim": 16,
    }
    model = CNNRNNModel(obs, obs_groups, "student", 6, **{k: v for k, v in student_cfg.items() if k != "class_name"})
    train_cfg = {"student": student_cfg, "obs_groups": obs_groups, "algorithm": {}}
    ckpt = {"student_state_dict": model.eval().state_dict()}

    with pytest.raises(ValueError, match="1D group"):
        rebuild_models(train_cfg, ckpt, obs_shapes={"image": (4, 12, 20)})
    policy = rebuild_models(train_cfg, ckpt, obs_shapes={"image": (4, 12, 20), "policy": (7,)})["policy"]
    step = TensorDict({"policy": torch.randn(1, 7), "image": torch.rand(1, 4, 12, 20)}, batch_size=[1])
    model.reset()
    with torch.inference_mode():
        torch.testing.assert_close(policy(step, stochastic_output=False), model(step, stochastic_output=False))


def test_a_subclass_that_changes_the_features_does_not_export() -> None:
    """A model that overrides the feature or head-input path gets no export built from the plain one."""
    from robot_rl.models import CNNRNNModel
    from robot_rl.models.cref_model import CrefModel

    class Gated(CNNRNNModel):
        def _head_input(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
            return torch.cat([feats * 0.5, rnn_out], dim=-1)

    model, _, _ = _image_student("gru")
    model.__class__ = Gated
    for build in (model.as_jit, model.as_onnx):
        with pytest.raises(NotImplementedError, match="Gated"):
            build()
    assert CrefModel._features is not CNNRNNModel._features  # the reference-action student is one such model


def test_an_image_group_cannot_take_an_export_input_name() -> None:
    """ONNX inputs are named after the image groups, so a group called ``obs`` or ``h_in`` is refused."""
    model, _, _ = _image_student("gru")
    model.obs_groups_2d = ["h_in"]
    model.cnns = nn.ModuleDict({"h_in": model.cnns["image"]})
    with pytest.raises(ValueError, match="h_in"):
        model.as_onnx()


def test_rebuild_requires_the_distribution_class_name() -> None:
    """A logged distribution cfg must name its class; the export does not guess it."""
    student_cfg = {
        "class_name": "MLPModel",
        "hidden_dims": [32, 32],
        "distribution_cfg": {"class_name": "BetaDistribution"},
    }
    train_cfg, ckpt, _, _ = _distillation_checkpoint(student_cfg)
    del train_cfg["student"]["distribution_cfg"]["class_name"]
    with pytest.raises(ValueError, match="class_name"):
        rebuild_models(train_cfg, ckpt)


@pytest.mark.parametrize("distribution", ["TruncatedGaussian", "SquashedTanhGaussian"])
def test_fuse_model_reports_its_output_width(distribution: str) -> None:
    """A fused actor's action width reads from its ``trunk`` head, as the FB-CPR rebuild sizes it."""
    from robot_rl.models import ResidualFuseModel
    from robot_rl.utils.export import _num_actions
    from tests.conftest import make_obs

    num_actions, z_dim = 6, 8
    obs = make_obs(2, 10)
    model_cfg = {
        "hidden_dims": [32, 32],
        "embedding_dims": [16, 16],
        "distribution_cfg": dict(_STUDENT_DISTRIBUTIONS[distribution]),
    }
    actor = ResidualFuseModel(obs, {"actor": ["policy"]}, "actor", (z_dim, 0), num_actions, **copy.deepcopy(model_cfg))
    logged = {"class_name": "ResidualFuseModel", **model_cfg}
    assert _num_actions(logged, actor.state_dict()) == num_actions
