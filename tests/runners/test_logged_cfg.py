# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building an algorithm leaves the runner cfg intact, so the logged train cfg is the cfg the run was built from."""

from __future__ import annotations

import copy
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.algorithms.fb_cpr import FbCpr
from robot_rl.runners import DistillationRunner, OffPolicyRunner, OnPolicyRunner
from robot_rl.utils.log_writer import LogWriter
from tests.algorithms.test_fb_cpr import _Env as FbCprEnv
from tests.algorithms.test_fb_cpr import _train_cfg as fbcpr_cfg
from tests.runners.test_off_policy_runner_sac import DummyEnv as SacEnv
from tests.runners.test_off_policy_runner_sac import _make_cfg as sac_cfg
from tests.runners.test_on_policy_runner import DummyEnv as PpoEnv
from tests.runners.test_on_policy_runner import _make_train_cfg as ppo_cfg


class CaptureWriter(LogWriter):
    """Keeps the train cfg handed to :meth:`store_config`."""

    def __init__(self, log_dir: str) -> None:  # noqa: D107
        self.train_cfg: dict | None = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D102
        pass

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:  # noqa: D102
        self.train_cfg = copy.deepcopy(train_cfg)


def _distillation_cfg() -> dict:
    model = {"class_name": "MLPModel", "hidden_dims": [32], "distribution_cfg": {"class_name": "GaussianDistribution"}}
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"student": ["policy"], "teacher": ["policy"]},
        "student": copy.deepcopy(model),
        "teacher": copy.deepcopy(model),
        "algorithm": {"class_name": "Distillation", "num_learning_epochs": 1, "gradient_length": 2},
    }


def _class_names(cfg: object, path: str = "") -> set[str]:
    """Paths of every dict in ``cfg`` that holds a ``class_name``."""
    if not isinstance(cfg, dict):
        return set()
    found = {path} if "class_name" in cfg else set()
    for key, value in cfg.items():
        found |= _class_names(value, f"{path}.{key}")
    return found


RUNNERS = {
    "ppo": (OnPolicyRunner, PpoEnv, lambda: ppo_cfg("mlp")),
    "ppo_rnn": (OnPolicyRunner, PpoEnv, lambda: ppo_cfg("rnn")),
    "sac": (OffPolicyRunner, SacEnv, sac_cfg),
    "distillation": (DistillationRunner, PpoEnv, _distillation_cfg),
}


@pytest.mark.parametrize("name", list(RUNNERS))
def test_logged_train_cfg_is_the_cfg_the_run_was_built_from(name: str, tmp_path: Path) -> None:
    """``store_config`` receives the given cfg, every ``class_name`` included."""
    runner_class, env_class, make_cfg = RUNNERS[name]
    cfg = make_cfg()
    cfg["logger"] = {"class_name": CaptureWriter}
    given = copy.deepcopy(cfg)
    runner = runner_class(env_class(), cfg, log_dir=str(tmp_path), device="cpu")
    runner.logger.init_logging_writer()
    logged = runner.logger.writer.train_cfg
    assert {".algorithm", ".logger"} <= _class_names(given)
    assert _class_names(logged) == _class_names(given)
    # the runner adds only its resolved multi-GPU settings
    assert {k: v for k, v in logged.items() if k != "multi_gpu"} == given


def test_fbcpr_build_keeps_the_cfg() -> None:
    """FB-CPR builds its six models without consuming their cfgs."""
    cfg = fbcpr_cfg("TruncatedGaussianDistribution")
    given = copy.deepcopy(cfg)
    obs = TensorDict({"policy": torch.zeros(FbCprEnv.num_envs, 8)}, batch_size=[FbCprEnv.num_envs])
    FbCpr.construct_algorithm(obs, FbCprEnv(), cfg, "cpu", inference=True)
    assert cfg == given
    assert ".actor.distribution_cfg" in _class_names(cfg)
