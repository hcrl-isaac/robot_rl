# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where FB-CPR places the expert motions."""

from __future__ import annotations

import pytest

from robot_rl.algorithms.fb_cpr import _expert_storage_device


def _cfg(storage_device: str | None, **algorithm: str | None) -> dict:
    return {"storage_device": storage_device, "algorithm": algorithm}


@pytest.mark.parametrize("algorithm", [{}, {"expert_storage_device": None}])
@pytest.mark.parametrize("storage_device", ["cpu", "cuda:1"])
def test_an_unset_expert_device_follows_the_storage_device(storage_device: str, algorithm: dict) -> None:
    """Without ``expert_storage_device`` the motions sit where the replay buffer does."""
    assert _expert_storage_device(_cfg(storage_device, **algorithm), "cuda:0") == storage_device


def test_an_explicit_expert_device_overrides_the_storage_device() -> None:
    """``expert_storage_device`` places the motions whatever ``storage_device`` says."""
    assert _expert_storage_device(_cfg("cuda", expert_storage_device="cpu"), "cuda:0") == "cpu"
    assert _expert_storage_device(_cfg("cpu", expert_storage_device="cuda:2"), "cuda:0") == "cuda:2"


@pytest.mark.parametrize("cfg", [_cfg("cpu", expert_storage_device="cuda"), _cfg("cuda")])
def test_a_bare_cuda_names_the_runner_gpu(cfg: dict) -> None:
    """A bare ``"cuda"``, given or inherited, is the runner's GPU and not torch's current device."""
    assert _expert_storage_device(cfg, "cuda:3") == "cuda:3"
    assert _expert_storage_device(cfg, "cpu") == "cuda"
