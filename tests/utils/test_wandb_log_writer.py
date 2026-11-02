# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The W&B writer's run settings are accepted by the installed wandb."""

import pytest

wandb = pytest.importorskip("wandb")

from robot_rl.utils.wandb_log_writer import run_settings  # noqa: E402


@pytest.mark.parametrize("shared", [False, True])
def test_run_settings_build(shared: bool) -> None:
    """A wandb release that rejects a setting the writer passes fails here, before a run's init does."""
    settings = run_settings(shared)
    assert isinstance(settings, wandb.Settings)
    if shared:
        assert (settings.mode, settings.x_label, settings.x_primary) == ("shared", "main", True)
