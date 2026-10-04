# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert-buffer priorities are replaced whole, in buffer order, and normalized."""

from __future__ import annotations

import torch

import pytest

from robot_rl.storage.trajectory_buffer import TrajectoryBuffer


def _buffer(num_motions: int) -> TrajectoryBuffer:
    buffer = TrajectoryBuffer.__new__(TrajectoryBuffer)
    buffer.device = "cpu"
    buffer.priorities = torch.ones(num_motions)
    return buffer


def test_set_priorities_normalizes_in_buffer_order() -> None:
    """Weights keep their buffer positions and sum to one."""
    buffer = _buffer(4)
    buffer.set_priorities(torch.tensor([1.0, 3.0, 0.0, 4.0]))
    assert torch.allclose(buffer.priorities, torch.tensor([0.125, 0.375, 0.0, 0.5]))


def test_set_priorities_rejects_wrong_shape() -> None:
    """A priority vector from a different corpus is refused."""
    with pytest.raises(ValueError):
        _buffer(4).set_priorities(torch.ones(5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_set_priorities_from_another_device() -> None:
    """Priorities computed on the GPU land in a CPU-resident buffer."""
    buffer = _buffer(2)
    buffer.set_priorities(torch.tensor([1.0, 3.0], device="cuda"))
    assert buffer.priorities.device.type == "cpu"
    assert torch.allclose(buffer.priorities, torch.tensor([0.25, 0.75]))
