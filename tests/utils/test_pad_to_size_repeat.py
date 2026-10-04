# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pad_to_size_repeat utility function."""

import torch

import pytest

from robot_rl.utils import pad_to_size_repeat


class TestPadToSizeRepeat:
    """Tests for padding a tensor by repeating its first slice."""

    def test_pads_by_repeating_first_row(self) -> None:
        """Padded rows should copy the first row rather than be zero-filled."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = pad_to_size_repeat(x, 4)
        assert out.shape == (4, 2)
        torch.testing.assert_close(out[:2], x)
        torch.testing.assert_close(out[2], x[0])
        torch.testing.assert_close(out[3], x[0])

    def test_returns_input_when_already_large_enough(self) -> None:
        """Padding to a size at or below the current one should be a no-op."""
        x = torch.zeros(5, 3)
        assert pad_to_size_repeat(x, 5) is x
        assert pad_to_size_repeat(x, 3) is x

    def test_padded_root_quaternion_stays_normalized(self) -> None:
        """The eval path pads root poses, where a zero quaternion is an invalid rotation."""
        root_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])
        out = pad_to_size_repeat(root_pose, 4)
        torch.testing.assert_close(out[:, 3:7].norm(dim=-1), torch.ones(4))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_pads_along_requested_dim(self, dim: int) -> None:
        """Only the requested dimension should grow."""
        x = torch.arange(6.0).reshape(2, 3)
        out = pad_to_size_repeat(x, 5, dim=dim)
        assert out.shape[dim] == 5
        assert out.shape[1 - dim] == x.shape[1 - dim]
