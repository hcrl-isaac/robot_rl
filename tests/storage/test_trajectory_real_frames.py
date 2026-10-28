# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert windows and reset states come only from a padded bundle's real frames."""

from __future__ import annotations

import pathlib
import torch
from tensordict import TensorDict

from robot_rl.storage.trajectory_buffer import TrajectoryBuffer


def _bundle(path: pathlib.Path, lengths: list[int], bucket: int, with_length: bool = True) -> str:
    frame = torch.arange(bucket, dtype=torch.float32).expand(len(lengths), bucket)
    data = {"expert": frame[..., None].expand(-1, -1, 15).clone()}
    if with_length:
        data["length"] = torch.tensor(lengths)[:, None].expand(-1, bucket).clone()
    torch.save(TensorDict(data, batch_size=[len(lengths), bucket]), path)
    return str(path)


def test_windows_stay_in_real_frames(tmp_path: pathlib.Path) -> None:
    """Every sampled frame and its successor precede the row's padding."""
    buffer = TrajectoryBuffer(_bundle(tmp_path / "b.pt", [12, 40, 100], 100), ["expert"])
    cur, nxt = buffer.sample(4096, seq_length=8)
    rows = cur["length"]
    assert (nxt["expert"][:, 0] < rows).all()
    assert (cur["expert"][:, 0] + 1 == nxt["expert"][:, 0]).all()
    assert nxt["expert"][:, 0].max() > 90  # full rows still reach their end


def test_reset_states_stay_in_real_frames(tmp_path: pathlib.Path) -> None:
    """Reset states are drawn from real frames only."""
    buffer = TrajectoryBuffer(_bundle(tmp_path / "b.pt", [5, 30], 50), ["expert"])
    state = buffer.sample_states(2048)
    assert state["root_pose"][:, 0].max() < 30


def test_bundle_without_length_uses_whole_rows(tmp_path: pathlib.Path) -> None:
    """A bundle without ``length`` samples the full row as before."""
    buffer = TrajectoryBuffer(_bundle(tmp_path / "b.pt", [10, 10], 60, with_length=False), ["expert"])
    _, nxt = buffer.sample(4096, seq_length=4)
    assert nxt["expert"][:, 0].max() > 50
