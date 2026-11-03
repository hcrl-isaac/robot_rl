# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode logs are averaged over the episodes that ended, not over env steps."""

import torch

from robot_rl.utils.logger import Logger


def test_episode_log_skips_stale_steps_and_weights_by_resets() -> None:
    """A step that resets nothing repeats the env's previous log dict; it must not count again."""
    logger = Logger(None, {"algorithm": {}}, {}, 4, False, 1, 0, "cpu")
    logger.writer = object()  # process_env_step buffers only with a writer
    rewards = torch.zeros(4)
    three_done = {"log": {"Metrics/success": 1.0}}
    logger.process_env_step(rewards, torch.tensor([1, 1, 1, 0]), three_done)
    logger.process_env_step(rewards, torch.zeros(4), three_done)  # stale: no env reset
    logger.process_env_step(rewards, torch.tensor([0, 0, 0, 1]), {"log": {"Metrics/success": torch.tensor(0.0)}})
    values = torch.cat([info["Metrics/success"] for info in logger.ep_extras])
    assert values.mean().item() == 0.75
