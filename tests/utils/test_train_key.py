# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the logger's Train/ grouping of training-internal scalars."""

import torch

from robot_rl.utils.logger import Logger, train_key


class _Writer:
    """Records the scalar names the logger writes."""

    def __init__(self) -> None:
        self.tags: list[str] = []

    def add_scalar(self, tag: str, value: float, it: int) -> None:
        """Record ``tag``."""
        self.tags.append(tag)


def _logger() -> Logger:
    """Build a logger writing to :class:`_Writer`, without a run."""
    logger = Logger.__new__(Logger)
    logger.writer = _Writer()
    logger.device = "cpu"
    return logger


def test_training_sections_move_under_train() -> None:
    """Training sections move under Train/, lowercased; Train/ names and bare names are kept."""
    assert train_key("Actor/grad_norm") == "Train/actor/grad_norm"
    assert train_key("Critic_Loss/td") == "Train/critic_loss/td"
    assert train_key("Buffer/size") == "Train/buffer/size"
    assert train_key("Loss/value_function") == "Train/loss/value_function"
    assert train_key("Train/mean_reward") == "Train/mean_reward"
    assert train_key("bare") == "bare"


def test_only_training_extras_are_renamed() -> None:
    """Algorithm and loss extras move under Train/; episode extras keep their sections."""
    logger = _logger()
    logger._log_extras([{"Actor/logp": torch.tensor(1.0), "kl": 0.1}], 0, 40, "Train", "Train/{}", train=True)
    logger._log_extras([{"critic": 0.5}], 0, 40, "Loss", "Mean {}", override_console_format=False, train=True)
    logger._log_extras([{"Metrics/success": 1.0, "Episode_Reward/track": 0.2}], 0, 40, "Episode", "Mean episode {}")
    assert sorted(logger.writer.tags) == sorted([
        "Train/actor/logp",
        "Train/kl",
        "Train/loss/critic",
        "Metrics/success",
        "Episode_Reward/track",
    ])
