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
    """Training sections move under Train/, lowercased; Train/, Loss/ and Policy/ names and bare names are kept."""
    assert train_key("Actor/grad_norm") == "Train/actor/grad_norm"
    assert train_key("Critic_Loss/td") == "Train/critic_loss/td"
    assert train_key("Buffer/size") == "Train/buffer/size"
    assert train_key("Loss/value_function") == "Loss/value_function"
    assert train_key("Style/a_expert") == "Train/style/a_expert"
    assert train_key("Policy/mean_std") == "Policy/mean_std"
    assert train_key("Train/mean_reward") == "Train/mean_reward"
    assert train_key("bare") == "bare"


def test_only_training_extras_are_renamed() -> None:
    """Algorithm extras move under Train/, losses under Loss/; episode extras keep their sections."""
    logger = _logger()
    logger._log_extras([{"Actor/logp": torch.tensor(1.0), "kl": 0.1}], 0, 40, "Train", "Train/{}", train=True)
    losses = [{"critic": 0.5, "Loss/actor": 0.2, "Train/alpha": 0.1}]
    logger._log_extras(losses, 0, 40, "Loss", "Mean {}", override_console_format=False, train=True)
    logger._log_extras([{"Metrics/success": 1.0, "Episode_Reward/track": 0.2}], 0, 40, "Episode", "Mean episode {}")
    assert sorted(logger.writer.tags) == sorted([
        "Train/actor/logp",
        "Train/kl",
        "Loss/critic",
        "Loss/actor",
        "Train/alpha",
        "Metrics/success",
        "Episode_Reward/track",
    ])


def test_on_policy_losses_and_learning_rate_sections() -> None:
    """On-policy loss dicts keep Loss/ and Train/ apart, and the learning rate is a Train/ scalar."""
    logger = _logger()
    logger.cfg = {"num_steps_per_env": 1, "algorithm": {}}
    logger.num_envs, logger.gpu_world_size, logger.tot_timesteps, logger.tot_time = 1, 1, 0, 0.0
    logger.logger_type = "tensorboard"
    logger.ep_extras, logger.eval_extras, logger.loss_extras, logger.algo_extras = [], [], [], []
    logger.rewbuffer = []
    loss_dict = {"Loss/value": 1.0, "surrogate": 0.5, "Train/entropy": 0.3, "Style/a_expert": 0.2}
    logger.log(0, 0, 1, 0.1, 0.1, loss_dict=loss_dict, learning_rate=1e-3)
    assert {
        "Loss/value",
        "Loss/surrogate",
        "Train/entropy",
        "Train/style/a_expert",
        "Train/learning_rate",
    } <= set(logger.writer.tags)
    assert not any(tag.startswith("Train/loss") for tag in logger.writer.tags)
