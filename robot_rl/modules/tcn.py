# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence

from robot_rl.utils import resolve_nn_activation


class TCNEncoder(nn.Module):
    """Temporal convolutional encoder over a fixed-length history of per-step feature vectors.

    Consumes a time-major flattened history ``[batch, history_length * step_dim]`` (oldest step first): each
    step is embedded with a shared linear layer, a 1D conv stack slides over the time axis, and the flattened
    result is projected to ``encoding_dim``.
    """

    def __init__(
        self,
        step_dim: int,
        history_length: int,
        step_embed_dim: int = 30,
        channels: Sequence[int] = (20, 10, 10),
        kernel_sizes: Sequence[int] = (8, 5, 5),
        strides: Sequence[int] = (4, 1, 1),
        encoding_dim: int = 64,
        activation: str = "elu",
    ) -> None:
        """Initialize the encoder.

        Args:
            step_dim: Feature dimension of a single history step.
            history_length: Number of history steps in the input.
            step_embed_dim: Output dimension of the shared per-step embedding layer.
            channels: Output channels of each conv layer.
            kernel_sizes: Kernel size of each conv layer (over the time axis).
            strides: Stride of each conv layer.
            encoding_dim: Dimension of the final encoding.
            activation: Activation function used after every layer.
        """
        super().__init__()
        self.step_dim = step_dim
        self.history_length = history_length
        act = resolve_nn_activation(activation)
        self.step_embed = nn.Sequential(nn.Linear(step_dim, step_embed_dim), act)
        conv_layers: list[nn.Module] = []
        in_channels = step_embed_dim
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides, strict=True):
            conv_layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride), act]
            in_channels = out_channels
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, step_embed_dim, history_length)).shape[-1]
        self.head = nn.Sequential(nn.Linear(flat_dim, encoding_dim), act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a flattened history ``[batch, history_length * step_dim]`` into ``[batch, encoding_dim]``."""
        x = x.view(-1, self.history_length, self.step_dim)
        x = self.step_embed(x)
        x = self.conv(x.permute(0, 2, 1))
        return self.head(x)
