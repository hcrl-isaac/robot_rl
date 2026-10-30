# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CNN activation checkpointing changes memory, not results."""

import torch

from robot_rl.modules.cnn import CNN


def test_activation_checkpointing_matches_the_plain_forward_and_gradients() -> None:
    """A checkpointed CNN gives the same outputs and parameter gradients as the plain one."""
    kw = dict(input_dim=(28, 48), input_channels=4, output_channels=[8, 16], kernel_size=3, stride=2, activation="elu")
    plain, ckpt = CNN(**kw), CNN(**kw, activation_checkpointing=True)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(6, 4, 28, 48)
    out_plain, out_ckpt = plain(x), ckpt(x)
    torch.testing.assert_close(out_ckpt, out_plain)
    out_plain.square().mean().backward()
    out_ckpt.square().mean().backward()
    for p, q in zip(plain.parameters(), ckpt.parameters(), strict=True):
        torch.testing.assert_close(q.grad, p.grad)
