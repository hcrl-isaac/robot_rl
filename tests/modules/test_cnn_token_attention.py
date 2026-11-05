# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CNN token self-attention: starts as the identity, learns, and survives the CNN-RNN export."""

import torch

from robot_rl.modules import CNN
from robot_rl.modules.cnn import TokenSelfAttention


def _cnn(heads: int) -> CNN:
    torch.manual_seed(0)
    return CNN(
        input_dim=(54, 96),
        input_channels=4,
        output_channels=[16, 32, 64],
        kernel_size=3,
        stride=2,
        token_attention_heads=heads,
    )


def test_block_starts_as_identity_and_trains() -> None:
    """With the block, the CNN's output equals the plain CNN's until the block's weights move."""
    plain, attn = _cnn(0), _cnn(4)
    attn.load_state_dict({**attn.state_dict(), **plain.state_dict()}, strict=False)
    x = torch.randn(3, 4, 54, 96)
    assert torch.allclose(plain(x), attn(x), atol=1e-6)
    assert plain.output_dim == attn.output_dim
    block = next(m for m in attn if isinstance(m, TokenSelfAttention))
    attn(x).pow(2).sum().backward()
    assert block.attn.out_proj.weight.grad.abs().sum() > 0


def test_cnn_rnn_with_block_exports() -> None:
    """A CNN-RNN model whose CNN carries the block still exports to ONNX and matches its own forward."""
    from tensordict import TensorDict

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {"output_channels": [16, 32, 64], "kernel_size": 3, "stride": 2, "token_attention_heads": 4}
    model = CNNRNNModel(
        obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
    )
    onnx_model = model.as_onnx()
    inputs = onnx_model.get_dummy_inputs()
    out = onnx_model(*inputs)
    assert out[0].shape == (1, 3)
