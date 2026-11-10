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


def test_attention_pool_shape_and_gradients() -> None:
    """Attention pooling returns queries x channels, and the loss reaches the convs, queries and null token."""
    import pytest

    torch.manual_seed(0)
    cnn = CNN(
        input_dim=(54, 96),
        input_channels=4,
        output_channels=[16, 32, 64],
        kernel_size=3,
        stride=2,
        token_attention_heads=4,
        attention_pool_queries=4,
        attention_pool_heads=4,
    )
    out = cnn(torch.randn(3, 4, 54, 96))
    assert cnn.output_dim == 256 and out.shape == (3, 256)
    out.pow(2).sum().backward()
    pool = cnn[-1]
    assert cnn[0].weight.grad.abs().sum() > 0
    assert pool.queries.grad.abs().sum() > 0 and pool.null.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        CNN(
            input_dim=(54, 96),
            input_channels=4,
            output_channels=[16],
            kernel_size=3,
            attention_pool_queries=4,
            global_pool="avg",
        )


def test_cnn_rnn_with_attention_pool_exports() -> None:
    """A CNN-RNN model whose CNN ends in attention pooling exports to ONNX."""
    from tensordict import TensorDict

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {
        "output_channels": [16, 32, 64],
        "kernel_size": 3,
        "stride": 2,
        "token_attention_heads": 4,
        "attention_pool_queries": 4,
    }
    model = CNNRNNModel(
        obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
    )
    onnx_model = model.as_onnx()
    out = onnx_model(*onnx_model.get_dummy_inputs())
    assert out[0].shape == (1, 3)


def test_channels_last_weights_match_and_load_into_a_plain_cnn() -> None:
    """NHWC weights (the SAC update's layout on CUDA) give the NCHW outputs and gradients, and checkpoint plainly."""
    torch.manual_seed(1)
    cnn = CNN((54, 96), 4, [16, 32, 64], 3, stride=[1, 2, 2], global_pool="none", attention_pool_queries=4)
    nhwc = CNN((54, 96), 4, [16, 32, 64], 3, stride=[1, 2, 2], global_pool="none", attention_pool_queries=4)
    nhwc.load_state_dict(cnn.state_dict())
    nhwc.to(memory_format=torch.channels_last)
    assert nhwc[0].weight.is_contiguous(memory_format=torch.channels_last)
    x = torch.rand(6, 4, 54, 96)
    out, out_nhwc = cnn(x), nhwc(x)
    torch.testing.assert_close(out_nhwc, out, rtol=1e-5, atol=1e-5)
    out.square().sum().backward()
    out_nhwc.square().sum().backward()
    torch.testing.assert_close(nhwc[0].weight.grad, cnn[0].weight.grad, rtol=1e-4, atol=1e-5)
    plain = CNN((54, 96), 4, [16, 32, 64], 3, stride=[1, 2, 2], global_pool="none", attention_pool_queries=4)
    plain.load_state_dict(nhwc.state_dict())
    assert plain[0].weight.is_contiguous()
    torch.testing.assert_close(plain(x), out_nhwc, rtol=1e-5, atol=1e-5)
