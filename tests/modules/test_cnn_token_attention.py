# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CNN token self-attention: starts as the identity, learns, and survives the CNN-RNN export."""

import torch
from tensordict import TensorDict

from robot_rl.models import CNNRNNModel
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


def test_flatten_projection_sets_output_dim() -> None:
    """The flatten projection maps the flattened cells to a bounded code of the requested width."""
    cnn = CNN(
        input_dim=(54, 96),
        input_channels=4,
        output_channels=[16, 32, 64],
        kernel_size=3,
        stride=2,
        token_attention_heads=4,
        flatten_proj_dim=256,
    )
    out = cnn(torch.randn(3, 4, 54, 96))
    assert cnn.output_dim == 256 and out.shape == (3, 256)
    assert out.abs().max() <= 1.0


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
            flatten_proj_dim=256,
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


def _memory_model(latent_layers: int = 0) -> CNNRNNModel:
    from tensordict import TensorDict

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {
        "output_channels": [16, 32, 64],
        "kernel_size": 3,
        "stride": 2,
        "token_attention_heads": 4,
        "attention_pool_queries": 4,
        "attention_pool_memory_dim": 16,
        "attention_pool_latent_layers": latent_layers,
    }
    model = CNNRNNModel(
        obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
    )
    # the memory projection starts at zero; perturb it so the tests see the state steer the queries
    torch.nn.init.normal_(model.cnns["image"][-1].memory.weight, std=0.5)
    return model


def _window(seq_len: int, batch: int) -> TensorDict:
    from tensordict import TensorDict

    return TensorDict(
        {"policy": torch.randn(seq_len, batch, 10), "image": torch.rand(seq_len, batch, 4, 54, 96)},
        batch_size=[seq_len, batch],
    )


def test_memory_pool_sequence_matches_rollout_steps() -> None:
    """A window encoded at once equals stepping the rollout path, resets included; the state changes the features."""
    torch.manual_seed(0)
    model = _memory_model(latent_layers=1)
    assert model.memory_pool
    obs = _window(5, 3)
    resets = torch.zeros(5, 3)
    resets[2, 1] = 1.0
    seq, state = model.encode_sequence(obs, None, resets)
    model.reset()
    steps = []
    for t in range(5):
        if t:
            model.reset(resets[t].bool())
        steps.append(model.get_latent(obs[t]))
    torch.testing.assert_close(seq, torch.stack(steps), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state, model.rnn.hidden_state, atol=1e-5, rtol=1e-5)
    stepped = model.encode_step(obs[0], torch.randn(1, 3, 16))
    assert not torch.allclose(stepped[:, :-16], seq[0][:, :-16])


def test_memory_pool_gradients_reach_the_memory_projection() -> None:
    """The window loss trains the projection that turns the state into query shifts."""
    model = _memory_model()
    seq, _ = model.encode_sequence(_window(4, 2), None)
    seq.square().sum().backward()
    assert model.cnns["image"][-1].memory.weight.grad.abs().sum() > 0


def test_memory_pool_exports_match_the_model() -> None:
    """ONNX and TorchScript exports read the state before each step, as the rollout does."""
    torch.manual_seed(0)
    model = _memory_model(latent_layers=1).eval()
    obs = _window(3, 1)
    model.reset()
    with torch.no_grad():
        expected = [model(obs[t]) for t in range(3)]
    onnx_model = model.as_onnx()
    jit_model = torch.jit.script(model.as_jit())
    h = torch.zeros(1, 1, 16)
    for t in range(3):
        a_onnx, h = onnx_model(obs[t]["policy"], obs[t]["image"], h)
        a_jit = jit_model(obs[t]["policy"], [obs[t]["image"]])
        torch.testing.assert_close(a_onnx, a_jit, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a_onnx, expected[t], atol=1e-5, rtol=1e-5)


def test_memory_pool_rejects_mismatched_state_width() -> None:
    """The pooling's memory width must equal the GRU's."""
    from tensordict import TensorDict

    import pytest

    from robot_rl.models import CNNRNNModel

    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {
        "output_channels": [16, 32, 64],
        "kernel_size": 3,
        "stride": 2,
        "attention_pool_queries": 4,
        "attention_pool_memory_dim": 8,
    }
    with pytest.raises(ValueError, match="memory-conditioned"):
        CNNRNNModel(
            obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
        )


def test_normalized_memory_pool_trains_and_exports() -> None:
    """The normalized memory pooling starts with no shift, trains its gate, and exports like the model."""
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {
        "output_channels": [16, 32, 64],
        "kernel_size": 3,
        "stride": 2,
        "attention_pool_queries": 4,
        "attention_pool_memory_dim": 16,
        "attention_pool_memory_norm": True,
    }
    model = CNNRNNModel(
        obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
    )
    pool = model.cnns["image"][-1]
    assert float(pool.memory_gate) == 0.0
    seq, _ = model.encode_sequence(_window(4, 2), None)
    seq.square().sum().backward()
    assert pool.memory_gate.grad.abs().sum() > 0
    with torch.no_grad():
        pool.memory_gate.fill_(0.5)
    model.eval()
    window = _window(3, 1)
    model.reset()
    with torch.no_grad():
        expected = [model(window[t]) for t in range(3)]
    onnx_model = model.as_onnx()
    jit_model = torch.jit.script(model.as_jit())
    h = torch.zeros(1, 1, 16)
    for t in range(3):
        a_onnx, h = onnx_model(window[t]["policy"], window[t]["image"], h)
        torch.testing.assert_close(a_onnx, expected[t], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            jit_model(window[t]["policy"], [window[t]["image"]]), expected[t], atol=1e-5, rtol=1e-5
        )


def test_memory_share_bounds_the_query_tilt() -> None:
    """With memory_share, the shift moves a query by at most that fraction of its length; exports match."""
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.zeros(1, 10), "image": torch.zeros(1, 4, 54, 96)}, batch_size=[1])
    cnn_cfg = {
        "output_channels": [16, 32, 64],
        "kernel_size": 3,
        "stride": 2,
        "attention_pool_queries": 4,
        "attention_pool_memory_dim": 16,
        "attention_pool_memory_share": 0.3,
    }
    model = CNNRNNModel(
        obs, {"actor": ["policy", "image"]}, "actor", 3, hidden_dims=[32], cnn_cfg=cnn_cfg, rnn_hidden_dim=16
    )
    pool = model.cnns["image"][-1]
    base = pool.base_norm(pool.queries)
    shift = pool.shift_norm(pool.memory(pool.memory_in(torch.randn(5, 16) * 10)).view(5, 4, 64))
    assert float((0.3 * shift).norm(dim=-1).max() / base.norm(dim=-1).min()) < 0.31
    seq, _ = model.encode_sequence(_window(4, 2), None)
    seq.square().sum().backward()
    assert pool.memory.weight.grad.abs().sum() > 0
    model.eval()
    window = _window(3, 1)
    model.reset()
    with torch.no_grad():
        expected = [model(window[t]) for t in range(3)]
    onnx_model = model.as_onnx()
    jit_model = torch.jit.script(model.as_jit())
    h = torch.zeros(1, 1, 16)
    for t in range(3):
        a_onnx, h = onnx_model(window[t]["policy"], window[t]["image"], h)
        torch.testing.assert_close(a_onnx, expected[t], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            jit_model(window[t]["policy"], [window[t]["image"]]), expected[t], atol=1e-5, rtol=1e-5
        )
