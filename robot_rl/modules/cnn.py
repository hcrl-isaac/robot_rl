# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
from torch import nn as nn

from robot_rl.utils import get_param, resolve_nn_activation


class CNN(nn.Sequential):
    """Convolutional Neural Network.

    The CNN network is a sequence of convolutional layers, optional normalization layers, optional activation functions,
    and optional pooling. The final output can be flattened.
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, ...] | list[int],
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
        token_attention_heads: int = 0,
        flatten_proj_dim: int = 0,
        attention_pool_queries: int = 0,
        attention_pool_heads: int = 4,
        attention_pool_memory_dim: int = 0,
        attention_pool_latent_layers: int = 0,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_dim: Height and width of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each convolutional layer or a single kernel size for all layers.
            stride: List of strides for each convolutional layer or a single stride for all layers.
            dilation: List of dilations for each convolutional layer or a single dilation for all layers.
            padding: Padding type to use. Either 'none', 'zeros', 'reflect', 'replicate', or 'circular'.
            norm: List of normalization types for each convolutional layer or a single type for all layers. Either
                'none', 'batch', or 'layer'.
            activation: Activation function to use.
            max_pool: List of booleans indicating whether to apply max pooling after each convolutional layer or a
                single boolean for all layers.
            global_pool: Global pooling type to apply at the end. Either 'none', 'max', or 'avg'.
            flatten: Whether to flatten the output tensor.
            token_attention_heads: Heads of a residual self-attention block over the last feature map's cells
                (position-tagged, starting as the identity), before pooling; 0 adds none.
            flatten_proj_dim: Width of a linear projection of the flattened output, followed by LayerNorm and
                tanh; 0 adds none. Requires ``flatten``.
            attention_pool_queries: Learned queries of an attention pooling of the last feature map's cells (plus a
                null token), replacing global pooling and flattening; 0 adds none. Output width is queries x channels.
            attention_pool_heads: Attention heads of that pooling.
            attention_pool_memory_dim: Width of the recurrent state that shifts the pooling's queries, passed to
                :meth:`forward` as ``cond``; 0 keeps the queries fixed.
            attention_pool_latent_layers: Self-attention blocks over the pooled queries, after the pooling.
        """
        super().__init__()

        # Resolve activation function
        activation_function = resolve_nn_activation(activation)

        # Create layers sequentially
        layers = []
        last_channels = input_channels
        last_dim = input_dim
        for idx in range(len(output_channels)):
            # Get parameters for the current layer
            k = get_param(kernel_size, idx)
            s = get_param(stride, idx)
            d = get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else (0, 0)
            )

            # Append convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            )

            # Append normalization layer if specified
            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            # Append activation function
            layers.append(activation_function)

            # Apply max pooling if specified
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            # Update last channels and dimensions
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        if token_attention_heads:
            layers.append(TokenSelfAttention(last_channels, last_dim[0] * last_dim[1], token_attention_heads))

        if not attention_pool_queries and (attention_pool_memory_dim or attention_pool_latent_layers):
            raise ValueError("attention_pool_memory_dim and attention_pool_latent_layers need attention_pool_queries.")
        if attention_pool_queries:
            if global_pool != "none" or flatten_proj_dim:
                raise ValueError("attention_pool_queries replaces global_pool and flatten_proj_dim.")
            layers.append(
                AttentionPool(
                    last_channels,
                    last_dim[0] * last_dim[1],
                    attention_pool_queries,
                    attention_pool_heads,
                    memory_dim=attention_pool_memory_dim,
                    latent_layers=attention_pool_latent_layers,
                )
            )
            self.memory_dim = attention_pool_memory_dim
            self._output_channels = None
            self._output_dim = attention_pool_queries * last_channels
            for idx, layer in enumerate(layers):
                self.add_module(f"{idx}", layer)
            return

        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        # Apply flattening if specified
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # Store final output dimension
        self.memory_dim = 0
        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1]

        if flatten_proj_dim:
            if not flatten:
                raise ValueError("flatten_proj_dim requires flatten=True.")
            # a compact, normalized image code instead of every raw cell feeding the next network
            layers += [nn.Linear(self._output_dim, flatten_proj_dim), nn.LayerNorm(flatten_proj_dim), nn.Tanh()]
            self._output_dim = flatten_proj_dim

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """Get the output height and width or total output dimension if output is flattened."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize the weights of the CNN with Kaiming initialization."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None, pool: bool = True) -> torch.Tensor:
        """Forward pass of the CNN.

        Args:
            x: Images ``(N, C, H, W)``.
            cond: Recurrent state ``(N, memory_dim)`` for a memory-conditioned attention pooling.
            pool: False stops before an attention pooling and returns its feature map.
        """
        for layer in self:
            # hasattr resolves per layer when TorchScript compiles, which isinstance on a module class does not
            if hasattr(layer, "num_queries"):
                if pool:
                    x = layer(x, cond)
            else:
                x = layer(x)
        return x

    def attention_pool(self, fmap: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the attention pooling alone to a feature map from ``forward(x, pool=False)``."""
        return self[len(self) - 1](fmap, cond)


def _compute_padding(input_hw: tuple[int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int]:
    """Compute the optimal padding for the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.ceil((stride * math.floor(input_hw[0] / stride) - input_hw[0] - stride + dilation * (kernel - 1) + 1) / 2)
    w = math.ceil((stride * math.floor(input_hw[1] / stride) - input_hw[1] - stride + dilation * (kernel - 1) + 1) / 2)
    return (h, w)


def _compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int],
    is_max_pool: bool = False,
) -> tuple[int, int]:
    """Compute the output height and width of the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1)

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)

    return (h, w)


class TokenSelfAttention(nn.Module):
    """Pre-LN residual self-attention and feed-forward over a feature map's cells, shape-preserving.

    Each cell is a token tagged with a learned position embedding; the output projections start at zero, so the
    block starts as the identity and only adds what the cells learn to exchange.
    """

    def __init__(self, channels: int, cells: int, heads: int) -> None:
        """Initialize the block.

        Args:
            channels: Feature channels per cell (the token width; divisible by ``heads``).
            cells: Number of cells in the feature map.
            heads: Attention heads.
        """
        super().__init__()
        self.pos = nn.Parameter(0.02 * torch.randn(cells, channels))
        self.norm_attn = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(channels, 2 * channels), nn.ELU(), nn.Linear(2 * channels, channels))
        for layer in (self.attn.out_proj, self.ff[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        """Mix information across cells of an ``(N, C, H, W)`` map; returns the same shape."""
        n, c, h, w = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)
        y = self.norm_attn(x + self.pos)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.ff(self.norm_ff(x))
        return x.transpose(1, 2).reshape(n, c, h, w)


class AttentionPool(nn.Module):
    """Learned queries attend over a feature map's cells and a null token; outputs concatenated and LayerNormed.

    Each cell carries a learned position embedding into its key and value, so a query's output says where its
    attention landed as well as what is there; the null token, with no position, lets a query report nothing in view.
    A recurrent state can shift the queries (starting at no shift), so memory steers where they look, and
    self-attention blocks over the pooled queries let what they found combine before the output.
    """

    def __init__(
        self, channels: int, cells: int, queries: int, heads: int, memory_dim: int = 0, latent_layers: int = 0
    ) -> None:
        """Initialize the pooling.

        Args:
            channels: Feature channels per cell (the token width; divisible by ``heads``).
            cells: Number of cells in the feature map.
            queries: Learned queries, each returning one ``channels``-wide vector.
            heads: Attention heads.
            memory_dim: Width of the recurrent state shifting the queries; 0 keeps them fixed.
            latent_layers: Self-attention blocks over the pooled queries.
        """
        super().__init__()
        self.num_queries = queries
        self.memory_dim = memory_dim
        self.pos = nn.Parameter(0.02 * torch.randn(cells, channels))
        self.queries = nn.Parameter(0.02 * torch.randn(queries, channels))
        self.null = nn.Parameter(0.02 * torch.randn(1, channels))
        self.norm_in = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.memory = nn.Linear(memory_dim, queries * channels) if memory_dim else None
        if self.memory is not None:
            nn.init.zeros_(self.memory.weight)
            nn.init.zeros_(self.memory.bias)
        self.latent = nn.ModuleList([TokenSelfAttention(channels, queries, heads) for _ in range(latent_layers)])
        self.norm_out = nn.LayerNorm(queries * channels)

    def forward(self, fmap: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Pool an ``(N, C, H, W)`` map into ``(N, queries * C)``; ``cond`` ``(N, memory_dim)`` shifts the queries."""
        n, c = fmap.shape[0], fmap.shape[1]
        cells = self.norm_in(fmap.flatten(2).transpose(1, 2)) + self.pos
        keys = torch.cat([cells, self.null.expand(n, -1, -1)], dim=1)
        queries = self.queries.expand(n, -1, -1)
        if self.memory is not None:
            if cond is None:
                raise ValueError("A memory-conditioned attention pooling needs the recurrent state as cond.")
            queries = queries + self.memory(cond).view(n, self.num_queries, c)
        pooled = self.attn(queries, keys, keys, need_weights=False)[0]
        for block in self.latent:
            pooled = block(pooled.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        return self.norm_out(pooled.flatten(1))
