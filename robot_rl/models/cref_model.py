# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.models.cnn_rnn_model import CNNRNNModel
from robot_rl.models.mlp_model import MLPModel
from robot_rl.modules import CNN
from robot_rl.modules.rnn import RNN
from robot_rl.utils import resolve_nn_activation


class CrefModel(CNNRNNModel):
    """CReF actor: proprioception-queried cross-attention over depth tokens, gated fusion, gated recurrence.

    Reference:
        Hao et al. "CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion."
        arXiv:2603.29452 (2026).

    Each image group is tokenized by its CNN (the unpooled feature map, one token per cell) and the
    normalized 1D observation, embedded by an MLP, is the single query of a multi-head attention over
    those tokens. ``[query, attended]`` passes a gated residual fusion block to give the per-step feature;
    a GRU integrates it over time and a highway gate blends the recurrent and feedforward features into
    the head input. Rollout state handling and sequence encoding are inherited.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        token_dim: int = 128,
        num_heads: int = 4,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        aux_target_dim: int = 0,
        aux_readout: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the CReF model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor").
            output_dim: Dimension of the output.
            token_dim: Width of the depth tokens and the proprioception embedding; the fused feature is twice it.
            num_heads: Attention heads of the cross-modal attention.
            rnn_type: "gru" or "lstm".
            rnn_hidden_dim: Recurrent hidden size.
            rnn_num_layers: Number of recurrent layers.
            aux_target_dim: Width of an auxiliary linear head on the fused feature; ``0`` adds none.
            aux_readout: Append the auxiliary head's (detached) prediction to the head input, so the
                supervised quantity reaches the policy as an explicit feature.
            **kwargs: Forwarded to :class:`CNNModel` (``hidden_dims``, ``cnn_cfg``, ``distribution_cfg``, ...).
        """
        self.token_dim = token_dim
        self.aux_readout = aux_readout and aux_target_dim > 0
        self.aux_target_dim = aux_target_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            aux_target_dim=aux_target_dim,
            **kwargs,
        )
        cnn_cfg = kwargs["cnn_cfg"]
        if not all(isinstance(v, dict) for v in cnn_cfg.values()):
            cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
        # tokenizers: the parent's CNNs are rebuilt unpooled so every feature-map cell is a token
        tokenizers, projections = {}, {}
        for idx, group in enumerate(self.obs_groups_2d):
            cfg = {**cnn_cfg[group], "global_pool": "none", "flatten": False}
            cnn = CNN(input_dim=self.obs_dims_2d[idx], input_channels=self.obs_channels_2d[idx], **cfg)
            tokenizers[group] = cnn
            projections[group] = nn.Linear(int(cnn.output_channels), token_dim)  # type: ignore[arg-type]
        self.cnns = nn.ModuleDict(tokenizers)
        self.token_proj = nn.ModuleDict(projections)
        self.token_norm = nn.LayerNorm(token_dim)
        act = resolve_nn_activation(kwargs.get("activation", "elu"))
        self.proprio_tokenizer = nn.Sequential(nn.Linear(self.obs_dim, token_dim), act, nn.Linear(token_dim, token_dim))
        self.query_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        fused = 2 * token_dim
        # gated residual fusion: x + c * sigmoid(g), with [c, g] from one hidden layer on LN(x)
        self.fusion_norm = nn.LayerNorm(fused)
        self.fusion_hidden = nn.Linear(fused, fused)
        self.fusion_out = nn.Linear(fused, 2 * fused)
        # recurrent fusion: GRU over the fused feature, then a highway gate against it
        self.rnn = RNN(fused, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.rec_proj = nn.Linear(rnn_hidden_dim, fused)
        self.highway_gate = nn.Linear(2 * fused, fused)
        # token grids (rows, cols) per image group, in the order tokens are concatenated
        self.token_grids = [tuple(int(v) for v in tokenizers[g].output_dim) for g in self.obs_groups_2d]  # type: ignore[union-attr]
        self._last_attn: torch.Tensor | None = None
        self._attn_weights: torch.Tensor | None = None

    def _features(self, obs: TensorDict) -> torch.Tensor:
        """Fused per-step feature for a flat ``(N,)`` batch, shape ``(N, 2 * token_dim)``."""
        proprio = MLPModel.get_latent(self, obs)
        query = self.proprio_tokenizer(proprio)
        tokens = []
        for group in self.obs_groups_2d:
            fmap = self.cnns[group](obs[group])  # (N, C, H, W)
            tokens.append(self.token_proj[group](fmap.flatten(2).transpose(1, 2)))  # (N, H*W, token_dim)
        keys = self.token_norm(torch.cat(tokens, dim=1))
        attended, weights = self.attn(self.query_norm(query).unsqueeze(1), keys, keys, need_weights=True)
        self._attn_weights = weights.squeeze(1)  # (N, num_tokens), heads averaged, differentiable
        self._last_attn = self._attn_weights.detach()
        x = torch.cat([query, attended.squeeze(1)], dim=-1)
        content, gate = self.fusion_out(nn.functional.elu(self.fusion_hidden(self.fusion_norm(x)))).chunk(2, dim=-1)
        return x + content * torch.sigmoid(gate)

    def _head_input(self, feats: torch.Tensor, rnn_out: torch.Tensor) -> torch.Tensor:
        """Highway gate between the recurrent feature and the current fused feature."""
        recurrent = self.rec_proj(rnn_out)
        beta = torch.sigmoid(self.highway_gate(torch.cat([recurrent, feats], dim=-1)))
        fused = beta * recurrent + (1.0 - beta) * feats
        if self.aux_readout:
            return torch.cat([fused, self.aux_head(fused).detach()], dim=-1)  # type: ignore[misc]
        return fused

    def attention_target_log_prob(self, uv: torch.Tensor) -> torch.Tensor:
        """Log attention mass on the first image group's cell containing normalized ``(u, v)``, shape ``(N,)``.

        Uses the weights of the last forward, so call it right after the forward that consumed the
        matching observations.
        """
        rows, cols = self.token_grids[0]
        col = (uv[:, 0] * cols).long().clamp(0, cols - 1)
        row = (uv[:, 1] * rows).long().clamp(0, rows - 1)
        idx = (row * cols + col).unsqueeze(1)
        return self._attn_weights.gather(1, idx).squeeze(1).clamp(min=1e-8).log()  # type: ignore[union-attr]

    def attention_entropy(self) -> torch.Tensor | None:
        """Mean attention entropy of the last forward, normalized to ``[0, 1]`` (1 = uniform over tokens)."""
        if self._last_attn is None:
            return None
        w = self._last_attn.clamp(min=1e-8)
        return (-(w * w.log()).sum(dim=-1) / math.log(w.shape[-1])).mean()

    def _aux_input_dim(self) -> int:
        """Return the fused feature width: the auxiliary head never reads its own readout."""
        return 2 * self.token_dim

    def _get_latent_dim(self) -> int:
        """Size the head for the fused feature width, plus the auxiliary readout when enabled."""
        return 2 * self.token_dim + (self.aux_target_dim if self.aux_readout else 0)
