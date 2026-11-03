"""A standalone recurrent regressor that estimates a privileged quantity from an actor's observations."""

from __future__ import annotations

import torch
from torch import nn


class SequenceEstimator(nn.Module):
    """Normalized input, a GRU, and a linear readout, trained only by supervision."""

    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int = 128) -> None:
        """Build the estimator.

        Args:
            input_dim: Width of the per-step input.
            target_dim: Width of the estimated quantity.
            hidden_dim: GRU state width.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, target_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
        """Estimate the target at every step of a window, starting from a zero state.

        Args:
            x: Inputs, ``(T, B, input_dim)``.
            resets: Episode starts, ``(T, B)``; the state is zeroed before a step that starts an episode.

        Returns:
            Estimates, ``(T, B, target_dim)``.
        """
        h = x.new_zeros(x.shape[1], self.hidden_dim)
        x = self.norm(x)
        outs = []
        for t in range(x.shape[0]):
            h = h * (1.0 - resets[t].float()).unsqueeze(-1)
            h = self.gru(x[t], h)
            outs.append(self.readout(h))
        return torch.stack(outs)
