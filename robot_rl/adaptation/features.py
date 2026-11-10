"""Frozen feature maps ``phi(s, a, z)`` for the linear correction ``F' = F + W^T phi``."""

from __future__ import annotations

import torch


class FeatureMap:
    """Base class: maps forward embeddings (and optional physical features) to correction features."""

    dim: int

    def fit(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> FeatureMap:
        """Fit any statistics on nominal data; the map is frozen afterwards.

        Args:
            forward: ``F(s, a, z)`` samples ``(n, d)``.
            physical: Physical features of the same samples ``(n, p)``, if the map uses them.

        Returns:
            ``self``.
        """
        return self

    def __call__(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``phi`` ``(..., dim)`` for ``forward`` ``(..., d)``."""
        raise NotImplementedError


class BiasFeatures(FeatureMap):
    """``phi = [1]``: one global additive shift of ``F``."""

    dim = 1

    def __call__(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> torch.Tensor:
        """Return ones ``(..., 1)``."""
        return torch.ones(*forward.shape[:-1], 1, device=forward.device, dtype=forward.dtype)


class ForwardFeatures(FeatureMap):
    """``phi = F / scale``: the full linear model ``F' = (I + W^T) F``; needs heavy regularization."""

    def __init__(self, z_dim: int) -> None:
        """Initialize with unit scale.

        Args:
            z_dim: Latent dimension ``d``.
        """
        self.dim = z_dim
        self.scale = 1.0

    def fit(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> ForwardFeatures:
        """Fit ``scale`` to the RMS per-dimension norm of ``forward``."""
        self.scale = forward.norm(dim=-1).mean().item() / forward.shape[-1] ** 0.5
        return self

    def __call__(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``F / scale``."""
        return forward / self.scale


class PCAFeatures(FeatureMap):
    """``phi = [1, whitened top-k PCA scores of F]``, PCA fit once on nominal data."""

    def __init__(self, k: int) -> None:
        """Initialize an unfit map.

        Args:
            k: Number of PCA components.
        """
        self.k = k
        self.dim = k + 1

    def fit(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> PCAFeatures:
        """Fit the PCA basis and score scales on nominal ``forward``."""
        self.mean = forward.mean(dim=0)
        _, s, vh = torch.linalg.svd(forward - self.mean, full_matrices=False)
        self.components = vh[: self.k].T
        self.std = (s[: self.k] / (forward.shape[0] - 1) ** 0.5).clamp_min(1e-6)
        self.explained = (s[: self.k].pow(2).sum() / s.pow(2).sum()).item()
        return self

    def __call__(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``[1, whitened scores]`` ``(..., k + 1)``."""
        scores = (forward - self.mean) @ self.components / self.std
        return torch.cat([torch.ones_like(scores[..., :1]), scores], dim=-1)


class PCAPhysicalFeatures(PCAFeatures):
    """``pca`` features plus standardized physical features (e.g. joint velocities, foot heights)."""

    def __init__(self, k: int, physical_dim: int) -> None:
        """Initialize an unfit map.

        Args:
            k: Number of PCA components.
            physical_dim: Number of physical features.
        """
        super().__init__(k)
        self.physical_dim = physical_dim
        self.dim = k + 1 + physical_dim

    def fit(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> PCAPhysicalFeatures:
        """Fit the PCA basis and the physical-feature standardization."""
        super().fit(forward)
        self.p_mean = physical.mean(dim=0)
        self.p_std = physical.std(dim=0).clamp_min(1e-6)
        return self

    def __call__(self, forward: torch.Tensor, physical: torch.Tensor | None = None) -> torch.Tensor:
        """Return PCA features concatenated with standardized ``physical``."""
        return torch.cat([super().__call__(forward), (physical - self.p_mean) / self.p_std], dim=-1)


def make_feature_map(kind: str, z_dim: int, k: int = 16, physical_dim: int = 0) -> FeatureMap:
    """Build a feature map by name: ``bias``, ``pca``, ``F`` or ``pca+phys``."""
    if kind == "bias":
        return BiasFeatures()
    if kind == "pca":
        return PCAFeatures(k)
    if kind == "F":
        return ForwardFeatures(z_dim)
    if kind == "pca+phys":
        return PCAPhysicalFeatures(k, physical_dim)
    raise ValueError(f"unknown feature map {kind!r}")
