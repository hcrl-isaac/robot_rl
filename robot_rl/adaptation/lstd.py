"""LSTD estimation of a linear correction ``W`` to frozen successor features, with an episode bootstrap.

For a transition ``(s, a, s')`` and latent ``z`` with ``a' = pi_z(s')``, the corrected forward model
``F' = F + W^T phi`` satisfies the successor-feature Bellman equation when
``W^T (phi - gamma phi') = delta``, ``delta = c(s') + gamma F(s', a', z) - F(s, a, z)``. Using ``phi`` as the
instrument gives ``A W = B`` with ``A = sum phi (phi - gamma phi')^T`` and ``B = sum phi delta^T``.
"""

from __future__ import annotations

import dataclasses
import torch


@dataclasses.dataclass
class LSTDStats:
    """Sufficient statistics of one episode: ``A`` ``(k, k)``, ``B`` ``(k, d)`` and the row count."""

    A: torch.Tensor
    B: torch.Tensor
    count: int
    delta_sq: float


def episode_stats(phi: torch.Tensor, phi_next: torch.Tensor, delta: torch.Tensor, gamma: float) -> LSTDStats:
    """Accumulates the LSTD statistics of one episode's ``(transition, z)`` rows.

    Args:
        phi: ``phi(s, a, z)`` ``(n, k)``.
        phi_next: ``phi(s', pi_z(s'), z)`` ``(n, k)``.
        delta: Vector TD errors of the frozen ``F`` ``(n, d)``.
        gamma: Discount of the successor features.

    Returns:
        The statistics (float64 sums).
    """
    phi, phi_next, delta = phi.double(), phi_next.double(), delta.double()
    return LSTDStats(
        A=phi.T @ (phi - gamma * phi_next),
        B=phi.T @ delta,
        count=phi.shape[0],
        delta_sq=delta.pow(2).sum().item(),
    )


def solve(A: torch.Tensor, B: torch.Tensor, lam: float, prior: torch.Tensor | None = None) -> torch.Tensor:  # noqa: N803
    """Return ``W = (A + lam I)^{-1} (B + lam W_0)``, ridge toward ``W_0`` (zero when ``prior`` is ``None``)."""
    k = A.shape[0]
    rhs = B if prior is None else B + lam * prior
    return torch.linalg.solve(A + lam * torch.eye(k, dtype=A.dtype, device=A.device), rhs)


@dataclasses.dataclass
class LSTDFit:
    """An ensemble fit: the point estimate, the members and conditioning diagnostics."""

    W: torch.Tensor
    W_full: torch.Tensor
    members: torch.Tensor
    cond: float
    eig_min_real: float
    eig_neg_frac: float


class LSTDEnsemble:
    """Bootstrap-over-episodes ensemble of ridge-regularized LSTD fits with randomized priors."""

    def __init__(self, num_members: int, lam: float, prior_scale: float = 0.0, seed: int = 0) -> None:
        """Configure the ensemble.

        Args:
            num_members: Bootstrap members ``K``.
            lam: Ridge strength toward ``W = 0`` (or toward each member's prior offset).
            prior_scale: Std of each member's random prior offset ``W_0``; keeps early ensembles from collapsing.
            seed: Seed of the bootstrap and prior draws.
        """
        self.num_members = num_members
        self.lam = lam
        self.prior_scale = prior_scale
        self.generator = torch.Generator().manual_seed(seed)

    def fit(self, episodes: list[LSTDStats]) -> LSTDFit:
        """Fit the point estimate and the members on the episodes collected so far.

        Args:
            episodes: Per-episode statistics (at least one).

        Returns:
            The fit; ``W`` is the member mean, ``W_full`` the plain ridge fit on all episodes.
        """
        A = torch.stack([e.A for e in episodes])
        B = torch.stack([e.B for e in episodes])
        k, d = B.shape[1:]
        A_all, B_all = A.sum(0), B.sum(0)
        W_full = solve(A_all, B_all, self.lam)
        n = len(episodes)
        counts = torch.multinomial(torch.ones(n), n * self.num_members, replacement=True, generator=self.generator)
        weights = torch.nn.functional.one_hot(counts.view(self.num_members, n), n).sum(1).double()
        members = []
        for b in range(self.num_members):
            w = weights[b].to(A.device)
            prior = None
            if self.prior_scale > 0:
                prior = (self.prior_scale * torch.randn(k, d, generator=self.generator, dtype=torch.float64)).to(
                    A.device
                )
            members.append(solve((w[:, None, None] * A).sum(0), (w[:, None, None] * B).sum(0), self.lam, prior))
        members = torch.stack(members)
        reg = A_all + self.lam * torch.eye(k, dtype=A_all.dtype, device=A_all.device)
        eig = torch.linalg.eigvals(reg)
        return LSTDFit(
            W=members.mean(0),
            W_full=W_full,
            members=members,
            cond=torch.linalg.cond(reg).item(),
            eig_min_real=eig.real.min().item(),
            eig_neg_frac=(eig.real < 0).double().mean().item(),
        )


def relative_residual(stats: LSTDStats, W: torch.Tensor) -> float:  # noqa: N803
    """Return the instrumented Bellman residual ``||A W - B|| / ||B||`` of ``W`` on one (held-out) episode."""
    resid = stats.A @ W - stats.B
    return (resid.norm() / stats.B.norm().clamp_min(1e-12)).item()


def td_fraction_explained(
    phi: torch.Tensor,
    phi_next: torch.Tensor,
    delta: torch.Tensor,
    W: torch.Tensor,  # noqa: N803
    gamma: float,
) -> float:
    """Return ``1 - ||delta - W^T (phi - gamma phi')||^2 / ||delta||^2`` on raw rows (held-out fit quality)."""
    pred = (phi.double() - gamma * phi_next.double()) @ W
    resid = (delta.double() - pred).pow(2).sum()
    return (1.0 - resid / delta.double().pow(2).sum().clamp_min(1e-12)).item()
