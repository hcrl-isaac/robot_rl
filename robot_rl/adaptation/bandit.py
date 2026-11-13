"""Linear-bandit choice of the next latent from corrected FB values: candidates, values, rules and safety."""

from __future__ import annotations

import math
import torch


def perturbed_candidates(
    z0: torch.Tensor, num: int, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Return ``z0`` followed by ``num`` sphere-projected perturbations ``normalize(z0 + sigma eps)``.

    Args:
        z0: Center latent ``(d,)`` on the radius-``sqrt(d)`` sphere.
        num: Number of perturbations ``N``.
        sigma: Perturbation std per coordinate.
        generator: Torch generator on ``z0``'s device.

    Returns:
        ``(num + 1, d)`` latents; row 0 is ``z0``.
    """
    d = z0.shape[-1]
    eps = torch.randn(num, d, device=z0.device, generator=generator)
    z = z0 + sigma * eps
    z = math.sqrt(d) * torch.nn.functional.normalize(z, dim=-1)
    return torch.cat([z0[None], z], dim=0)


def vec_features(phi_bar: torch.Tensor, z_r: torch.Tensor) -> torch.Tensor:
    """Return ``x`` with ``phi^T W z_r = x . W.flatten()`` (row-major ``vec``), i.e. ``phi kron z_r``.

    Args:
        phi_bar: Features ``(..., k)``.
        z_r: Reward latent ``(d,)``.

    Returns:
        ``(..., k * d)``.
    """
    return (phi_bar[..., :, None] * z_r).flatten(-2)


def candidate_values(
    j0: torch.Tensor,
    phi_bar: torch.Tensor,
    z_r: torch.Tensor,
    W: torch.Tensor,  # noqa: N803
    members: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Corrected values ``J(z) = J_0(z) + phi_bar(z)^T W z_r`` of each candidate.

    Args:
        j0: Uncorrected values ``E_D[F^T z_r]`` ``(N,)``.
        phi_bar: ``E_D[phi(s, pi_z(s), z)]`` ``(N, k)``.
        z_r: Reward latent ``(d,)``.
        W: Point estimate ``(k, d)``.
        members: Ensemble ``(K, k, d)`` or ``None``.

    Returns:
        Point values ``(N,)``, ensemble std ``(N,)`` (zeros without members) and member values ``(K, N)``.
    """
    phi_bar, z_r = phi_bar.double(), z_r.double()
    j = j0.double() + phi_bar @ (W.double() @ z_r)
    if members is None:
        return j, torch.zeros_like(j), None
    jb = j0.double() + torch.einsum("nk,bk->bn", phi_bar, members.double() @ z_r)
    return j, jb.std(dim=0, unbiased=jb.shape[0] > 1), jb


def select(
    rule: str,
    value: torch.Tensor,
    std: torch.Tensor,
    member_values: torch.Tensor | None,
    allowed: torch.Tensor,
    beta: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Pick a candidate index by ``greedy``, ``ucb`` or ``thompson`` among the ``allowed`` ones.

    Args:
        rule: Selection rule.
        value: Point values ``(N,)``.
        std: Ensemble std ``(N,)``.
        member_values: ``(K, N)``; Thompson samples one row.
        allowed: Boolean mask ``(N,)`` from the safety filter.
        beta: UCB exploration bonus multiplier.
        generator: CPU generator for Thompson's member draw.

    Returns:
        The chosen index.
    """
    if rule == "greedy":
        score = value
    elif rule == "ucb":
        score = value + beta * std
    elif rule == "thompson":
        b = torch.randint(0, member_values.shape[0], (1,), generator=generator).item()
        score = member_values[b]
    else:
        raise ValueError(f"unknown selection rule {rule!r}")
    score = torch.where(allowed, score, torch.full_like(score, -math.inf))
    return int(score.argmax().item())


def safety_mask(value: torch.Tensor, std: torch.Tensor, beta_s: float, reference: int = 0) -> torch.Tensor:
    """Keep candidates whose lower bound ``J - beta_s std`` is at least the reference's (``z0``) lower bound."""
    lcb = value - beta_s * std
    mask = lcb >= lcb[reference]
    mask[reference] = True
    return mask


class ReturnRidge:
    """Baseline linear bandit fit on episode returns only: ``y_e = G_e - J_0(z_e) ~ x(z_e) . theta``."""

    def __init__(self, dim: int, lam: float) -> None:
        """Initialize an empty ridge fit.

        Args:
            dim: Feature dimension ``k*d``.
            lam: Ridge strength.
        """
        self.dim = dim
        self.lam = lam
        self.xs: list[torch.Tensor] = []
        self.ys: list[float] = []

    def add(self, x: torch.Tensor, y: float) -> None:
        """Record one pulled arm's features ``(k*d,)`` and its return residual."""
        self.xs.append(x.double().cpu())
        self.ys.append(float(y))

    def posterior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ridge mean ``theta`` and the inverse design ``V^{-1}`` (LinUCB's covariance up to noise)."""
        V = self.lam * torch.eye(self.dim, dtype=torch.float64)
        b = torch.zeros(self.dim, dtype=torch.float64)
        for x, y in zip(self.xs, self.ys, strict=True):
            V += torch.outer(x, x)
            b += y * x
        V_inv = torch.linalg.inv(V)
        return V_inv @ b, V_inv

    def values(self, j0: torch.Tensor, x: torch.Tensor, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted values and LinUCB widths ``noise_std * sqrt(x^T V^{-1} x)`` for candidates ``(N, k*d)``."""
        theta, V_inv = self.posterior()
        x = x.double().cpu()
        width = noise_std * torch.einsum("ni,ij,nj->n", x, V_inv, x).clamp_min(0).sqrt()
        return j0.double().cpu() + x @ theta, width


class GaussianArms:
    """Independent Gaussian beliefs over each candidate's mean return, with a prior per candidate.

    Arm ``i`` has prior ``N(m_i, tau^2)`` and observations with noise ``sigma``; its posterior uses only its own
    returns, so a wrong prior costs extra plays of the arms it overrates but cannot bias an arm that is played.
    """

    def __init__(self, num_arms: int, sigma: float) -> None:
        """Create the beliefs.

        Args:
            num_arms: Number of candidates ``K``.
            sigma: Per-episode return noise.
        """
        self.sigma = sigma
        self.sums = torch.zeros(num_arms, dtype=torch.float64)
        self.counts = torch.zeros(num_arms, dtype=torch.float64)

    def add(self, arm: int, value: float) -> None:
        """Record one observed return of ``arm``."""
        self.sums[arm] += value
        self.counts[arm] += 1

    def posterior(self, prior_mean: torch.Tensor, prior_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior means and stds given per-arm priors ``(K,)`` (an infinite std is a flat prior)."""
        prior_prec = 1.0 / prior_std.double().pow(2)
        prec = prior_prec + self.counts / self.sigma**2
        mean = (prior_mean.double() * prior_prec + self.sums / self.sigma**2) / prec
        return mean, prec.rsqrt()

    def ucb(self, prior_mean: torch.Tensor, prior_std: torch.Tensor, beta: float) -> int:
        """Return the arm with the highest ``mean + beta * std``."""
        mean, std = self.posterior(prior_mean, prior_std)
        return int((mean + beta * std).argmax().item())

    def thompson(self, prior_mean: torch.Tensor, prior_std: torch.Tensor, generator: torch.Generator) -> int:
        """Return the arm with the highest sample from its posterior."""
        mean, std = self.posterior(prior_mean, prior_std)
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
        return int((mean + std * noise).argmax().item())
