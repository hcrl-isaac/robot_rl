"""Unit tests of the LSTD correction and bandit selection modules."""

import math
import torch

from robot_rl.adaptation import (
    LSTDEnsemble,
    PCAFeatures,
    ReturnRidge,
    candidate_values,
    episode_stats,
    perturbed_candidates,
    safety_mask,
    select,
    solve,
    vec_features,
)


def test_vec_kronecker_identity() -> None:
    """Vec kronecker identity."""
    g = torch.Generator().manual_seed(0)
    k, d = 5, 7
    phi = torch.randn(11, k, generator=g, dtype=torch.float64)
    W = torch.randn(k, d, generator=g, dtype=torch.float64)
    z = torch.randn(d, generator=g, dtype=torch.float64)
    direct = torch.einsum("nk,kd,d->n", phi, W, z)
    assert torch.allclose(vec_features(phi, z) @ W.flatten(), direct)


def _planted(
    n: int, k: int, d: int, gamma: float, noise: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    W_star = torch.randn(k, d, generator=g, dtype=torch.float64)
    phi = torch.randn(n, k, generator=g, dtype=torch.float64)
    phi_next = 0.6 * phi + 0.8 * torch.randn(n, k, generator=g, dtype=torch.float64)
    delta = (phi - gamma * phi_next) @ W_star + noise * torch.randn(n, d, generator=g, dtype=torch.float64)
    return W_star, phi, phi_next, delta


def test_lstd_recovers_planted_correction() -> None:
    """Lstd recovers planted correction."""
    gamma = 0.98
    W_star, phi, phi_next, delta = _planted(20_000, 6, 10, gamma, noise=0.1, seed=1)
    stats = [
        episode_stats(p, q, r, gamma)
        for p, q, r in zip(phi.split(1000), phi_next.split(1000), delta.split(1000), strict=True)
    ]
    fit = LSTDEnsemble(num_members=8, lam=1e-3, seed=0).fit(stats)
    rel = (fit.W_full - W_star).norm() / W_star.norm()
    assert rel < 0.02, rel
    assert (fit.W - W_star).norm() / W_star.norm() < 0.05
    assert fit.members.std(0).mean() < 0.05


def test_ensemble_spread_shrinks_with_data() -> None:
    """Ensemble spread shrinks with data."""
    gamma = 0.9
    _, phi, phi_next, delta = _planted(40_000, 4, 3, gamma, noise=1.0, seed=2)
    stats = [
        episode_stats(p, q, r, gamma)
        for p, q, r in zip(phi.split(500), phi_next.split(500), delta.split(500), strict=True)
    ]
    ens = LSTDEnsemble(num_members=32, lam=1.0, prior_scale=0.5, seed=0)
    few = ens.fit(stats[:3]).members.std(0).mean()
    many = ens.fit(stats).members.std(0).mean()
    assert many < 0.5 * few, (few, many)


def test_ridge_shrinks_toward_zero() -> None:
    """Ridge shrinks toward zero."""
    A = torch.eye(3, dtype=torch.float64) * 1e-6
    B = torch.ones(3, 2, dtype=torch.float64)
    assert solve(A, B, lam=1e6).abs().max() < 1e-5


def test_pca_features_whiten_and_add_bias() -> None:
    """Pca features whiten and add bias."""
    g = torch.Generator().manual_seed(3)
    F = torch.randn(5000, 8, generator=g) @ torch.randn(8, 8, generator=g) + 3.0
    fm = PCAFeatures(4).fit(F)
    phi = fm(F)
    assert phi.shape == (5000, 5) and torch.allclose(phi[:, 0], torch.ones(5000))
    assert torch.allclose(phi[:, 1:].std(0), torch.ones(4), atol=1e-3)


def test_candidates_stay_on_sphere_and_keep_center() -> None:
    """Candidates stay on sphere and keep center."""
    z0 = math.sqrt(16) * torch.nn.functional.normalize(torch.randn(16), dim=0)
    z = perturbed_candidates(z0, 32, 0.3, torch.Generator().manual_seed(0))
    assert z.shape == (33, 16) and torch.equal(z[0], z0)
    assert torch.allclose(z.norm(dim=-1), torch.full((33,), 4.0), atol=1e-5)


def test_linucb_finds_best_arm_on_toy_problem() -> None:
    """Linucb finds best arm on toy problem."""
    g = torch.Generator().manual_seed(4)
    num_arms, dim = 50, 6
    arms = torch.randn(num_arms, dim, generator=g, dtype=torch.float64)
    theta = torch.randn(dim, generator=g, dtype=torch.float64)
    true = arms @ theta
    best = int(true.argmax())
    bandit = ReturnRidge(dim, lam=1.0)
    picks = []
    for _ in range(150):
        value, width = bandit.values(torch.zeros(num_arms), arms, noise_std=0.3)
        i = select("ucb", value, width, None, torch.ones(num_arms, dtype=torch.bool), beta=1.0)
        bandit.add(arms[i], float(true[i] + 0.3 * torch.randn(1, generator=g, dtype=torch.float64)))
        picks.append(i)
    assert sum(p == best for p in picks[-50:]) >= 40, picks[-50:]


def test_ensemble_rules_pick_the_best_corrected_candidate() -> None:
    """Ensemble rules pick the best corrected candidate."""
    g = torch.Generator().manual_seed(5)
    k, d, n = 3, 4, 20
    phi_bar = torch.randn(n, k, generator=g, dtype=torch.float64)
    z_r = torch.randn(d, generator=g, dtype=torch.float64)
    W = torch.randn(k, d, generator=g, dtype=torch.float64)
    members = W + 1e-3 * torch.randn(16, k, d, generator=g, dtype=torch.float64)
    value, std, mv = candidate_values(torch.zeros(n), phi_bar, z_r, W, members)
    best = int((phi_bar @ W @ z_r).argmax())
    allowed = torch.ones(n, dtype=torch.bool)
    for rule in ("greedy", "ucb", "thompson"):
        assert select(rule, value, std, mv, allowed, generator=torch.Generator().manual_seed(0)) == best


def test_safety_mask_keeps_reference_and_drops_risky() -> None:
    """Safety mask keeps reference and drops risky."""
    value = torch.tensor([1.0, 2.0, 1.5, 3.0])
    std = torch.tensor([0.1, 0.1, 1.0, 5.0])
    mask = safety_mask(value, std, beta_s=1.0)
    assert mask.tolist() == [True, True, False, False]


def test_sandwich_members_center_on_the_fit_and_shrink_with_data() -> None:
    """Sandwich members scatter around the ridge fit, less so with more episodes."""
    gamma = 0.9
    _, phi, phi_next, delta = _planted(40_000, 4, 3, gamma, noise=1.0, seed=6)
    chunks = zip(phi.split(500), phi_next.split(500), delta.split(500), strict=True)
    stats = [episode_stats(p, q, r, gamma) for p, q, r in chunks]
    ens = LSTDEnsemble(num_members=256, lam=1.0, seed=0, uncertainty="sandwich")
    few, many = ens.fit(stats[:4]), ens.fit(stats)
    assert (many.members.mean(0) - many.W_full).abs().max() < 0.05
    assert many.members.std(0).mean() < 0.5 * few.members.std(0).mean()
