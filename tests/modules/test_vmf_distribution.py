# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The von Mises-Fisher sampler draws the right distribution with fixed shapes and no host sync."""

import torch

import pytest

from robot_rl.modules.distribution import (
    VonMisesFisherDistribution,
    _has_triton,
    _vmf_bessel_terms,
    _vmf_bessel_terms_fused,
    _wood_noise,
)

P = 256


def _dist(kappa: float, n: int, device: str = "cpu") -> VonMisesFisherDistribution:
    dist = VonMisesFisherDistribution(P, init_std=kappa**-0.5, kappa_range=(1e-3, 1e6)).to(device)
    dist.update(torch.randn(n, P, device=device))
    return dist


@pytest.mark.parametrize("kappa", [1.0, 100.0, 1e3])
def test_samples_match_the_analytic_moments(kappa: float) -> None:
    """``w = mu . x`` has mean ``A_p(kappa)`` and variance ``dA_p/dkappa = 1 - A_p^2 - (p-1)/kappa A_p``.

    Up to kappa = 1e3, where the Bessel recurrence matches scipy to 1e-12. Far above its seed depth it drifts
    (8e-6 at 3e4), more than the variance's near-cancellation tolerates.
    """
    torch.manual_seed(0)
    n = 40_000
    dist = _dist(kappa, n)
    x = dist.sample()
    assert torch.allclose(x.norm(dim=-1), torch.ones(n), atol=1e-5)
    w = (x * dist.mean).sum(dim=-1).double()
    # float64 reference: the variance is a near-cancellation at large kappa
    _, a = _vmf_bessel_terms(torch.tensor([kappa], dtype=torch.float64), P // 2, 64)
    var = 1.0 - a * a - (P - 1) / kappa * a
    assert abs(w.mean() - a) < 5 * (var / n).sqrt()
    assert abs(w.var() / var - 1.0) < 0.05


def test_every_row_accepts_a_proposal() -> None:
    """At the lowest acceptance rate (large kappa) every row accepts one of its proposals."""
    torch.manual_seed(0)
    _, accepted = _wood_noise(torch.tensor(1e5), P - 1.0, 200_000, torch.device("cpu"))
    assert accepted.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sampling_captures_in_a_cuda_graph() -> None:
    """The sampler has no data-dependent shapes or host reads, so a CUDA graph can replay it."""
    dist = _dist(1e3, 1024, "cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dist.sample()  # warm-up (allocations, the fused Bessel kernel)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x = dist.sample()
    graph.replay()
    first = x.clone()
    graph.replay()
    assert torch.allclose(x.norm(dim=-1), torch.ones(1024, device="cuda"), atol=1e-5)
    assert not torch.equal(first, x), "each replay draws fresh noise"


@pytest.mark.skipif(not torch.cuda.is_available() or not _has_triton(), reason="needs CUDA and Triton")
@pytest.mark.parametrize("kappa", [0.3, 30.0, 1e3, 1e5])
def test_fused_bessel_terms_match_eager(kappa: float) -> None:
    """The fused Bessel kernel gives the eager recurrence's values and kappa-gradient."""
    out = []
    for fn in (_vmf_bessel_terms, _vmf_bessel_terms_fused):
        k = torch.tensor([kappa], device="cuda", requires_grad=True)
        log_norm, a = fn(k, P // 2, 64)
        (grad,) = torch.autograd.grad(log_norm + a, k)
        out.append(torch.stack([log_norm.detach(), a.detach(), grad.squeeze()]))
    assert torch.allclose(out[0], out[1], rtol=1e-3, atol=1e-6)
