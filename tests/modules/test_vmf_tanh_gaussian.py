"""The vMF direction joined with a squashed Gaussian tail: each part keeps its own law and their scores add."""

from __future__ import annotations

import torch

from robot_rl.modules import SquashedTanhGaussianDistribution, VonMisesFisherDistribution
from robot_rl.modules.distribution import VonMisesFisherTanhGaussianDistribution

HEAD, TAIL = 8, 2


def _dist() -> VonMisesFisherTanhGaussianDistribution:
    dist = VonMisesFisherTanhGaussianDistribution(HEAD + TAIL, tail_dim=TAIL, init_std=0.5, tail_init_noise_std=0.3)
    dist.update(torch.randn(5, HEAD + TAIL))
    return dist


def test_samples_are_a_unit_direction_then_a_bounded_tail() -> None:
    """The head of a sample lies on the sphere and the tail inside the bounds."""
    action, logp = _dist().sample_and_log_prob()
    assert action.shape == (5, HEAD + TAIL) and logp.shape == (5,)
    torch.testing.assert_close(action[:, :HEAD].norm(dim=-1), torch.ones(5))
    assert action[:, HEAD:].abs().max() < 1.0


def test_scores_are_the_sum_of_the_parts() -> None:
    """log_prob and entropy equal the vMF part's plus the squashed Gaussian part's."""
    torch.manual_seed(0)
    raw = torch.randn(5, HEAD + TAIL)
    dist = VonMisesFisherTanhGaussianDistribution(HEAD + TAIL, tail_dim=TAIL, init_std=0.5, tail_init_noise_std=0.3)
    dist.update(raw)
    head = VonMisesFisherDistribution(HEAD, init_std=0.5)
    head.update(raw[:, :HEAD])
    tail = SquashedTanhGaussianDistribution(TAIL, init_noise_std=0.3)
    tail.update(raw[:, HEAD:])
    action = dist.sample()
    torch.testing.assert_close(dist.log_prob(action), head.log_prob(action[:, :HEAD]) + tail.log_prob(action[:, HEAD:]))
    torch.testing.assert_close(dist.entropy, head.entropy + tail.entropy)


def test_the_export_module_matches_the_deterministic_output() -> None:
    """The scripted deterministic module returns the normalized head and the squashed tail."""
    dist = _dist()
    raw = torch.randn(3, HEAD + TAIL)
    module = torch.jit.script(dist.as_deterministic_output_module())
    torch.testing.assert_close(module(raw), dist.deterministic_output(raw))
