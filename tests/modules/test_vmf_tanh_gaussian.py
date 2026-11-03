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


def test_a_per_state_kappa_is_read_from_its_own_output_and_initialized_alone() -> None:
    """With per-state kappa the network emits HEAD + 1 + TAIL values; only the kappa row is reinitialized."""
    dist = VonMisesFisherTanhGaussianDistribution(HEAD + TAIL, tail_dim=TAIL, init_std=0.5, state_dependent_kappa=True)
    assert dist.input_dim == HEAD + 1 + TAIL
    mlp = torch.nn.Sequential(torch.nn.Linear(4, dist.input_dim))
    before = mlp[0].weight.detach().clone()
    dist.init_mlp_weights(mlp)
    assert torch.equal(mlp[0].weight[HEAD], torch.zeros(4))
    assert torch.equal(mlp[0].weight[HEAD + 1 :], before[HEAD + 1 :])
    raw = torch.randn(3, dist.input_dim)
    dist.update(raw)
    action, logp = dist.sample_and_log_prob()
    assert action.shape == (3, HEAD + TAIL) and logp.shape == (3,)
    module = torch.jit.script(dist.as_deterministic_output_module())
    torch.testing.assert_close(module(raw), dist.deterministic_output(raw))


def test_a_per_state_tail_std_is_read_after_the_tail_mean_and_starts_at_the_init_std() -> None:
    """With per-state kappa and tail std the network emits HEAD + 1 + 2 * TAIL values; the std rows start constant."""
    dist = VonMisesFisherTanhGaussianDistribution(
        HEAD + TAIL,
        tail_dim=TAIL,
        init_std=0.5,
        tail_init_noise_std=0.3,
        state_dependent_kappa=True,
        tail_state_dependent_std=True,
    )
    assert dist.input_dim == HEAD + 1 + 2 * TAIL
    mlp = torch.nn.Sequential(torch.nn.Linear(4, dist.input_dim))
    dist.init_mlp_weights(mlp)
    dist.update(mlp(torch.randn(3, 4)))
    torch.testing.assert_close(dist.tail.std, torch.full((3, TAIL), 0.3))
    raw = torch.randn(3, dist.input_dim)
    dist.update(raw)
    torch.testing.assert_close(dist.tail.std, raw[:, HEAD + 1 + TAIL :].exp())
    action, logp = dist.sample_and_log_prob()
    assert action.shape == (3, HEAD + TAIL) and logp.shape == (3,)
    module = torch.jit.script(dist.as_deterministic_output_module())
    torch.testing.assert_close(module(raw), dist.deterministic_output(raw))
