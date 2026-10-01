# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distribution modules."""

import math
import torch

from robot_rl.modules.distribution import (
    BetaDistribution,
    GaussianDistribution,
    HeteroscedasticGaussianDistribution,
    SquashedTanhGaussianDistribution,
)


class TestGaussianDistribution:
    """Tests for ``GaussianDistribution``."""

    def test_set_std_on_a_slice(self) -> None:
        """``set_std_`` sets the std of the given dims only, in scalar or log parameterization."""
        for std_type in ("scalar", "log"):
            dist = GaussianDistribution(output_dim=4, init_std=1.0, std_type=std_type)
            dist.set_std_(0.25, slice(1, 3))
            dist.update(torch.zeros(1, 4))
            assert torch.allclose(dist.std, torch.tensor([[1.0, 0.25, 0.25, 1.0]]))

    def test_project_std_restores_gradient_past_bound(self) -> None:
        """A std parameter pushed past ``std_range`` gets no gradient through the clamp until it is projected back."""
        for std_type, name in (("scalar", "std_param"), ("log", "log_std_param")):
            dist = GaussianDistribution(output_dim=2, init_std=1.0, std_range=(0.05, 1.0), std_type=std_type)
            param = getattr(dist, name)
            with torch.no_grad():
                param.add_(0.01)  # e.g. the entropy bonus stepped it just past the cap

            dist.update(torch.zeros(1, 2))
            dist.std.sum().backward()
            assert torch.all(param.grad == 0.0)

            param.grad = None
            dist.project_std_()
            dist.update(torch.zeros(1, 2))
            dist.std.sum().backward()
            assert torch.allclose(dist.std, torch.ones(1, 2))
            assert torch.all(param.grad != 0.0)

    def test_log_prob_standard_normal(self) -> None:
        """log_prob at the mean of N(0,1) should equal -0.5*log(2*pi) per dimension, summed."""
        dim = 4
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.zeros(1, dim)
        dist.update(mean)

        log_p = dist.log_prob(torch.zeros(1, dim))
        expected = -0.5 * math.log(2 * math.pi) * dim
        assert torch.allclose(log_p, torch.tensor([expected]), atol=1e-5)

    def test_log_prob_nonzero_mean(self) -> None:
        """log_prob should decrease as the sample moves away from the mean."""
        dist = GaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar")
        mean = torch.tensor([[3.0, 3.0]])
        dist.update(mean)

        lp_at_mean = dist.log_prob(mean)
        lp_far = dist.log_prob(mean + 5.0)
        assert lp_at_mean > lp_far, "log_prob should be higher at the mean"

    def test_entropy_analytical(self) -> None:
        """Entropy should match the analytical formula 0.5 * sum(log(2*pi*e*std^2))."""
        dim = 3
        std_val = 2.0
        dist = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist.update(torch.zeros(1, dim))

        expected = 0.5 * dim * math.log(2 * math.pi * math.e * std_val**2)
        assert torch.allclose(dist.entropy, torch.tensor([expected]), atol=1e-4)

    def test_kl_divergence_analytical(self) -> None:
        """KL(N(0,1) || N(mu,sigma)) should match the closed-form KL for univariate Gaussians."""
        dist = GaussianDistribution(output_dim=1, init_std=1.0, std_type="scalar")
        mu_old, sigma_old = 0.0, 1.0
        mu_new, sigma_new = 1.0, 2.0

        old_params = (torch.tensor([[mu_old]]), torch.tensor([[sigma_old]]))
        new_params = (torch.tensor([[mu_new]]), torch.tensor([[sigma_new]]))

        kl = dist.kl_divergence(old_params, new_params)

        # Analytical KL: log(s2/s1) + (s1^2 + (m1-m2)^2) / (2*s2^2) - 0.5
        expected = math.log(sigma_new / sigma_old) + (sigma_old**2 + (mu_old - mu_new) ** 2) / (2 * sigma_new**2) - 0.5
        assert torch.allclose(kl, torch.tensor([expected]), atol=1e-5)

    def test_kl_divergence_identical_is_zero(self) -> None:
        """KL(p || p) should be zero."""
        dist = GaussianDistribution(output_dim=4, init_std=1.5, std_type="scalar")
        params = (torch.zeros(1, 4), torch.full((1, 4), 1.5))
        kl = dist.kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros(1), atol=1e-6)

    def test_scalar_vs_log_std_equivalence(self) -> None:
        """Scalar and log parameterizations should give identical results for the same effective std."""
        dim = 3
        std_val = 1.5
        dist_scalar = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist_log = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="log")

        mean = torch.randn(2, dim)
        dist_scalar.update(mean)
        dist_log.update(mean)

        sample_point = torch.randn(2, dim)

        assert torch.allclose(dist_scalar.log_prob(sample_point), dist_log.log_prob(sample_point), atol=1e-5)
        assert torch.allclose(dist_scalar.entropy, dist_log.entropy, atol=1e-5)

    def test_log_prob_gradient_flows_to_mean(self) -> None:
        """log_prob should allow gradient flow back to the distribution mean."""
        dim = 3
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.randn(1, dim, requires_grad=True)
        dist.update(mean)

        sample = dist.sample().detach()
        log_p = dist.log_prob(sample)
        log_p.sum().backward()
        assert mean.grad is not None, "Gradient should flow from log_prob to mean"
        assert not torch.all(mean.grad == 0), "Gradient should be non-zero"

    def test_std_clamped_to_range_scalar(self) -> None:
        """The std should be clamped to both bounds of std_range for std_type='scalar'."""
        dim = 2
        std_range = (0.1, 2.0)
        # Above the upper bound.
        dist_high = GaussianDistribution(output_dim=dim, init_std=10.0, std_type="scalar", std_range=std_range)
        dist_high.update(torch.zeros(1, dim))
        assert torch.allclose(dist_high.std, torch.full((1, dim), std_range[1]), atol=1e-6)
        # Below the lower bound.
        dist_low = GaussianDistribution(output_dim=dim, init_std=0.01, std_type="scalar", std_range=std_range)
        dist_low.update(torch.zeros(1, dim))
        assert torch.allclose(dist_low.std, torch.full((1, dim), std_range[0]), atol=1e-6)

    def test_std_clamped_to_range_log(self) -> None:
        """The std should be clamped to both bounds of std_range for std_type='log'."""
        dim = 2
        std_range = (0.1, 2.0)
        # Above the upper bound.
        dist_high = GaussianDistribution(output_dim=dim, init_std=10.0, std_type="log", std_range=std_range)
        dist_high.update(torch.zeros(1, dim))
        assert torch.allclose(dist_high.std, torch.full((1, dim), std_range[1]), atol=1e-6)
        # Below the lower bound.
        dist_low = GaussianDistribution(output_dim=dim, init_std=0.01, std_type="log", std_range=std_range)
        dist_low.update(torch.zeros(1, dim))
        assert torch.allclose(dist_low.std, torch.full((1, dim), std_range[0]), atol=1e-6)

    def test_std_range_min_floor(self) -> None:
        """The minimum of std_range should be floored to 1e-6 for numerical stability."""
        dist = GaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar", std_range=(0.0, 10.0))
        assert dist.std_range[0] == 1e-6

    def test_learn_std_scalar(self) -> None:
        """learn_std should control whether the scalar std parameter is learnable."""
        dim = 3
        init_std = 0.7
        # learn_std=True: parameter is trainable and receives non-zero gradient.
        dist_learn = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="scalar", learn_std=True)
        assert dist_learn.std_param.requires_grad is True
        dist_learn.update(torch.randn(2, dim))
        sample = dist_learn.sample().detach()
        dist_learn.log_prob(sample).sum().backward()
        assert dist_learn.std_param.grad is not None and not torch.all(dist_learn.std_param.grad == 0)
        # learn_std=False: parameter is frozen and receives no gradient.
        dist_fixed = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="scalar", learn_std=False)
        assert dist_fixed.std_param.requires_grad is False
        mean = torch.randn(2, dim, requires_grad=True)
        dist_fixed.update(mean)
        sample = dist_fixed.sample().detach()
        dist_fixed.log_prob(sample).sum().backward()
        assert dist_fixed.std_param.grad is None, "Non-learnable std should not receive gradients"
        assert torch.allclose(dist_fixed.std_param, torch.full((dim,), init_std), atol=1e-6)

    def test_learn_std_log(self) -> None:
        """learn_std should control whether the log std parameter is learnable."""
        dim = 3
        init_std = 0.7
        # learn_std=True: parameter is trainable and receives non-zero gradient.
        dist_learn = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="log", learn_std=True)
        assert dist_learn.log_std_param.requires_grad is True
        dist_learn.update(torch.randn(2, dim))
        sample = dist_learn.sample().detach()
        dist_learn.log_prob(sample).sum().backward()
        assert dist_learn.log_std_param.grad is not None and not torch.all(dist_learn.log_std_param.grad == 0)
        # learn_std=False: parameter is frozen and receives no gradient.
        dist_fixed = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="log", learn_std=False)
        assert dist_fixed.log_std_param.requires_grad is False
        mean = torch.randn(2, dim, requires_grad=True)
        dist_fixed.update(mean)
        sample = dist_fixed.sample().detach()
        dist_fixed.log_prob(sample).sum().backward()
        assert dist_fixed.log_std_param.grad is None, "Non-learnable log std should not receive gradients"
        assert torch.allclose(dist_fixed.log_std_param, torch.log(torch.full((dim,), init_std)), atol=1e-6)


class TestHeteroscedasticGaussianDistribution:
    """Tests for ``HeteroscedasticGaussianDistribution``."""

    def test_update_splits_mean_and_std(self) -> None:
        """update() should parse MLP output into separate mean and std."""
        dim = 4
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.randn(2, dim)
        std_val = torch.abs(torch.randn(2, dim)) + 0.1
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.mean, mean_val, atol=1e-6)
        assert torch.allclose(dist.std, std_val, atol=1e-6)

    def test_deterministic_output_returns_mean(self) -> None:
        """deterministic_output() should extract the mean from the MLP output."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.tensor([[1.0, 2.0, 3.0]])
        std_val = torch.tensor([[0.5, 0.5, 0.5]])
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        result = dist.deterministic_output(mlp_output)
        assert torch.allclose(result, mean_val)

    def test_log_std_parameterization(self) -> None:
        """With std_type='log', the second slice should be treated as log(std)."""
        dim = 2
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="log")

        mean_val = torch.zeros(1, dim)
        log_std_val = torch.zeros(1, dim)  # log(1) = 0, so std = 1
        mlp_output = torch.stack([mean_val, log_std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.std, torch.ones(1, dim), atol=1e-6)

    def test_input_dim_is_pair(self) -> None:
        """input_dim should be [2, output_dim] to accommodate mean and std."""
        dim = 5
        dist = HeteroscedasticGaussianDistribution(output_dim=dim)
        assert dist.input_dim == [2, dim]

    def test_std_clamped_to_range_scalar(self) -> None:
        """The state-dependent std should be clamped to std_range for std_type='scalar'."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(
            output_dim=dim, init_std=1.0, std_type="scalar", std_range=(0.1, 2.0)
        )

        mean_val = torch.zeros(1, dim)
        std_val = torch.tensor([[10.0, 0.01, 1.0]])  # above, below, inside
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        dist.update(mlp_output)
        expected = torch.tensor([[2.0, 0.1, 1.0]])
        assert torch.allclose(dist.std, expected, atol=1e-6)

    def test_std_clamped_to_range_log(self) -> None:
        """The state-dependent std should be clamped to std_range for std_type='log'."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="log", std_range=(0.1, 2.0))

        mean_val = torch.zeros(1, dim)
        # log values: log(10) above, log(0.01) below, log(1) inside
        log_std_val = torch.log(torch.tensor([[10.0, 0.01, 1.0]]))
        mlp_output = torch.stack([mean_val, log_std_val], dim=-2)

        dist.update(mlp_output)
        expected = torch.tensor([[2.0, 0.1, 1.0]])
        assert torch.allclose(dist.std, expected, atol=1e-6)

    def test_std_range_min_floor(self) -> None:
        """The minimum of std_range should be floored to 1e-6 for numerical stability."""
        dist = HeteroscedasticGaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar", std_range=(0.0, 10.0))
        assert dist.std_range[0] == 1e-6


class TestSquashedTanhGaussianDistribution:
    """Tests for ``SquashedTanhGaussianDistribution``."""

    def test_samples_within_bounds(self) -> None:
        """Squashed samples should lie within [low, high] (tanh may saturate to the bound in float32)."""
        low, high = -2.0, 2.0
        dist = SquashedTanhGaussianDistribution(output_dim=4, init_noise_std=1.0, low=low, high=high)
        dist.update(torch.randn(256, 4) * 5.0)  # large pre-squash means to stress the bounds
        samples = dist.sample()
        assert (samples >= low).all() and (samples <= high).all()

    def test_deterministic_output_is_squashed_mean(self) -> None:
        """deterministic_output() and the `mean` property should equal scale*tanh(mean)+offset."""
        low, high = -1.0, 1.0
        dist = SquashedTanhGaussianDistribution(output_dim=3, low=low, high=high)
        mlp_output = torch.tensor([[0.5, -1.0, 2.0]])
        dist.update(mlp_output)
        expected = torch.tanh(mlp_output)  # scale=1, offset=0
        assert torch.allclose(dist.deterministic_output(mlp_output), expected, atol=1e-6)
        assert torch.allclose(dist.mean, expected, atol=1e-6)
        assert torch.allclose(dist.as_deterministic_output_module()(mlp_output), expected, atol=1e-6)

    def test_log_prob_matches_manual_change_of_variables(self) -> None:
        """log_prob should equal Normal.log_prob(u) minus the tanh+scale Jacobian, summed over dims."""
        dim, scale = 3, 2.0  # low=-2, high=2 -> scale=2, offset=0
        dist = SquashedTanhGaussianDistribution(output_dim=dim, init_noise_std=0.7, low=-2.0, high=2.0)
        mean = torch.randn(5, dim)
        dist.update(mean)
        std = dist.std
        # Pick pre-squash points, map through the squash, and compare log_prob to a manual computation.
        u = torch.randn(5, dim)
        actions = scale * torch.tanh(u) + 0.0
        base = torch.distributions.Normal(mean, std).log_prob(u).sum(dim=-1)
        jac = (torch.log(1.0 - torch.tanh(u) ** 2) + math.log(scale)).sum(dim=-1)
        expected = base - jac
        assert torch.allclose(dist.log_prob(actions), expected, atol=1e-4)

    def test_sample_and_log_prob_consistent(self) -> None:
        """sample_and_log_prob's log-prob should match log_prob() re-evaluated on the returned action."""
        dist = SquashedTanhGaussianDistribution(output_dim=4, init_noise_std=0.5)
        dist.update(torch.randn(16, 4))
        action, logp = dist.sample_and_log_prob()
        assert action.shape == (16, 4) and logp.shape == (16,)
        assert torch.allclose(logp, dist.log_prob(action), atol=1e-3)
        assert torch.isfinite(logp).all()

    def test_reparameterized_gradient_flows_to_mean_and_std(self) -> None:
        """sample_and_log_prob is reparameterized: gradients flow to the pre-squash mean and the log-std param."""
        dim = 3
        dist = SquashedTanhGaussianDistribution(output_dim=dim, init_noise_std=1.0, learn_std=True)
        mean = torch.randn(8, dim, requires_grad=True)
        dist.update(mean)
        action, logp = dist.sample_and_log_prob()
        # A loss on the action itself must reach the mean (only possible via rsample).
        action.sum().backward(retain_graph=True)
        assert mean.grad is not None and not torch.all(mean.grad == 0)
        dist.log_std_param.grad = None
        logp.sum().backward()
        assert dist.log_std_param.grad is not None and not torch.all(dist.log_std_param.grad == 0)

    def test_log_std_clamped(self) -> None:
        """The std should be clamped in log-space to [exp(log_std_min), exp(log_std_max)]."""
        dim = 2
        dist_hi = SquashedTanhGaussianDistribution(output_dim=dim, init_noise_std=100.0, log_std_max=1.0)
        dist_hi.update(torch.zeros(1, dim))
        assert torch.allclose(dist_hi.std, torch.full((1, dim), math.exp(1.0)), atol=1e-5)
        dist_lo = SquashedTanhGaussianDistribution(output_dim=dim, init_noise_std=1e-6, log_std_min=-3.0)
        dist_lo.update(torch.zeros(1, dim))
        assert torch.allclose(dist_lo.std, torch.full((1, dim), math.exp(-3.0)), atol=1e-5)

    def test_learn_std_false_freezes(self) -> None:
        """learn_std=False should freeze the log-std parameter (no gradient)."""
        dim = 3
        dist = SquashedTanhGaussianDistribution(output_dim=dim, init_noise_std=0.4, learn_std=False)
        assert dist.log_std_param.requires_grad is False
        mean = torch.randn(4, dim, requires_grad=True)  # grad path through the mean so backward has a target
        dist.update(mean)
        _, logp = dist.sample_and_log_prob()
        logp.sum().backward()
        assert dist.log_std_param.grad is None
        assert torch.allclose(dist.log_std_param, torch.log(torch.full((dim,), 0.4)), atol=1e-6)


class TestBetaDistribution:
    """Tests for ``BetaDistribution``."""

    def test_alpha_beta_greater_than_one(self) -> None:
        """After update(), alpha and beta should both be strictly greater than 1."""
        dist = BetaDistribution(output_dim=4)
        dist.update(torch.randn(8, 2, 4))
        assert (dist._alpha > 1.0).all()
        assert (dist._beta > 1.0).all()

    def test_samples_within_action_range(self) -> None:
        """Samples should lie within the specified action_range."""
        dist = BetaDistribution(output_dim=4, action_range=(-1.0, 1.0))
        dist.update(torch.randn(64, 2, 4))
        samples = dist.sample()
        assert (samples >= -1.0).all() and (samples <= 1.0).all()

    def test_log_prob_unit_range_matches_torch(self) -> None:
        """With action_range (0, 1), log_prob should match torch.distributions.Beta directly."""
        dim = 3
        dist = BetaDistribution(output_dim=dim, action_range=(0.0, 1.0))
        mlp_output = torch.randn(4, 2, dim)
        dist.update(mlp_output)

        samples = dist.sample().clamp(1e-6, 1.0 - 1e-6)
        expected = torch.distributions.Beta(dist._alpha, dist._beta).log_prob(samples).sum(dim=-1)
        assert torch.allclose(dist.log_prob(samples), expected, atol=1e-5)

    def test_log_prob_jacobian_correction(self) -> None:
        """log_prob with a scaled action_range should differ from the unscaled case by log(scale) per dimension."""
        dim = 3
        scale = 2.0
        dist_unit = BetaDistribution(output_dim=dim, action_range=(0.0, 1.0))
        dist_scaled = BetaDistribution(output_dim=dim, action_range=(0.0, scale))

        mlp_output = torch.randn(4, 2, dim)
        dist_unit.update(mlp_output)
        dist_scaled.update(mlp_output)

        unit_samples = dist_unit.sample().clamp(1e-6, 1.0 - 1e-6)
        scaled_samples = unit_samples * scale

        lp_unit = dist_unit.log_prob(unit_samples)
        lp_scaled = dist_scaled.log_prob(scaled_samples)
        # Change-of-variables: log p(y) = log p(x) - dim * log(scale)
        assert torch.allclose(lp_unit - dim * math.log(scale), lp_scaled, atol=1e-5)

    def test_deterministic_output_matches_module(self) -> None:
        """as_deterministic_output_module() should produce the same result as deterministic_output()."""
        dist = BetaDistribution(output_dim=4, action_range=(-1.0, 1.0))
        mlp_output = torch.randn(8, 2, 4)
        assert torch.allclose(dist.deterministic_output(mlp_output), dist.as_deterministic_output_module()(mlp_output))

    def test_log_prob_gradient_flows(self) -> None:
        """log_prob should allow gradient flow back through the MLP output."""
        dim = 3
        dist = BetaDistribution(output_dim=dim)
        mlp_output = torch.randn(4, 2, dim, requires_grad=True)
        dist.update(mlp_output)
        samples = dist.sample().detach()
        dist.log_prob(samples).sum().backward()
        assert mlp_output.grad is not None
        assert not torch.all(mlp_output.grad == 0)
