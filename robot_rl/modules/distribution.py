# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
from typing import Any


class Distribution(nn.Module):
    """Base class for distribution modules.

    Distribution modules encapsulate the stochastic output of a neural model. They define the output structure expected
    from the MLP, manage learnable distribution parameters, and provide methods for sampling, log probability
    computation, and entropy calculation.

    Subclasses must implement all abstract methods and properties to define a specific distribution type.
    """

    def __init__(self, output_dim: int) -> None:
        """Initialize the distribution module.

        Args:
            output_dim: Dimension of the action/output space.
        """
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution parameters given the MLP output.

        Args:
            mlp_output: Raw output from the MLP.
        """
        raise NotImplementedError

    def sample(self, **kwargs: Any) -> torch.Tensor:
        """Sample from the distribution.

        Returns:
            Sampled values.
        """
        raise NotImplementedError

    def sample_and_log_prob(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample and return the sample together with its log probability, summed over the last dimension.

        The default computes the two separately; distributions with a reparameterized sample whose log-prob is
        best evaluated from the pre-sample latent (e.g. squashed Gaussians) should override this so both are
        derived from the *same* draw.

        Returns:
            A tuple ``(sample, log_prob)``.
        """
        sample = self.sample(**kwargs)
        return sample, self.log_prob(sample)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the deterministic (mean) output from the raw MLP output.

        Args:
            mlp_output: Raw output from the MLP.

        Returns:
            The deterministic output (typically the distribution mean).
        """
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the deterministic output from the MLP output."""
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Return the input dimension required by the distribution."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation (or spread measure) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the distribution, summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the distribution parameters as a tuple of tensors.

        These are the distribution-specific parameters needed to reconstruct the distribution (e.g., mean and std for
        Gaussian, alpha and beta for Beta). They are stored during rollouts and used for KL divergence computation.
        """
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given outputs, summed over the last dimension.

        Args:
            outputs: Values to compute the log probability for.

        Returns:
            Log probability summed over the last dimension.
        """
        raise NotImplementedError

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the KL divergence KL(old || new) between two distributions of this type.

        The KL divergence measures how the old distribution diverges from the new distribution.
        This is used for adaptive learning rate scheduling in policy optimization.

        Args:
            old_params: Parameters of the old distribution (as returned by :attr:`params`).
            new_params: Parameters of the new distribution (as returned by :attr:`params`).

        Returns:
            KL divergence summed over the last dimension.
        """
        raise NotImplementedError

    def export_std(self) -> torch.Tensor | None:
        """Return the input-independent std of a plain Gaussian for export, or ``None`` for any other distribution."""
        return None

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize distribution-specific weights in the MLP.

        This is called after MLP creation to set up any special weight initialization
        required by the distribution (e.g., initializing std head weights).

        Args:
            mlp: The MLP module whose weights may need initialization.
        """
        pass


class GaussianDistribution(Distribution):
    """Gaussian distribution module with state-independent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation can be a learnable parameter or a constant. It can be parameterized in either "scalar" space or
    "log" space and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
            learn_std: Whether the standard deviation should be learnable. If False, it will be fixed to `init_std`.
        """
        super().__init__(output_dim)
        self.std_type = std_type

        # Learnable std parameters
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)), requires_grad=learn_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _clamped_std(self) -> torch.Tensor:
        """Return the std parameter clamped to its configured range."""
        if self.std_type == "scalar":
            return self.std_param.clamp(self.std_range[0], self.std_range[1])
        return torch.exp(self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1]))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        self._distribution = Normal(mlp_output, self._clamped_std())

    def export_std(self) -> torch.Tensor | None:
        """Return the clamped std as a detached constant for export."""
        return self._clamped_std().detach().clone()

    def sample(self, **kwargs: Any) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the mean from the MLP output."""
        return _IdentityDeterministicOutput()

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Gaussian distribution."""
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Gaussian distribution."""
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Gaussian distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current Gaussian distribution."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Gaussian, summed over the last dimension."""
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Gaussian distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian distribution module with state-dependent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation is output by the MLP alongside the mean, making it state-dependent. It can be parameterized in
    either "scalar" space or "log" space, and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (used to initialize the MLP's std head bias).
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = torch.clamp(std, self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.clamp(log_std, self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def export_std(self) -> torch.Tensor | None:
        """Return ``None``: the std is state-dependent, so it cannot be exported as a constant."""
        return None

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that extracts the mean from the MLP output."""
        return _FirstSliceDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the mean and the second is the standard deviation (or log standard deviation).
        """
        return [2, self.output_dim]

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the std head weights in the MLP."""
        # Initialize weights and biases for the std portion of the last layer
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)  # type: ignore
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_std_log)  # type: ignore


class TruncatedGaussianDistribution(GaussianDistribution):
    """Truncated Gaussian (Normal) distribution module with state-independent standard deviation.

    This distribution parametrizes actions using a multivariate Gaussian with diagonal covariance. The standard
    deviation can be a fixed or learnable parameter that is independent of the model input. It can be parameterized in
    either "scalar" space (directly) or "log" space. The distribution output is truncated, but gradients are preserved.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        learn_std: bool = True,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the truncated Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
            learn_std: Whether the std is learnable. If False, it is held fixed at ``init_std``.
            low: Lower bound for Gaussian output.
            high: Upper bound for Gaussian output.
            eps: A small tolerance to subtract from the upper bound and add to the lower bound.
        """
        Distribution.__init__(self, output_dim)
        self.std_type = std_type

        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)), requires_grad=learn_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

        self._low = low
        self._high = high
        self._eps = eps
        # Affine factors for the tanh-shaped mean.
        self._mean_scale = (high - low) / 2
        self._mean_offset = (high + low) / 2

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the truncated Gaussian distribution from MLP output, with the mean shaped by tanh into (low, high)."""
        mean = self._mean_scale * torch.tanh(mlp_output) + self._mean_offset
        if self.std_type == "scalar":
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std)

    def export_std(self) -> torch.Tensor | None:
        """Return ``None``: a truncated Gaussian is not described by ``(mean, std)`` alone."""
        return None

    def sample(self, std_clip: float | None = None) -> torch.Tensor:
        """Sample from the Gaussian distribution.

        Uses :meth:`Normal.rsample` so the sampled action carries gradient w.r.t. the actor's parameters via ``mean``.
        """
        x = self._distribution.rsample()  # type: ignore
        if std_clip is not None:
            x = torch.clamp(x, self.mean - std_clip, self.mean + std_clip)
        return self._clamp(x)

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp sampled x to range [low + eps, high - eps] while preserving original gradient."""
        clamped_x = torch.clamp(x, self._low + self._eps, self._high - self._eps)
        return x - x.detach() + clamped_x.detach()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output, shaped by tanh into (low, high)."""
        return self._mean_scale * torch.tanh(mlp_output) + self._mean_offset

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that returns the MLP output shaped by tanh into ``[low, high]``."""
        return _TanhScaledDeterministicOutput(self._mean_scale, self._mean_offset)


class BetaDistribution(Distribution):
    """Beta distribution module for bounded action spaces.

    This distribution parameterizes stochastic outputs using a Beta distribution, which naturally constrains samples
    to [0, 1]. Samples are linearly rescaled to ``action_range``, which defaults to ``(-1.0, 1.0)``.

    The MLP must output a tensor of shape ``[..., 2, output_dim]``, where the first slice along the second-to-last
    dimension contains the raw alpha parameters and the second contains the raw beta parameters. Both are passed
    through ``Softplus + 1`` to ensure they are strictly greater than 1, which guarantees a unimodal distribution.
    """

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initialize the Beta distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            action_range: Interval ``(min, max)`` to which Beta samples in ``[0, 1]`` are linearly rescaled.
                Defaults to ``(-1.0, 1.0)``.
        """
        super().__init__(output_dim)

        # Compute scaling and offset for rescaling samples
        self.action_range = action_range
        self._range_scale = action_range[1] - action_range[0]
        self._range_offset = action_range[0]
        self._log_range_scale = np.log(self._range_scale)

        self._distribution: Beta | None = None
        self._alpha: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None

        # Disable args validation for speedup
        Beta.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Beta distribution from MLP output."""
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        self._alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        self._beta = torch.nn.functional.softplus(beta_raw) + 1.0
        self._distribution = Beta(self._alpha, self._beta)

    def sample(self, **kwargs: Any) -> torch.Tensor:
        """Sample from the Beta distribution and rescale to ``action_range``."""
        return self._distribution.sample() * self._range_scale + self._range_offset  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output and rescale to ``action_range``."""
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self._range_scale + self._range_offset

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that computes the mean from the MLP output."""
        return _BetaDeterministicOutput(self._range_scale, self._range_offset)

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the raw alpha parameter and the second is the raw beta parameter.
        """
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Beta distribution, rescaled to ``action_range``."""
        return (self._alpha / (self._alpha + self._beta)) * self._range_scale + self._range_offset  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Beta distribution, rescaled to ``action_range``."""
        return self._distribution.stddev * self._range_scale  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Beta distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return ``(alpha, beta)`` of the current Beta distribution."""
        return (self._alpha, self._beta)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Beta distribution, summed over the last dimension.

        Outputs are unscaled from ``action_range`` back to ``[0, 1]`` before computing the log probability.
        The Jacobian correction for the linear rescaling is included.
        """
        unscaled = (outputs - self._range_offset) / self._range_scale
        unscaled = unscaled.clamp(1e-6, 1.0 - 1e-6)
        # Jacobian correction: log p(y) = log p(x) - log(scale), where y = x * scale + offset
        return (self._distribution.log_prob(unscaled) - self._log_range_scale).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Beta distributions."""
        old_alpha, old_beta = old_params
        new_alpha, new_beta = new_params
        return torch.distributions.kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the beta-parameter head weights to zero for a near-uniform initial distribution."""
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        torch.nn.init.zeros_(mlp[-2].bias[self.output_dim :])  # type: ignore


class VonMisesFisherDistribution(Distribution):
    r"""von Mises-Fisher distribution on the unit hypersphere :math:`S^{d-1}`.

    The directional analogue of a Gaussian: a unit mean direction :math:`\hat{\mu}` and a scalar concentration
    :math:`\kappa \ge 0`; samples are unit vectors. Entropy is bounded above by the uniform-sphere entropy
    (:math:`\kappa \to 0`) and vanishes as :math:`\kappa \to \infty`. ``init_std`` sets the initial
    concentration :math:`\kappa_0 = 1/\text{init\_std}^2` and :attr:`std` reports :math:`1/\sqrt{\kappa}`, so a
    smaller std means more concentrated (matching Gaussian semantics). :meth:`sample_and_log_prob` is
    reparameterized for SAC-style updates; the :math:`\kappa`-dependence of the rejection acceptance probability
    is ignored.

    Reference:
        - Wood. "Simulation of the von Mises Fisher distribution." Communications in Statistics 23(1) (1994).
        - Davidson et al. "Hyperspherical Variational Auto-Encoders." arXiv preprint arXiv:1804.00891 (2018).
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        learn_std: bool = True,
        kappa_range: tuple[float, float] = (1e-2, 1e5),
        cf_extra_terms: int = 64,
    ) -> None:
        """Initialize the von Mises-Fisher distribution module.

        Args:
            output_dim: Dimension ``p`` of the ambient space (the sphere is ``S^{p-1}``). Must be even.
            init_std: Initial (asymptotic tangent) standard deviation; sets ``kappa = 1 / init_std**2``.
            learn_std: Whether the concentration is a learnable parameter. If False, it is fixed at its initial value.
            kappa_range: ``(min, max)`` clamp applied to the concentration for numerical stability.
            cf_extra_terms: Extra terms used to seed the backward Bessel-ratio recurrence (higher = more accurate).
        """
        super().__init__(output_dim)
        if output_dim % 2 != 0:
            raise ValueError(f"VonMisesFisherDistribution requires an even output_dim, got {output_dim}.")

        # Concentration is a single scalar, stored in log-space for positivity and a well-conditioned parameterization.
        init_kappa = 1.0 / max(init_std, 1e-6) ** 2
        self.log_kappa = nn.Parameter(torch.log(torch.tensor([init_kappa])), requires_grad=learn_std)
        self.kappa_range = (float(kappa_range[0]), float(kappa_range[1]))
        self._cf_extra = int(cf_extra_terms)

        # Current state (populated by update())
        self._mu: torch.Tensor | None = None  # [B, p] unit mean direction
        self._kappa: torch.Tensor | None = None  # scalar concentration
        self._log_norm: torch.Tensor | None = None  # log C_p(kappa)
        self._a_ratio: torch.Tensor | None = None  # A_p(kappa) = I_{p/2}(kappa) / I_{p/2-1}(kappa)

    def _bessel_terms(self, kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute ``(log C_p(kappa), A_p(kappa))`` for a scalar concentration, differentiably.

        Uses a downward recurrence for the Bessel ratios :math:`r_k = I_k(\kappa)/I_{k-1}(\kappa)`, which are bounded
        in ``(0, 1)`` and hence numerically stable. ``log I_{p/2-1}`` is then accumulated from
        :math:`\log I_0(\kappa) = \kappa + \log(\text{i0e}(\kappa))` and the log-ratios; everything is plain arithmetic
        so autograd yields the exact gradients (with :math:`\frac{d}{d\kappa}(-\log C_p) = A_p`). ``kappa`` is a single
        scalar, so the Python loop is cheap.
        """
        p = self.output_dim
        half = p // 2  # = p/2 ; A_p = r_{p/2} = ratios[half]
        nu = half - 1  # order of the normalizer's Bessel term, I_{p/2-1}
        m = half + self._cf_extra
        # Downward recurrence r_k = 1/(2k/kappa + r_{k+1}); seed r_{m+1} at the continued-fraction fixed
        # point (sqrt(1+s^2)-s, s=(m+1)/kappa) so it converges immediately for large kappa.
        ratios: list[torch.Tensor | None] = [None] * (half + 1)
        s = (m + 1) / kappa
        r = torch.sqrt(1.0 + s * s) - s
        for k in range(m, 0, -1):
            r = 1.0 / (2.0 * k / kappa + r)
            if k <= half:
                ratios[k] = r
        a_ratio = ratios[half]  # I_{p/2}/I_{p/2-1}
        # log I_{nu} = log I_0 + sum_{j=1}^{nu} log r_j, with log I_0 = kappa + log(i0e(kappa)).
        log_i_nu = kappa + torch.log(torch.special.i0e(kappa))
        for j in range(1, nu + 1):
            log_i_nu = log_i_nu + torch.log(ratios[j])
        log_norm = nu * torch.log(kappa) - half * math.log(2.0 * math.pi) - log_i_nu
        return log_norm.squeeze(), a_ratio.squeeze()

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution: mean direction = normalized MLP output, concentration = kappa param."""
        self._mu = torch.nn.functional.normalize(mlp_output, dim=-1)
        self._kappa = torch.exp(self.log_kappa).clamp(self.kappa_range[0], self.kappa_range[1])
        self._log_norm, self._a_ratio = self._bessel_terms(self._kappa)

    def sample(self, std_clip: float | None = None) -> torch.Tensor:
        """Sample unit vectors from the vMF distribution via Wood's rejection algorithm (no gradient).

        Supports arbitrary leading dims (e.g. meta-RL's [batch, seq, p] tensors): leading dims are flattened
        for the per-row rejection sampler, then restored. ``std_clip`` is accepted for interface compatibility
        but unused (vMF spread is set by the concentration).
        """
        with torch.no_grad():
            return self._rsample()

    def sample_and_log_prob(self, std_clip: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized ``(sample, log_prob)`` from a single draw.

        Gradients flow to the mean direction and concentration through both the sample and its log-prob;
        the acceptance probability's kappa-dependence is ignored (Davidson et al., 2018).
        """
        x = self._rsample()
        return x, self.log_prob(x)

    def _rsample(self) -> torch.Tensor:
        """Sample unit vectors, differentiably w.r.t. ``(mu, kappa)`` when grad is enabled.

        The rejection noise (accepted Beta draws) is sampled without gradient; the tangential component
        ``w = mu . x`` is then rebuilt from it as a differentiable function of the concentration, and the
        orthogonal-projection construction keeps the sample differentiable in the mean direction.
        """
        mu = self._mu  # [..., p]
        p = mu.shape[-1]
        lead = mu.shape[:-1]
        flat_mu = mu.reshape(-1, p)  # [N, p]
        n = flat_mu.shape[0]
        device = mu.device
        d = float(p - 1)
        kappa = self._kappa.squeeze()
        with torch.no_grad():
            z = self._sample_weight_noise(n, p, float(kappa.item()), device)  # [N]
        b = (-2.0 * kappa + torch.sqrt(4.0 * kappa * kappa + d * d)) / d
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        # Direction orthogonal to mu, uniform on the (p-2)-subsphere.
        v = torch.randn(n, p, device=device)
        v = v - (v * flat_mu).sum(dim=-1, keepdim=True) * flat_mu
        v = torch.nn.functional.normalize(v, dim=-1)
        x = w.unsqueeze(-1) * flat_mu + torch.sqrt((1.0 - w * w).clamp_min(1e-12)).unsqueeze(-1) * v
        x = torch.nn.functional.normalize(x, dim=-1)  # defensive re-normalization
        return x.reshape(*lead, p)

    def _sample_weight_noise(self, batch: int, p: int, kappa: float, device: torch.device) -> torch.Tensor:
        """Rejection-sample the Beta noise behind the tangential component ``w`` (Wood, 1994), with refill.

        Returns the accepted ``Beta(d/2, d/2)`` draws; rows still unaccepted after the retry cap (should not
        happen) keep the initial ``z = 0.5``, which maps exactly to the mode ``w = x0``.
        """
        d = float(p - 1)
        b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + d * d)) / d
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + d * math.log(max(1.0 - x0 * x0, 1e-300))
        beta = Beta(torch.tensor(d / 2.0, device=device), torch.tensor(d / 2.0, device=device))

        z = torch.full((batch,), 0.5, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        # Refill only the not-yet-accepted entries each round until all are accepted.
        for _ in range(100):
            todo = (~done).nonzero(as_tuple=True)[0]
            n = todo.numel()
            if n == 0:
                break
            z_prop = beta.sample((n,))
            w_prop = (1.0 - (1.0 + b) * z_prop) / (1.0 - (1.0 - b) * z_prop)
            u = torch.rand(n, device=device)
            accept = kappa * w_prop + d * torch.log((1.0 - x0 * w_prop).clamp_min(1e-300)) - c >= torch.log(u)
            acc_idx = todo[accept]
            z[acc_idx] = z_prop[accept]
            done[acc_idx] = True
        return z

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the unit mean direction (the deterministic action is the mode of the vMF)."""
        return torch.nn.functional.normalize(mlp_output, dim=-1)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that normalizes the MLP output to the unit mean direction."""
        return _NormalizeDeterministicOutput()

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution (the MLP outputs the raw mean direction)."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the unit mean direction."""
        return self._mu  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the (asymptotic tangent) standard deviation ``1/sqrt(kappa)``, broadcast to the mean's shape."""
        return torch.ones_like(self._mu) * self._kappa.rsqrt()  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        r"""Return the vMF entropy ``-log C_p(kappa) - kappa * A_p(kappa)``, broadcast over leading dims."""
        ent = -self._log_norm - self._kappa.squeeze() * self._a_ratio
        lead = self._mu.shape[:-1]  # type: ignore
        return ent.reshape((1,) * len(lead)).expand(lead)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return ``(mean_direction, kappa)``; kappa broadcast to ``[..., 1]`` over leading dims for KL."""
        lead = self._mu.shape[:-1]  # type: ignore
        kappa_b = self._kappa.reshape((1,) * len(lead) + (1,)).expand(*lead, 1)  # type: ignore
        return (self._mu, kappa_b)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute ``log C_p(kappa) + kappa * (mu . x)`` for unit-vector ``outputs``."""
        dot = (self._mu * outputs).sum(dim=-1)  # type: ignore
        return self._log_norm + self._kappa.squeeze() * dot

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Compute ``KL(old || new)`` between two vMF distributions.

        :math:`\mathrm{KL} = \log C_p(\kappa_0) - \log C_p(\kappa_1)
        + A_p(\kappa_0)\,(\kappa_0 - \kappa_1\,\hat{\mu}_0^\top\hat{\mu}_1)`.
        """
        mu0, kappa0_b = old_params
        mu1, kappa1_b = new_params
        kappa0 = kappa0_b.reshape(-1)[0]
        kappa1 = kappa1_b.reshape(-1)[0]
        log_norm0, a0 = self._bessel_terms(kappa0)
        log_norm1, _ = self._bessel_terms(kappa1)
        dot = (mu0 * mu1).sum(dim=-1)
        return (log_norm0 - log_norm1) + a0 * (kappa0 - kappa1 * dot)


class SquashedTanhGaussianDistribution(Distribution):
    r"""Squashed (tanh) diagonal Gaussian for SAC-style reparameterized, bounded actions.

    A diagonal Gaussian in *pre-squash* space (mean = MLP output, state-independent std) is passed through
    ``tanh`` and affinely mapped onto ``(low, high)``: :math:`a = \text{scale}\cdot\tanh(u) + \text{offset}`.
    Sampling is reparameterized (``rsample``) so gradients flow through the action, and :meth:`log_prob`
    includes the tanh change-of-variables (Jacobian) correction. The scalar std is learnable or fixed and is
    clamped in log-space to ``[log_std_min, log_std_max]``.

    Use :meth:`sample_and_log_prob` for the SAC actor update: it derives the action and its log-prob from the
    *same* pre-squash draw (numerically stable). :meth:`log_prob` on an arbitrary action inverts the squash
    with ``atanh`` and is less precise near the bounds.
    """

    def __init__(
        self,
        output_dim: int,
        init_noise_std: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        learn_std: bool = True,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the squashed-tanh Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_noise_std: Initial (pre-squash) standard deviation.
            log_std_min: Lower clamp on the log standard deviation.
            log_std_max: Upper clamp on the log standard deviation.
            learn_std: Whether the std is learnable. If False it is held fixed at ``init_noise_std``.
            low: Lower bound of the squashed action range.
            high: Upper bound of the squashed action range.
            eps: Small tolerance for the ``atanh`` inversion in :meth:`log_prob`.
        """
        super().__init__(output_dim)
        self.log_std_param = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)), requires_grad=learn_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self._eps = eps
        self._scale = (high - low) / 2
        self._offset = (high + low) / 2

        self._mean_pre: torch.Tensor | None = None  # pre-squash mean (MLP output)
        self._std: torch.Tensor | None = None
        self._distribution: Normal | None = None

        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the pre-squash Gaussian from MLP output (mean = output; std = clamped log-std param)."""
        self._mean_pre = mlp_output
        log_std = self.log_std_param.clamp(self.log_std_min, self.log_std_max)
        self._std = torch.exp(log_std).expand_as(mlp_output)
        self._distribution = Normal(self._mean_pre, self._std)

    def _squash(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        """Map a pre-squash sample into ``(low, high)`` via tanh + affine scaling."""
        return self._scale * torch.tanh(pre_tanh) + self._offset

    def _log_prob_from_pre_tanh(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        """Log-prob of the squashed action derived from its pre-squash sample, summed over the last dim.

        Applies the tanh Jacobian correction with the numerically stable identity
        ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))`` plus ``log(scale)`` for the affine map.
        """
        base = self._distribution.log_prob(pre_tanh)  # type: ignore
        jac = 2.0 * (math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        return (base - jac - math.log(self._scale)).sum(dim=-1)

    def sample(self, std_clip: float | None = None) -> torch.Tensor:
        """Reparameterized sample of a squashed action (``std_clip`` accepted for interface compat, unused)."""
        return self._squash(self._distribution.rsample())  # type: ignore

    def sample_and_log_prob(self, std_clip: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized ``(action, log_prob)`` from a single pre-squash draw (the SAC actor-update path)."""
        pre_tanh = self._distribution.rsample()  # type: ignore
        return self._squash(pre_tanh), self._log_prob_from_pre_tanh(pre_tanh)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Log-prob of an arbitrary squashed action, inverting the squash with ``atanh`` (less precise)."""
        t = ((outputs - self._offset) / self._scale).clamp(-1.0 + self._eps, 1.0 - self._eps)
        pre_tanh = torch.atanh(t)
        return self._log_prob_from_pre_tanh(pre_tanh)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the squashed mean action (the deterministic policy output)."""
        return self._squash(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that squashes the MLP output into ``[low, high]``."""
        return _TanhScaledDeterministicOutput(self._scale, self._offset)

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution (the MLP outputs the pre-squash mean)."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the squashed mean action."""
        return self._squash(self._mean_pre)  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the pre-squash standard deviation."""
        return self._std  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the *pre-squash* Gaussian entropy summed over the last dim.

        The true squashed entropy has no closed form (SAC uses ``-log_prob`` instead); this exposes the
        pre-squash Gaussian entropy so generic entropy consumers keep working.
        """
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return ``(pre_squash_mean, std)`` of the current distribution."""
        return (self._mean_pre, self._std)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL between the *pre-squash* Gaussians."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _NormalizeDeterministicOutput(nn.Module):
    """Exportable module that L2-normalizes the MLP output to a unit vector (vMF mean direction)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(mlp_output, dim=-1)


class _ClampedDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output clamped to ``[low, high]``."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.clamp(mlp_output, self.low, self.high)


class _TanhScaledDeterministicOutput(nn.Module):
    """Exportable module that returns ``scale * tanh(mlp_output) + offset``, bounded to ``(low, high)``."""

    def __init__(self, scale: float, offset: float) -> None:
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(mlp_output) + self.offset


class _FirstSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]


class _BetaDeterministicOutput(nn.Module):
    """Exportable module that computes the mean of the Beta distribution from the MLP output."""

    def __init__(self, range_scale: float, range_offset: float) -> None:
        super().__init__()
        self.range_scale = range_scale
        self.range_offset = range_offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self.range_scale + self.range_offset
