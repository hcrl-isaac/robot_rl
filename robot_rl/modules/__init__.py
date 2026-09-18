# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .cnn import CNN
from .dict_module import DictModule
from .distribution import (
    BetaDistribution,
    Distribution,
    GaussianDistribution,
    HeteroscedasticGaussianDistribution,
    SquashedTanhGaussianDistribution,
    TangentStepVonMisesFisherDistribution,
    TruncatedGaussianDistribution,
    VonMisesFisherDistribution,
)
from .mlp import MLP
from .normalization import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
    ExponentialMovingAverageNormalization,
)
from .parallel import ParallelLayerNorm, ParallelLinear
from .residual import ResMLP
from .rnn import RNN, HiddenState
from .target_network import TargetNetwork
from .tcn import TCNEncoder
from .transformer_xl import TransformerXL

__all__ = [
    "CNN",
    "MLP",
    "RNN",
    "BetaDistribution",
    "DictModule",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "ExponentialMovingAverageNormalization",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
    "ParallelLayerNorm",
    "ParallelLinear",
    "ResMLP",
    "SquashedTanhGaussianDistribution",
    "TangentStepVonMisesFisherDistribution",
    "TCNEncoder",
    "TargetNetwork",
    "TransformerXL",
    "TruncatedGaussianDistribution",
    "VonMisesFisherDistribution",
]
