# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .cnn_rnn_model import CNNRNNModel
from .cref_model import CrefModel
from .fuse_model import FuseModel, ResidualFuseModel
from .mlp_model import MLPModel
from .rnn_model import RNNModel
from .txl_model import TXLModel

__all__ = [
    "CNNModel",
    "CNNRNNModel",
    "CrefModel",
    "FuseModel",
    "MLPModel",
    "RNNModel",
    "ResidualFuseModel",
    "TXLModel",
]
