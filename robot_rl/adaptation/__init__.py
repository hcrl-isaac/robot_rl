"""Few-shot dynamics adaptation of frozen forward-backward models (LSTD correction + linear-bandit latent choice)."""

from .bandit import (
    ReturnRidge,
    candidate_values,
    perturbed_candidates,
    safety_mask,
    select,
    vec_features,
)
from .features import (
    BiasFeatures,
    FeatureMap,
    ForwardFeatures,
    PCAFeatures,
    PCAPhysicalFeatures,
    make_feature_map,
)
from .lstd import LSTDEnsemble, LSTDFit, LSTDStats, episode_stats, relative_residual, solve, td_fraction_explained

__all__ = [
    "BiasFeatures",
    "FeatureMap",
    "ForwardFeatures",
    "LSTDEnsemble",
    "LSTDFit",
    "LSTDStats",
    "PCAFeatures",
    "PCAPhysicalFeatures",
    "ReturnRidge",
    "candidate_values",
    "episode_stats",
    "make_feature_map",
    "perturbed_candidates",
    "relative_residual",
    "safety_mask",
    "select",
    "solve",
    "td_fraction_explained",
    "vec_features",
]
