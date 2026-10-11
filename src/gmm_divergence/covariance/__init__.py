"""Public covariance regularizers and epsilon heuristics."""

from gmm_divergence.covariance._api import regularize_covariance
from gmm_divergence.covariance._epsilon import (
    EpsilonHeuristic,
    EpsilonSpec,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
    estimate_epsilon,
)
from gmm_divergence.covariance._options import (
    CovarianceRegularizer,
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    IsotropicShrinkage,
    LowRank,
)

__all__ = [
    "CovarianceRegularizer",
    "DiagonalLoading",
    "DiagonalShrinkage",
    "EigenvalueClipping",
    "EpsilonHeuristic",
    "EpsilonSpec",
    "IsotropicShrinkage",
    "LowRank",
    "RelativeToTrace",
    "ResidualVariance",
    "TargetConditionNumber",
    "estimate_epsilon",
    "regularize_covariance",
]
