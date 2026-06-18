"""Public covariance regularization API."""

from gmm_divergence.covariance._api import regularize_covariance
from gmm_divergence.covariance._epsilon import (
    EpsilonMethod,
    EpsilonMethodName,
    EpsilonSpec,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
    estimate_epsilon,
)
from gmm_divergence.covariance._options import (
    CovarianceRegularizationMethod,
    CovarianceRegularizer,
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    LinearShrinkage,
    LowRank,
)
from gmm_divergence.covariance._regularize import (
    diagonal_loading,
    diagonal_shrinkage,
    eigenvalue_clipping,
    linear_shrinkage,
    lowrank,
)

__all__ = [
    "CovarianceRegularizationMethod",
    "CovarianceRegularizer",
    "DiagonalLoading",
    "DiagonalShrinkage",
    "EigenvalueClipping",
    "EpsilonMethod",
    "EpsilonMethodName",
    "EpsilonSpec",
    "LinearShrinkage",
    "LowRank",
    "RelativeToTrace",
    "ResidualVariance",
    "TargetConditionNumber",
    "diagonal_loading",
    "diagonal_shrinkage",
    "eigenvalue_clipping",
    "estimate_epsilon",
    "linear_shrinkage",
    "lowrank",
    "regularize_covariance",
]
