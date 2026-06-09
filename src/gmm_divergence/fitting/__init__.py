"""Public fitting API."""

from gmm_divergence.fitting._api import fit_mixture_weights, prune_mixture
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitObjective,
    FitParameterization,
    ForwardKL,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    WeightFitMethod,
    WeightFitObjective,
)
from gmm_divergence.fitting._selector import (
    CandidateSelection,
    CandidateSelector,
    KLQuantileSelector,
    KLThresholdSelector,
    KLToleranceSelector,
    TopKSelection,
)

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "FitObjective",
    "FitParameterization",
    "ForwardKL",
    "KLQuantileSelector",
    "KLThresholdSelector",
    "KLToleranceSelector",
    "MomentMatching",
    "ReverseKL",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "TopKSelection",
    "WeightFitMethod",
    "WeightFitObjective",
    "fit_mixture_weights",
    "prune_mixture",
]
