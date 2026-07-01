"""Public fitting API."""

from gmm_divergence.fitting._api import fit_mixture_weights, prune_mixture
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitObjective,
    FitParameterization,
    ForwardKL,
    JensenShannon,
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
    QuantileSelector,
    ThresholdSelector,
    ToleranceSelector,
    TopKSelector,
)

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "FitObjective",
    "FitParameterization",
    "ForwardKL",
    "JensenShannon",
    "MomentMatching",
    "QuantileSelector",
    "ReverseKL",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "ThresholdSelector",
    "ToleranceSelector",
    "TopKSelector",
    "WeightFitMethod",
    "WeightFitObjective",
    "fit_mixture_weights",
    "prune_mixture",
]
