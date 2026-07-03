"""Public fitting API."""

from gmm_divergence.fitting._api import fit_mixture_weights, prune_mixture
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitMethod,
    FitObjective,
    FitParameterization,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.fitting._selector import (
    CandidateSelection,
    CandidateSelector,
    QuantileSelector,
    ThresholdSelector,
    ToleranceSelector,
    TopKSelector,
    rank_candidates,
    score_candidates,
)

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "FitMethod",
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
    "fit_mixture_weights",
    "prune_mixture",
    "rank_candidates",
    "score_candidates",
]
