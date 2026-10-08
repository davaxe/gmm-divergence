"""Public fitting API."""

from gmm_divergence.fitting._api import (
    fit_mixture_weights,
    prepare_mixture_weight_fit,
    prune_mixture,
)
from gmm_divergence.fitting._component_statistics import ComponentStatistics, component_statistics
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitMethod,
    FitObjective,
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
from gmm_divergence.fitting._weight_fitting import FitSolution, PreparedFit

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "ComponentStatistics",
    "FitMethod",
    "FitObjective",
    "FitSolution",
    "ForwardKL",
    "JensenShannon",
    "MomentMatching",
    "PreparedFit",
    "QuantileSelector",
    "ReverseKL",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "ThresholdSelector",
    "ToleranceSelector",
    "TopKSelector",
    "component_statistics",
    "fit_mixture_weights",
    "prepare_mixture_weight_fit",
    "prune_mixture",
    "rank_candidates",
    "score_candidates",
]
