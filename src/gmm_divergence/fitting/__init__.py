"""Public fitting API."""

from gmm_divergence.fitting._api import (
    fit_categorical_mixture_weights,
    fit_gaussian_mixture_weights,
    prepare_gaussian_mixture_fit,
    prune_mixture,
)
from gmm_divergence.fitting._component_statistics import ComponentStatistics, component_statistics
from gmm_divergence.fitting._gaussian import PreparedGaussianMixtureFit
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
from gmm_divergence.fitting._simplex import SimplexOptimizationResult

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "ComponentStatistics",
    "FitMethod",
    "FitObjective",
    "ForwardKL",
    "JensenShannon",
    "MomentMatching",
    "PreparedGaussianMixtureFit",
    "QuantileSelector",
    "ReverseKL",
    "SimplexOptimizationResult",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "ThresholdSelector",
    "ToleranceSelector",
    "TopKSelector",
    "component_statistics",
    "fit_categorical_mixture_weights",
    "fit_gaussian_mixture_weights",
    "prepare_gaussian_mixture_fit",
    "prune_mixture",
    "rank_candidates",
    "score_candidates",
]
