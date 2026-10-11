"""Public fitting API."""

from gmm_divergence.fitting._api import (
    fit_categorical_weights,
    fit_gaussian_weights,
    prepare_gaussian_weight_fit,
    prune_mixture,
)
from gmm_divergence.fitting._component_statistics import ComponentStatistics, component_statistics
from gmm_divergence.fitting._gaussian import PreparedGaussianWeightFit
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitObjective,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    Optimizer,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.fitting._selector import (
    BidirectionalKLScore,
    CandidateScorer,
    CandidateSelection,
    CandidateSelector,
    ForwardKLScore,
    QuantileSelector,
    ReverseKLScore,
    ThresholdSelector,
    ToleranceSelector,
    TopKSelector,
    rank_candidates,
    score_candidates,
)
from gmm_divergence.fitting._simplex import SimplexOptimizationResult
from gmm_divergence.results import CategoricalWeightFitResult, GaussianWeightFitResult

__all__ = [
    "BidirectionalKL",
    "BidirectionalKLScore",
    "CandidateScorer",
    "CandidateSelection",
    "CandidateSelector",
    "CategoricalWeightFitResult",
    "ComponentStatistics",
    "FitObjective",
    "ForwardKL",
    "ForwardKLScore",
    "GaussianWeightFitResult",
    "JensenShannon",
    "MomentMatching",
    "Optimizer",
    "PreparedGaussianWeightFit",
    "QuantileSelector",
    "ReverseKL",
    "ReverseKLScore",
    "SimplexOptimizationResult",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "ThresholdSelector",
    "ToleranceSelector",
    "TopKSelector",
    "component_statistics",
    "fit_categorical_weights",
    "fit_gaussian_weights",
    "prepare_gaussian_weight_fit",
    "prune_mixture",
    "rank_candidates",
    "score_candidates",
]
