"""Public package API for Gaussian-mixture divergence and fitting.

The top-level package provides two complementary access patterns:

- A curated convenience layer for the most common classes and functions.
- Stable domain namespaces: `distributions`, `divergence`, `fitting`, and `covariance`.
"""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import covariance, distributions, divergence, fitting
from gmm_divergence._core._sampling import (
    DrawSamples,
    SampleBatchSpec,
    SampleSpec,
    StratifiedSamples,
    UseSampleBatches,
    UseSamples,
)
from gmm_divergence.covariance import (
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    LinearShrinkage,
    LowRank,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
    estimate_epsilon,
    regularize_covariance,
)
from gmm_divergence.distributions import (
    CombinedGaussianMixture,
    Gaussian,
    GaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.divergence import (
    ClosedForm,
    GaussianApproximation,
    KLMethod,
    MonteCarlo,
    Unscented,
    Variational,
    component_kl_matrix,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.fitting import (
    BidirectionalKL,
    CandidateSelection,
    CandidateSelector,
    FitObjective,
    FitParameterization,
    ForwardKL,
    JensenShannon,
    KLQuantileSelector,
    KLThresholdSelector,
    KLToleranceSelector,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    TopKSelector,
    WeightFitMethod,
    WeightFitObjective,
    fit_mixture_weights,
    prune_mixture,
)
from gmm_divergence.results import DivergenceResult, KLFitResult

__all__ = [
    "BidirectionalKL",
    "CandidateSelection",
    "CandidateSelector",
    "ClosedForm",
    "CombinedGaussianMixture",
    "DiagonalLoading",
    "DiagonalShrinkage",
    "DivergenceResult",
    "DrawSamples",
    "EigenvalueClipping",
    "FitObjective",
    "FitParameterization",
    "ForwardKL",
    "Gaussian",
    "GaussianApproximation",
    "GaussianMixture",
    "JensenShannon",
    "KLFitResult",
    "KLMethod",
    "KLQuantileSelector",
    "KLThresholdSelector",
    "KLToleranceSelector",
    "LinearShrinkage",
    "LowRank",
    "MixtureMapping",
    "MomentMatching",
    "MonteCarlo",
    "RelativeToTrace",
    "ResidualVariance",
    "ReverseKL",
    "SampleBatchSpec",
    "SampleSpec",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "StratifiedSamples",
    "TargetConditionNumber",
    "TopKSelector",
    "Unscented",
    "UseSampleBatches",
    "UseSamples",
    "Variational",
    "WeightFitMethod",
    "WeightFitObjective",
    "combine_gaussians",
    "component_kl_matrix",
    "covariance",
    "distributions",
    "divergence",
    "estimate_epsilon",
    "fit_mixture_weights",
    "fitting",
    "jensen_shannon_divergence",
    "kl_divergence",
    "prune_mixture",
    "regularize_covariance",
    "symmetric_kl_divergence",
]

try:  # noqa: RUF067
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
