"""Public package API for Gaussian-mixture divergence and fitting.

The top-level package provides two complementary access patterns:

- A curated convenience layer for the most common classes and functions.
- Stable domain namespaces: `distributions`, `divergence`, `fitting`, and `covariance`.
"""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import covariance, distributions, divergence, fitting
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
    MonteCarlo,
    Unscented,
    Variational,
    kl_divergence,
)
from gmm_divergence.fitting import (
    BidirectionalKL,
    ForwardKL,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    fit_mixture_weights,
    prune_mixture,
)
from gmm_divergence.results import DivergenceResult, KLFitResult

__all__ = [
    "BidirectionalKL",
    "ClosedForm",
    "CombinedGaussianMixture",
    "DiagonalLoading",
    "DiagonalShrinkage",
    "DivergenceResult",
    "EigenvalueClipping",
    "ForwardKL",
    "Gaussian",
    "GaussianApproximation",
    "GaussianMixture",
    "KLFitResult",
    "LinearShrinkage",
    "LowRank",
    "MixtureMapping",
    "MomentMatching",
    "MonteCarlo",
    "RelativeToTrace",
    "ResidualVariance",
    "ReverseKL",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "TargetConditionNumber",
    "Unscented",
    "Variational",
    "combine_gaussians",
    "covariance",
    "distributions",
    "divergence",
    "estimate_epsilon",
    "fit_mixture_weights",
    "fitting",
    "kl_divergence",
    "prune_mixture",
    "regularize_covariance",
]

try:  # noqa: RUF067
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
