"""Public package API for Gaussian-mixture divergence and fitting.

The top-level package provides two complementary access patterns:

- A curated convenience layer for the most common classes and functions.
- Stable domain namespaces: ``distributions``, ``divergence``, and ``fitting``.
"""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import distributions, divergence, fitting
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
    "DivergenceResult",
    "ForwardKL",
    "Gaussian",
    "GaussianApproximation",
    "GaussianMixture",
    "KLFitResult",
    "MixtureMapping",
    "MomentMatching",
    "MonteCarlo",
    "ReverseKL",
    "SimplexSLSQP",
    "SoftmaxLBFGSB",
    "Unscented",
    "Variational",
    "combine_gaussians",
    "distributions",
    "divergence",
    "fit_mixture_weights",
    "fitting",
    "kl_divergence",
    "prune_mixture",
]

try:  # noqa: RUF067
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
