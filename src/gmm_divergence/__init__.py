"""Tools for Gaussian-mixture KL divergence and mixture fitting.

The top-level package exports the main distribution classes, KL divergence
entrypoint, and mixture-weight fitting helper:

- ``Gaussian`` and ``GaussianMixture`` define the supported distributions.
- ``kl_divergence`` estimates or computes ``D_KL(p || q)``.
- ``fit_mixture_weights`` fits weights for fixed candidate mixtures.
- ``CombinedGaussianMixture`` and ``MixtureMapping`` describe mixtures built
  from multiple source distributions.
"""

from importlib.metadata import version

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
    "fit_mixture_weights",
    "kl_divergence",
    "prune_mixture",
]
__version__: str = version("gmm-divergence")
