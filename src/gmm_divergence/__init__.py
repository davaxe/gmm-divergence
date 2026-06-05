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

from gmm_divergence.distribution.combine import (
    CombinedGaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.distribution.gmm import GaussianMixture
from gmm_divergence.divergence import (
    ClosedForm,
    GaussianApproximation,
    MonteCarlo,
    Unscented,
    kl_divergence,
)
from gmm_divergence.fit import fit_mixture_weights
from gmm_divergence.fitting.config import (
    BidirectionalKL,
    ForwardKL,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
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
    "combine_gaussians",
    "fit_mixture_weights",
    "kl_divergence",
]
__version__: str = version("gmm-divergence")
