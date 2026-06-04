from importlib.metadata import version

from gmm_divergence.distribution.combine import CombinedGaussianMixture, MixtureMapping
from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.distribution.gmm import GaussianMixture
from gmm_divergence.divergence import kl_divergence
from gmm_divergence.fit import fit_mixture_weights
from gmm_divergence.formatting import format_kl_fit_result
from gmm_divergence.results import DivergenceResult, KLFitResult

__all__ = [
    "CombinedGaussianMixture",
    "DivergenceResult",
    "Gaussian",
    "GaussianMixture",
    "KLFitResult",
    "MixtureMapping",
    "fit_mixture_weights",
    "format_kl_fit_result",
    "kl_divergence",
]
__version__: str = version("gmm-divergence")
