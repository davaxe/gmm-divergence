from importlib.metadata import version

from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.distribution.gmm import GaussianMixture
from gmm_divergence.divergence import kl_divergence
from gmm_divergence.results import DivergenceResult

__all__ = ["DivergenceResult", "Gaussian", "GaussianMixture", "kl_divergence"]
__version__: str = version("gmm-divergence")
