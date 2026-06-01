from importlib.metadata import version

from gmm_divergence.divergence import kl_divergence
from gmm_divergence.gmm.model import GaussianMixture
from gmm_divergence.results import DivergenceResult

__all__ = ["DivergenceResult", "GaussianMixture", "kl_divergence"]
__version__: str = version("gmm-divergence")
