"""Public distribution API."""

from gmm_divergence.distributions._combine import (
    CombinedGaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture
from gmm_divergence.distributions._typing import GaussianLike

__all__ = [
    "CombinedGaussianMixture",
    "Gaussian",
    "GaussianLike",
    "GaussianMixture",
    "MixtureMapping",
    "combine_gaussians",
]
