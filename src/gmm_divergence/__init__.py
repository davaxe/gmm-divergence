"""Curated public API for Gaussian-mixture divergence and fitting."""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import covariance, distributions, divergence, fitting, sampling
from gmm_divergence.distributions import Gaussian, GaussianMixture, combine_gaussians
from gmm_divergence.divergence import (
    aligned_component_kl,
    component_kl_matrix,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.fitting import fit_gaussian_mixture_weights, prune_mixture
from gmm_divergence.results import (
    AlignedKLResult,
    DivergenceResult,
    GaussianMixtureFitResult,
    MonteCarloStatistics,
)

__all__ = [
    "AlignedKLResult",
    "DivergenceResult",
    "Gaussian",
    "GaussianMixture",
    "GaussianMixtureFitResult",
    "MonteCarloStatistics",
    "aligned_component_kl",
    "combine_gaussians",
    "component_kl_matrix",
    "covariance",
    "distributions",
    "divergence",
    "fit_gaussian_mixture_weights",
    "fitting",
    "jensen_shannon_divergence",
    "kl_divergence",
    "prune_mixture",
    "sampling",
    "symmetric_kl_divergence",
]

try:  # ruff: ignore[non-empty-init-module]
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
