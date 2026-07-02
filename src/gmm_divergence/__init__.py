"""Curated public API for Gaussian-mixture divergence and fitting."""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import covariance, distributions, divergence, fitting, sampling
from gmm_divergence.distributions import Gaussian, GaussianMixture, combine_gaussians
from gmm_divergence.divergence import (
    component_kl_matrix,
    estimate_divergence,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.fitting import fit_mixture_weights, prune_mixture
from gmm_divergence.results import DivergenceResult, FitResult

__all__ = [
    "DivergenceResult",
    "FitResult",
    "Gaussian",
    "GaussianMixture",
    "combine_gaussians",
    "component_kl_matrix",
    "covariance",
    "distributions",
    "divergence",
    "estimate_divergence",
    "fit_mixture_weights",
    "fitting",
    "jensen_shannon_divergence",
    "kl_divergence",
    "prune_mixture",
    "sampling",
    "symmetric_kl_divergence",
]

try:  # noqa: RUF067
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
