"""Curated public API for Gaussian-mixture divergence and fitting."""

from importlib.metadata import PackageNotFoundError, version

from gmm_divergence import covariance, distributions, divergence, fitting, sampling
from gmm_divergence.distributions import Gaussian, GaussianMixture, combine_distributions
from gmm_divergence.divergence import (
    aligned_joint_kl_divergence,
    categorical_kl_divergence,
    component_kl_matrix,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.fitting import (
    fit_categorical_weights,
    fit_gaussian_weights,
    prepare_gaussian_weight_fit,
    prune_mixture,
)
from gmm_divergence.results import (
    CategoricalWeightFitResult,
    DivergenceResult,
    GaussianWeightFitResult,
)

__all__ = [
    "CategoricalWeightFitResult",
    "DivergenceResult",
    "Gaussian",
    "GaussianMixture",
    "GaussianWeightFitResult",
    "aligned_joint_kl_divergence",
    "categorical_kl_divergence",
    "combine_distributions",
    "component_kl_matrix",
    "covariance",
    "distributions",
    "divergence",
    "fit_categorical_weights",
    "fit_gaussian_weights",
    "fitting",
    "jensen_shannon_divergence",
    "kl_divergence",
    "prepare_gaussian_weight_fit",
    "prune_mixture",
    "sampling",
    "symmetric_kl_divergence",
]

try:  # ruff: ignore[non-empty-init-module]
    __version__: str = version("gmm-divergence")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
