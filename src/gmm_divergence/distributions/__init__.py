"""Immutable Gaussian distributions and composition."""

from gmm_divergence.distributions._combine import combine_distributions
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mapping import MixtureMapping
from gmm_divergence.distributions._mixture import GaussianMixture, MixtureDiagnostics
from gmm_divergence.distributions._typing import GaussianDistribution

__all__ = [
    "Gaussian",
    "GaussianDistribution",
    "GaussianMixture",
    "MixtureDiagnostics",
    "MixtureMapping",
    "combine_distributions",
]
