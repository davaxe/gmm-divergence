"""Moment-matched Gaussian approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.divergence.methods._closed_form import kl_closed_form

if TYPE_CHECKING:
    from gmm_divergence.distributions._typing import GaussianDistribution
    from gmm_divergence.results import DivergenceResult


def kl_gaussian_approximation(p: GaussianDistribution, q: GaussianDistribution) -> DivergenceResult:
    """Compute Gaussian KL using the aggregate mean and covariance of each input."""
    return kl_closed_form(Gaussian(p.mean, p.covariance), Gaussian(q.mean, q.covariance))
