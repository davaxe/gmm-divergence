from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from gmm_divergence.distribution import Gaussian
from gmm_divergence.estimators._components import as_gaussian_components
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import FloatArray

Approximation = Literal["nearest", "moment_matching"]


def kl_gaussian_approximation(
    p: Gaussian | GaussianMixture,
    q: Gaussian | GaussianMixture,
    /,
    approximation: Approximation = "moment_matching",
) -> DivergenceResult:
    """Estimate KL divergence by approximating each distribution with a Gaussian."""
    if approximation == "moment_matching":
        mean_p, cov_p = _moment_matching_approximation(p)
        mean_q, cov_q = _moment_matching_approximation(q)
        return DivergenceResult(
            value=_kl_single_gaussian(mean_p, cov_p, mean_q, cov_q),
            method="moment_matching",
        )
    return DivergenceResult(
        value=_nearest_component_approximation(p, q), method="nearest_component"
    )


def _moment_matching_approximation(
    distribution: Gaussian | GaussianMixture,
) -> tuple[FloatArray, FloatArray]:
    """Compute mean and covariance of the Gaussian approximation."""
    if isinstance(distribution, Gaussian):
        return distribution.mean, distribution.covariance

    weights = distribution.weights / np.sum(distribution.weights)
    mean = np.sum(weights[:, None] * distribution.means, axis=0)
    mean_delta = distribution.means - mean
    cov = np.sum(
        weights[:, None, None]
        * (distribution.covariances + mean_delta[:, :, None] * mean_delta[:, None, :]),
        axis=0,
    )
    return mean, cov


def _nearest_component_approximation(
    p: Gaussian | GaussianMixture,
    q: Gaussian | GaussianMixture,
    /,
) -> float:
    """Find the component of q closest to p in KL divergence."""
    _weights_p, means_p, covariances_p = as_gaussian_components(p)
    _weights_q, means_q, covariances_q = as_gaussian_components(q)
    min_kl = np.inf
    for a in range(means_q.shape[0]):
        for b in range(means_p.shape[0]):
            kl = _kl_single_gaussian(
                mean_p=means_p[b],
                cov_p=covariances_p[b],
                mean_q=means_q[a],
                cov_q=covariances_q[a],
            )
            min_kl = min(min_kl, kl)
    return min_kl


def _kl_single_gaussian(
    mean_p: FloatArray,
    cov_p: FloatArray,
    mean_q: FloatArray,
    cov_q: FloatArray,
) -> float:
    """Compute KL divergence between two Gaussians."""
    dim = mean_p.shape[0]
    inv_cov_q = np.linalg.inv(cov_q)
    trace_term = np.trace(inv_cov_q @ cov_p)
    mean_diff = mean_q - mean_p
    quadratic_term = mean_diff.T @ inv_cov_q @ mean_diff
    log_det_ratio = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    return 0.5 * (trace_term + quadratic_term - dim + log_det_ratio)
