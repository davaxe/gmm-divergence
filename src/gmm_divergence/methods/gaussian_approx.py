from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.gmm.model import GaussianMixture, PrecisionT

Approximation = Literal["nearest", "moment_matching"]


def kl_gaussian_approximation(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    approximation: Approximation = "moment_matching",
) -> DivergenceResult:
    """Estimate KL divergence by approximating each mixture with a single Gaussian."""
    if approximation == "moment_matching":
        mean_p, cov_p = _moment_matching_approximation(p)
        mean_q, cov_q = _moment_matching_approximation(q)
        return DivergenceResult(
            value=_kl_single_gaussian(mean_p, cov_p, mean_q, cov_q),
            method="moment_matching",
        )
    if approximation == "nearest":
        return DivergenceResult(
            value=_nearest_component_approximation(p, q), method="nearest_component"
        )
    msg = f"Unsupported approximation method: {approximation}"
    raise ValueError(msg)


def _moment_matching_approximation(
    gmm: GaussianMixture[PrecisionT],
) -> tuple[npt.NDArray[PrecisionT], npt.NDArray[PrecisionT]]:
    """Compute mean and covariance of the Gaussian approximation."""
    weights = gmm.weights / np.sum(gmm.weights)
    mean = np.sum(weights[:, None] * gmm.means, axis=0)
    if gmm.covariance_type == "diag":
        cov = np.sum(
            weights[:, None] * (gmm.covariances + (gmm.means - mean) * (gmm.means - mean).T),
            axis=0,
        )
    elif gmm.covariance_type == "full":
        cov = np.sum(
            weights[:, None, None]
            * (gmm.covariances + (gmm.means - mean)[:, :, None] * (gmm.means - mean)[:, None, :]),
            axis=0,
        )
    else:
        msg = f"Unsupported covariance type: {gmm.covariance_type}"
        raise ValueError(msg)
    return mean, cov


def _nearest_component_approximation(
    p: GaussianMixture[PrecisionT], q: GaussianMixture[PrecisionT], /
) -> float:
    """Find the component of q closest to p in KL divergence."""
    min_kl = np.inf
    for a in range(q.n_components):
        for b in range(p.n_components):
            kl = _kl_single_gaussian(
                mean_p=p.means[b],
                cov_p=p.covariances[b]
                if p.covariance_type == "full"
                else np.diag(p.covariances[b]),
                mean_q=q.means[a],
                cov_q=q.covariances[a]
                if q.covariance_type == "full"
                else np.diag(q.covariances[a]),
            )
            min_kl = min(min_kl, kl)
    return min_kl


def _kl_single_gaussian(
    mean_p: npt.NDArray[PrecisionT],
    cov_p: npt.NDArray[PrecisionT],
    mean_q: npt.NDArray[PrecisionT],
    cov_q: npt.NDArray[PrecisionT],
) -> float:
    """Compute KL divergence between two Gaussians."""
    dim = mean_p.shape[0]
    inv_cov_q = np.linalg.inv(cov_q)
    trace_term = np.trace(inv_cov_q @ cov_p)
    mean_diff = mean_q - mean_p
    quadratic_term = mean_diff.T @ inv_cov_q @ mean_diff
    log_det_ratio = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    return 0.5 * (trace_term + quadratic_term - dim + log_det_ratio)
