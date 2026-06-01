from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.gmm.model import GaussianMixture, PrecisionT


def kl_unscented(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
) -> DivergenceResult:
    """Estimate KL divergence with unscented sigma points."""
    sigma_points = _sigma_points(p)
    n_components, n_sigma, dim = sigma_points.shape
    flat_points = sigma_points.reshape(-1, dim)
    log_p = p.logpdf(flat_points).reshape(n_components, n_sigma)
    log_q = q.logpdf(flat_points).reshape(n_components, n_sigma)
    log_ratio = log_p - log_q
    # log_ratio[a, k] = log(f(x_{a,k}) / g(x_{a,k}))
    value = np.sum(p.weights[:, None] * log_ratio) / n_sigma
    return DivergenceResult(
        value=float(value), method="unscented", num_samples=n_components * n_sigma
    )


def _sigma_points(gmm: GaussianMixture[PrecisionT], /) -> npt.NDArray[PrecisionT]:
    """Compute sigma points for each component of GMM."""
    means = gmm.means
    covariances = gmm.covariances
    n_components, dim = means.shape
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    eigenvalues = np.maximum(eigenvalues, 0)
    scales = np.sqrt(dim * eigenvalues)  # (K, d)
    offsets = eigenvectors * scales[:, None, :]  # (K, d, d)
    sigma_points = np.empty(
        (n_components, 2 * dim, dim),
        dtype=gmm.dtype,
    )
    sigma_points[:, :dim, :] = means[:, None, :] + offsets.transpose(0, 2, 1)
    sigma_points[:, dim:, :] = means[:, None, :] - offsets.transpose(0, 2, 1)
    return sigma_points
