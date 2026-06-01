from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from gmm_divergence.gmm.model import GaussianMixture, PrecisionT


def _logsumexp(a: npt.NDArray[PrecisionT], axis: int = -1) -> npt.NDArray[PrecisionT]:
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(
        a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)),
        axis=axis,
    )


def gmm_logpdf_full(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the log-density of a full-covariance Gaussian mixture."""
    if gmm.covariance_type != "full":
        msg = f"Expected full covariance type, got {gmm.covariance_type}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=gmm.dtype)
    weights = gmm.weights / np.sum(gmm.weights)

    n_samples, n_features = x.shape
    n_components = weights.shape[0]

    log_weights = np.log(weights)
    chol = gmm.chol()
    log_probs = np.empty((n_samples, n_components), dtype=gmm.dtype)
    constant = n_features * np.log(2.0 * np.pi)

    for k in range(n_components):
        diff = x - gmm.means[k]
        y = np.linalg.solve(chol[k], diff.T)
        mahal = np.sum(y * y, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[k])))
        log_gaussian = -0.5 * (constant + log_det + mahal)
        log_probs[:, k] = log_weights[k] + log_gaussian

    return _logsumexp(log_probs, axis=1)


def gmm_pdf_full(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the density of a full-covariance Gaussian mixture."""
    return np.exp(gmm_logpdf_full(x, gmm))


def gmm_logpdf_diag(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the log-density of a diagonal-covariance Gaussian mixture."""
    if gmm.covariance_type != "diag":
        msg = f"Expected diagonal covariance type, got {gmm.covariance_type}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=gmm.dtype)
    weights = gmm.weights / np.sum(gmm.weights)
    means = gmm.means
    variances = gmm.covariances

    _n_samples, n_features = x.shape
    log_weights = np.log(weights)

    diff = x[:, None, :] - means[None, :, :]
    mahal = np.sum((diff * diff) / variances[None, :, :], axis=2)
    log_det = np.sum(np.log(variances), axis=1)
    constant = n_features * np.log(2.0 * np.pi)
    log_gaussian = -0.5 * (constant + log_det[None, :] + mahal)
    log_probs = log_weights[None, :] + log_gaussian

    return _logsumexp(log_probs, axis=1)


def gmm_pdf_diag(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the density of a diagonal-covariance Gaussian mixture."""
    return np.exp(gmm_logpdf_diag(x, gmm))
