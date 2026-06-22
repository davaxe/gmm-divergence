from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    from gmm_divergence._core._types import Covariances, FloatArray


def logsumexp(a: FloatArray, axis: int = -1) -> FloatArray:
    """Compute log-sum-exp."""
    return np.asarray(scipy.special.logsumexp(a, axis=axis), dtype=np.float64)


def pairwise_gaussian_kl(
    p_means: FloatArray, p_covariances: Covariances, q_means: FloatArray, q_covariances: Covariances
) -> FloatArray:
    """Compute KL(N_p_i || N_q_j) for all Gaussian component pairs.

    Parameters
    ----------
    p_means : FloatArray
        Array of shape (n_p, d).
    p_covariances : Covariances
        Array of shape (n_p, d, d).
    q_means : FloatArray
        Array of shape (n_q, d).
    q_covariances : Covariances
        Array of shape (n_q, d, d).

    Returns
    -------
    FloatArray
        Matrix of shape (n_p, n_q), where entry (i, j) is
        KL(N_p_i || N_q_j).
    """
    if p_means.ndim != 2:
        msg = f"p_means must have shape (n_p, d), got {p_means.shape}."
        raise ValueError(msg)

    if q_means.ndim != 2:
        msg = f"q_means must have shape (n_q, d), got {q_means.shape}."
        raise ValueError(msg)

    if p_means.shape[1] != q_means.shape[1]:
        msg = (
            "p_means and q_means must have the same feature dimension, "
            f"got {p_means.shape[1]} and {q_means.shape[1]}."
        )
        raise ValueError(msg)

    d = p_means.shape[1]

    if p_covariances.shape != (p_means.shape[0], d, d):
        msg = (
            "p_covariances must have shape "
            f"({p_means.shape[0]}, {d}, {d}), got {p_covariances.shape}."
        )
        raise ValueError(msg)

    if q_covariances.shape != (q_means.shape[0], d, d):
        msg = (
            "q_covariances must have shape "
            f"({q_means.shape[0]}, {d}, {d}), got {q_covariances.shape}."
        )
        raise ValueError(msg)

    chol_p = np.linalg.cholesky(p_covariances)
    chol_q = np.linalg.cholesky(q_covariances)
    logdet_p = 2.0 * np.sum(np.log(np.diagonal(chol_p, axis1=1, axis2=2)), axis=1)
    logdet_q = 2.0 * np.sum(np.log(np.diagonal(chol_q, axis1=1, axis2=2)), axis=1)
    logdet_term = logdet_q[None, :] - logdet_p[:, None]
    whitened_covariances = np.linalg.solve(chol_q[:, None, :, :], p_covariances[None, :, :, :])
    solved_covariances = np.linalg.solve(
        np.swapaxes(chol_q, 1, 2)[:, None, :, :], whitened_covariances
    )
    trace_term = np.trace(solved_covariances, axis1=2, axis2=3).T
    diff = q_means[None, :, :] - p_means[:, None, :]
    whitened_diff = np.linalg.solve(chol_q[None, :, :, :], diff[..., None])[..., 0]
    quad_term = np.einsum("ija,ija->ij", whitened_diff, whitened_diff)
    return 0.5 * (logdet_term - d + trace_term + quad_term)
