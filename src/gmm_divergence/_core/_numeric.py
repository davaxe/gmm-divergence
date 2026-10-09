from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    from gmm_divergence._core._types import Covariances, FloatArray


def logsumexp(a: FloatArray, axis: int = -1) -> FloatArray:
    """Compute log-sum-exp."""
    return np.asarray(scipy.special.logsumexp(a, axis=axis), dtype=np.float64)


def symmetrize(array: FloatArray) -> FloatArray:
    """Average a matrix or batch with its transpose over the last two axes."""
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def pairwise_gaussian_kl(
    p_means: FloatArray, p_covariances: Covariances, q_means: FloatArray, q_covariances: Covariances
) -> FloatArray:
    r"""Compute $D_{\mathrm{KL}}(p_i \| q_j)$ for all Gaussian component pairs.

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
        $D_{\mathrm{KL}}(p_i \| q_j)$.
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

    chol_p = np.linalg.cholesky(p_covariances).astype(np.float64, copy=False)
    chol_q = np.linalg.cholesky(q_covariances).astype(np.float64, copy=False)
    return gaussian_kl_from_factors(
        p_covariances[:, None],
        q_means[None, :] - p_means[:, None],
        chol_q[None, :],
        logdet_from_cholesky(chol_p)[:, None],
        logdet_from_cholesky(chol_q)[None, :],
    )


def logdet_from_cholesky(chol: FloatArray) -> FloatArray:
    """Return scalar or batched log-determinants from Cholesky factors."""
    return np.asarray(
        2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1), dtype=np.float64
    )


def gaussian_kl_from_factors(
    p_covariance: FloatArray,
    mean_difference: FloatArray,
    q_chol: FloatArray,
    p_logdet: FloatArray | float,
    q_logdet: FloatArray | float,
) -> FloatArray:
    """Evaluate Gaussian KL using broadcast-compatible arrays and cached factors."""
    whitened_covariance = np.linalg.solve(q_chol, p_covariance)
    solved_covariance = np.linalg.solve(q_chol.swapaxes(-1, -2), whitened_covariance)
    trace = np.trace(solved_covariance, axis1=-2, axis2=-1)
    whitened_difference = np.linalg.solve(q_chol, mean_difference[..., None])[..., 0]
    quadratic = np.sum(whitened_difference**2, axis=-1)
    return np.asarray(
        0.5 * (trace + quadratic - mean_difference.shape[-1] + q_logdet - p_logdet),
        dtype=np.float64,
    )
