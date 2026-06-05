from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    from gmm_divergence._core._types import Covariances, FloatArray


def logsumexp(a: FloatArray, axis: int = -1) -> FloatArray:
    """Compute log-sum-exp."""
    return np.asarray(scipy.special.logsumexp(a, axis=axis), dtype=np.float64)


def pairwise_kl(means: FloatArray, covariances: Covariances) -> FloatArray:
    """Compute pairwise KL divergence between Gaussian components."""
    dim = means.shape
    inv_cov = np.linalg.inv(covariances)
    _, logdet = np.linalg.slogdet(covariances)
    logdet_term = logdet[None, :] - logdet[:, None]
    trace_term = np.einsum("jkl,ilk->ij", inv_cov, covariances)
    diff = means[None, :, :] - means[:, None, :]
    quad_term = np.einsum("ijd,jde,ije->ij", diff, inv_cov, diff)
    return 0.5 * (logdet_term - dim + trace_term + quad_term)
