from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distribution import Gaussian
    from gmm_divergence.typing import PrecisionT


def kl_exact(p: Gaussian[PrecisionT], q: Gaussian[PrecisionT]) -> DivergenceResult:
    """Compute KL divergence exactly by summing over all component pairs."""
    dim = p.dim
    inv_cov_q = np.linalg.inv(q.covariance)
    trace_term = np.trace(inv_cov_q @ p.covariance)
    mean_diff = q.mean - p.mean
    quadratic_term = mean_diff.T @ inv_cov_q @ mean_diff
    log_det_ratio = np.log(np.linalg.det(q.covariance) / np.linalg.det(p.covariance))
    kl_value = 0.5 * (trace_term + quadratic_term - dim + log_det_ratio)
    return DivergenceResult(value=kl_value, method="exact")
