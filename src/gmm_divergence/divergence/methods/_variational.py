from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import logsumexp, pairwise_gaussian_kl
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture


def kl_variational(
    p: Gaussian | GaussianMixture, q: Gaussian | GaussianMixture, /
) -> DivergenceResult:
    weights_p, means_p, covariances_p = p.component_arrays()
    weights_q, means_q, covariances_q = q.component_arrays()
    kl_pp = pairwise_gaussian_kl(means_p, covariances_p, means_p, covariances_p)
    kl_pq = pairwise_gaussian_kl(means_p, covariances_p, means_q, covariances_q)
    log_weights_p = np.log(weights_p)
    log_weights_q = np.log(weights_q)
    numerator = logsumexp(log_weights_p[None, :] - kl_pp, axis=1)
    denominator = logsumexp(log_weights_q[None, :] - kl_pq, axis=1)
    return DivergenceResult(value=np.dot(weights_p, numerator - denominator), method="variational")
