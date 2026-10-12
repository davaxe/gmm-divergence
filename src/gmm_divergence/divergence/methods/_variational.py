# pyright: reportPrivateUsage=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import logsumexp, pairwise_gaussian_kl
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture


def kl_variational(
    p: Gaussian | GaussianMixture, q: Gaussian | GaussianMixture, /
) -> DivergenceResult:
    r"""Estimate KL divergence using a variational method.

    For Gaussian component densities $p_k$ and $q_j$, reference probabilities
    $\pi_{p,k}$, comparison probabilities $\pi_{q,j}$, and component counts $K_p$, $K_q$,
    the implemented approximation is

    $$
    \widehat D_{\mathrm{KL}}^{\mathrm{var}}(p\|q)
    =\sum_{k=1}^{K_p}\pi_{p,k}
    \log\frac{\sum_{j=1}^{K_p}\pi_{p,j}
    \exp[-D_{\mathrm{KL}}(p_k\|p_j)]}
    {\sum_{j=1}^{K_q}\pi_{q,j}\exp[-D_{\mathrm{KL}}(p_k\|q_j)]}.
    $$

    Pairwise component KL values are exact Gaussian divergences in nats.
    The mixture result is a deterministic approximation, not a certified bound;
    the hat denotes approximation, without implying Monte Carlo sampling.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution.
    q : Gaussian or GaussianMixture
        Approximating distribution.

    Returns
    -------
    DivergenceResult
        Result object containing the variational estimate of the KL divergence.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    weights_p, means_p, covariances_p = p.component_arrays()
    weights_q, means_q, covariances_q = q.component_arrays()
    kl_pp = pairwise_gaussian_kl(means_p, covariances_p, means_p, covariances_p)
    kl_pq = pairwise_gaussian_kl(means_p, covariances_p, means_q, covariances_q)
    log_weights_p = np.log(weights_p)
    log_weights_q = np.log(weights_q)
    numerator = logsumexp(log_weights_p[None, :] - kl_pp, axis=1)
    denominator = logsumexp(log_weights_q[None, :] - kl_pq, axis=1)
    return DivergenceResult(value=np.dot(weights_p, numerator - denominator), divergence="kl")
