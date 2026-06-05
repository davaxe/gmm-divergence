from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import pairwise_kl
from gmm_divergence.distributions import Gaussian
from gmm_divergence.divergence.methods._closed_form import kl_closed_form
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distributions._base import GaussianFamily
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.divergence._options import Approximation


def kl_gaussian_approximation(
    p: Gaussian | GaussianMixture,
    q: Gaussian | GaussianMixture,
    /,
    approximation: Approximation = "moment_matching",
) -> DivergenceResult:
    r"""Estimate KL divergence using Gaussian approximations.

    Approximates `p` and `q` as Gaussian distributions and computes

    $$
    D_{\mathrm{KL}}(p \| q).
    $$

    This is a crude but computationally efficient method that can be used as a
    quick heuristic or as a baseline for more sophisticated estimators.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution.
    q : Gaussian or GaussianMixture
        Approximating distribution.
    approximation : {"nearest", "moment_matching"}
        Approximation strategy to use.

        - `"moment_matching"`:
          Replace each distribution by a single Gaussian with matching mean and
          covariance, then compute the Gaussian KL divergence.
        - `"nearest"`:
          Compute pairwise KL divergences between Gaussian components and return
          the smallest value.

    Returns
    -------
    DivergenceResult
        Result object containing the approximate KL divergence and the
        approximation method used.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    if approximation == "moment_matching":
        p = _moment_matching_approximation(p)
        q = _moment_matching_approximation(q)
        return DivergenceResult(value=kl_closed_form(p, q).value, method="moment_matching")
    return DivergenceResult(value=_nearest_component_approximation(p, q), method="nearest")


def _moment_matching_approximation(distribution: Gaussian | GaussianMixture) -> Gaussian:
    """Compute mean and covariance of the Gaussian approximation."""
    if isinstance(distribution, Gaussian):
        return distribution

    mean = np.sum(distribution.weights[:, None] * distribution.means, axis=0)
    mean_delta = distribution.means - mean
    cov = np.sum(
        distribution.weights[:, None, None]
        * (distribution.covariances + mean_delta[:, :, None] * mean_delta[:, None, :]),
        axis=0,
    )
    return Gaussian(mean=mean, covariance=0.5 * (cov + cov.T))


def _nearest_component_approximation(p: GaussianFamily, q: GaussianFamily, /) -> float:
    """Find the component of q closest to p in KL divergence."""
    _, means_p, covariances_p = p.component_arrays()
    _, means_q, covariances_q = q.component_arrays()
    means = np.concatenate([means_p, means_q], axis=0)
    covariances = np.concatenate([covariances_p, covariances_q], axis=0)
    kl_matrix = pairwise_kl(means, covariances)
    return np.min(kl_matrix)
