from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.estimators._components import as_gaussian_components
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import FloatArray


def kl_unscented(p: Gaussian | GaussianMixture, q: Distribution, /) -> DivergenceResult:
    r"""Estimate KL divergence using unscented sigma points.

    Estimates

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{x \sim p}
    \left[
        \log p(x) - \log q(x)
    \right]
    $$

    by evaluating the log-density ratio at deterministic sigma points generated
    from the Gaussian components of `p`. This is similar to monte carlo, but
    with deterministic points chosen to capture the mean and covariance
    structure of `p` more effectively.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution used to generate the sigma points.
    q : Distribution
        Approximating distribution evaluated at the sigma points.

    Returns
    -------
    DivergenceResult
        Result object containing the unscented estimate of the KL divergence.
        The reported number of samples equals the total number of sigma points.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    weights, means, covariances = as_gaussian_components(p)
    sigma_points = _sigma_points(means, covariances)
    n_components, n_sigma, dim = sigma_points.shape
    flat_points = sigma_points.reshape(-1, dim)
    log_p = p.logpdf(flat_points).reshape(n_components, n_sigma)
    log_q = q.logpdf(flat_points).reshape(n_components, n_sigma)
    log_ratio = log_p - log_q
    # log_ratio[a, k] = log(f(x_{a,k}) / g(x_{a,k}))
    value = np.sum(weights[:, None] * log_ratio) / n_sigma
    return DivergenceResult(
        value=float(value), method="unscented", num_samples=n_components * n_sigma
    )


def _sigma_points(means: FloatArray, covariances: FloatArray, /) -> FloatArray:
    """Compute sigma points for Gaussian components."""
    n_components, dim = means.shape
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    eigenvalues = np.maximum(eigenvalues, 0)
    scales = np.sqrt(dim * eigenvalues)  # (K, d)
    offsets = eigenvectors * scales[:, None, :]  # (K, d, d)
    sigma_points = np.empty(
        (n_components, 2 * dim, dim),
        dtype=np.float64,
    )
    sigma_points[:, :dim, :] = means[:, None, :] + offsets.transpose(0, 2, 1)
    sigma_points[:, dim:, :] = means[:, None, :] - offsets.transpose(0, 2, 1)
    return sigma_points
