from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence._core._types import Covariances, FloatArray
    from gmm_divergence.distributions._typing import GaussianDistribution


def kl_unscented(p: GaussianDistribution, q: GaussianDistribution, /) -> DivergenceResult:
    r"""Estimate KL divergence using unscented sigma points.

    Estimates

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{X \sim p}
    \left[
        \log p(X) - \log q(X)
    \right]
    $$

    by evaluating the log-density ratio at deterministic sigma points generated
    from the Gaussian components of `p`. Each component contributes twice its
    feature dimension in points with equal within-component integration weights.

    For component $k$ of the reference density, let its mean be $\mu_{p,k}$ and its
    covariance eigenpairs be $(\lambda_{k,j}, u_{k,j})$, for $j=1,\ldots,d$.
    The points are

    $$
    x_{k,j}^{\pm}=\mu_{p,k}\pm\sqrt{d\lambda_{k,j}}\,u_{k,j},
    $$

    and the implemented approximation is

    $$
    \widehat D_{\mathrm{KL}}(p\|q)
    =\sum_{k=1}^{K_p}\frac{\pi_{p,k}}{2d}
    \sum_{j=1}^{d}\sum_{s\in\{+,-\}}
    \left[\log p(x_{k,j}^{s})-\log q(x_{k,j}^{s})\right].
    $$

    Here $K_p$ is the reference component count, $\pi_{p,k}$ its component
    probabilities, and $d$ the feature dimension. The densities evaluated at
    each point are the full mixtures, not individual component densities.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution used to generate the sigma points.
    q : Gaussian or GaussianMixture
        Approximating distribution evaluated at the sigma points.

    Returns
    -------
    DivergenceResult
        Result object containing the unscented estimate of the KL divergence.
        n_evaluations counts sigma points. n_samples and monte_carlo are None
        because the method draws no random observations.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    weights, means, covariances = p.component_arrays()
    sigma_points = _sigma_points(means, covariances)
    n_components, n_sigma, n_features = sigma_points.shape
    flat_points = sigma_points.reshape(-1, n_features)
    log_p = p.logpdf(flat_points).reshape(n_components, n_sigma)
    log_q = q.logpdf(flat_points).reshape(n_components, n_sigma)
    log_ratio = log_p - log_q
    # log_ratio[a, k] = log(f(x_{a,k}) / g(x_{a,k}))
    value = np.sum(weights[:, None] * log_ratio) / n_sigma
    return DivergenceResult(
        value=float(value), divergence="kl", n_evaluations=n_components * n_sigma
    )


def _sigma_points(means: FloatArray, covariances: Covariances, /) -> FloatArray:
    """Compute sigma points for Gaussian components."""
    n_components, n_features = means.shape
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    eigenvalues = np.maximum(eigenvalues, 0)
    scales = np.sqrt(n_features * eigenvalues)  # (K, d)
    offsets = eigenvectors * scales[:, None, :]  # (K, d, d)
    sigma_points = np.empty((n_components, 2 * n_features, n_features), dtype=np.float64)
    sigma_points[:, :n_features, :] = means[:, None, :] + offsets.transpose(0, 2, 1)
    sigma_points[:, n_features:, :] = means[:, None, :] - offsets.transpose(0, 2, 1)
    return sigma_points
