from __future__ import annotations

from typing import TYPE_CHECKING

from gmm_divergence._core._numeric import gaussian_kl_from_factors
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distributions._gaussian import Gaussian


def kl_closed_form(p: Gaussian, q: Gaussian) -> DivergenceResult:
    r"""Compute the closed-form KL divergence between two Gaussian distributions.

    Computes

    $$
    D_{\mathrm{KL}}(p \| q) = \frac{1}{2}
    \left[
        \mathrm{tr}(\Sigma_q^{-1} \Sigma_p)
        + (\mu_q - \mu_p)^T \Sigma_q^{-1} (\mu_q - \mu_p)
        - k
        + \log\left(\frac{\det(\Sigma_q)}{\det(\Sigma_p)}\right)
    \right]
    $$

    where $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q,
    \Sigma_q)$ are Gaussian distributions.

    Parameters
    ----------
    p : Gaussian
        Reference Gaussian distribution.
    q : Gaussian
        Approximating Gaussian distribution.

    Returns
    -------
    DivergenceResult
        Result object containing the analytically computed KL divergence.

    Raises
    ------
    ValueError
        If `p` and `q` have different dimensionality.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    if p.dim != q.dim:
        msg = "Both distributions must have the same dimensionality."
        raise ValueError(msg)
    value = gaussian_kl_from_factors(
        p.covariance, q.mean - p.mean, q.chol(), p.log_det(), q.log_det()
    )
    return DivergenceResult(value=float(value), method="closed_form")
