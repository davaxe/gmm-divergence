"""Moment-matched Gaussian approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.divergence.methods._closed_form import kl_closed_form

if TYPE_CHECKING:
    from gmm_divergence.distributions._typing import GaussianDistribution
    from gmm_divergence.results import DivergenceResult


def kl_gaussian_approximation(p: GaussianDistribution, q: GaussianDistribution) -> DivergenceResult:
    r"""Compute Gaussian KL using the aggregate moments of each input.

    With aggregate means $\mu_p$, $\mu_q$ and covariances $\Sigma_p$, $\Sigma_q$,
    the approximation is

    $$
    \widehat D_{\mathrm{KL}}(p\|q)
    =D_{\mathrm{KL}}\!\left(
    \mathcal{N}(\mu_p,\Sigma_p)\,\|\,\mathcal{N}(\mu_q,\Sigma_q)\right).
    $$

    These moments include the input mixtures' component probabilities and
    between-component mean variation. Distinct mixtures with identical first
    and second moments receive zero here; the result is not a bound on mixture KL.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Reference and comparison distributions with equal feature dimensions.

    Returns
    -------
    DivergenceResult
        Closed-form KL in nats between the moment-matched Gaussians. This is
        generally an approximation to mixture KL, without sampling diagnostics.
    """
    return kl_closed_form(Gaussian(p.mean, p.covariance), Gaussian(q.mean, q.covariance))
