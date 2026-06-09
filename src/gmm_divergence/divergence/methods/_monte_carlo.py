from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._sampling import resolve_samples
from gmm_divergence.results import DivergenceResult, MonteCarloStatistics

if TYPE_CHECKING:
    from gmm_divergence.distributions._base import Distribution


def kl_monte_carlo(
    p: Distribution,
    q: Distribution,
    /,
    *,
    sampling: npt.ArrayLike | int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> DivergenceResult:
    r"""Estimate KL divergence using Monte Carlo sampling.

    Estimates

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{x \sim p}
    \left[
        \log p(x) - \log q(x)
    \right]
    $$

    using samples from `p`.

    Parameters
    ----------
    p : Distribution
        Reference distribution to sample from.
    q : Distribution
        Approximating distribution evaluated at the sampled points.
    sampling : int or array-like, default=10_000
        Number of samples drawn from `p`, or precomputed samples from `p`.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed used when sampling is required.

    Returns
    -------
    DivergenceResult
        Result object containing the Monte Carlo estimate of the KL divergence.

    References
    ----------
    - Hershey, John R., and Peder A. Olsen. "Approximating the Kullback
        Leibler divergence between Gaussian mixture models." 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4.
        IEEE, 2007.
    """
    sampling = resolve_samples(p, sampling, rng)

    pointwise_kl = np.asarray(p.logpdf(sampling) - q.logpdf(sampling), dtype=np.float64)
    value = float(np.mean(pointwise_kl))
    num_samples = int(pointwise_kl.shape[0])

    if num_samples > 1:
        sample_variance = float(np.var(pointwise_kl, ddof=1))
        standard_error = float(np.sqrt(sample_variance / num_samples))
    else:
        sample_variance = float("nan")
        standard_error = float("nan")

    return DivergenceResult(
        value=value,
        method="monte_carlo",
        num_samples=num_samples,
        monte_carlo_stats=MonteCarloStatistics(
            sample_mean=value,
            sample_variance=sample_variance,
            standard_error=standard_error,
            effective_sample_size=num_samples,
        ),
    )
