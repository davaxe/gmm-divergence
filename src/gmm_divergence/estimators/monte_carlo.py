from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence.results import DivergenceResult
from gmm_divergence.utils import resolve_samples

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import Distribution


def kl_monte_carlo(
    p: Distribution,
    q: Distribution,
    /,
    *,
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
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
    num_samples : int, default=10_000
        Number of samples drawn from `p` when `samples` is not provided.
    samples : array-like, optional
        Precomputed samples from `p`. If provided, no new samples are drawn.
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
    samples = resolve_samples(p, num_samples, samples, rng)
    return DivergenceResult(
        value=float(np.mean(p.logpdf(samples) - q.logpdf(samples))),
        method="monte_carlo",
        num_samples=num_samples,
    )
