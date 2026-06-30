from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._sampling import resolve_samples
from gmm_divergence._core._validation import as_positive_sample_count
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
    target_standard_error: float | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
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
        Number of samples drawn from `p`, or precomputed samples from `p`. When
        adaptive sampling is enabled, an integer value is the initial sample
        count.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed used when sampling is required.
    target_standard_error : float, optional
        If provided, draw additional batches until the Monte Carlo standard
        error is at or below this target, or until `max_samples` is reached.
    max_samples : int, optional
        Maximum sample count for adaptive sampling. Defaults to ten times the
        initial sample count.
    batch_size : int, optional
        Number of samples per additional adaptive batch. Defaults to the
        initial sample count.

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
    if target_standard_error is not None:
        return _kl_monte_carlo_adaptive(
            p,
            q,
            sampling=sampling,
            rng=rng,
            target_standard_error=target_standard_error,
            max_samples=max_samples,
            batch_size=batch_size,
        )

    sampling = resolve_samples(p, sampling, rng)
    pointwise_kl = _pointwise_kl(p, q, sampling)
    return _result_from_pointwise(pointwise_kl)


def _kl_monte_carlo_adaptive(
    p: Distribution,
    q: Distribution,
    /,
    *,
    sampling: npt.ArrayLike | int,
    rng: np.random.Generator | int | None,
    target_standard_error: float,
    max_samples: int | None,
    batch_size: int | None,
) -> DivergenceResult:
    if not isinstance(sampling, int) or isinstance(sampling, bool):
        msg = "Adaptive Monte Carlo requires sampling to be an integer sample count."
        raise TypeError(msg)
    if target_standard_error <= 0.0 or not np.isfinite(target_standard_error):
        msg = f"target_standard_error must be a positive finite value, got {target_standard_error}."
        raise ValueError(msg)

    initial_samples = as_positive_sample_count(sampling, name="sampling")
    max_samples = 10 * initial_samples if max_samples is None else max_samples
    batch_size = initial_samples if batch_size is None else batch_size
    max_samples = as_positive_sample_count(max_samples, name="max_samples")
    batch_size = as_positive_sample_count(batch_size, name="batch_size")
    if max_samples < initial_samples:
        msg = (
            "max_samples must be greater than or equal to the initial sampling count, "
            f"got max_samples={max_samples} and sampling={initial_samples}."
        )
        raise ValueError(msg)
    rng = np.random.default_rng(rng)

    stats = _RunningStats()
    while stats.n < max_samples:
        required_initial = max(0, initial_samples - stats.n)
        draw_count = min(max(required_initial, batch_size), max_samples - stats.n)
        samples = p.sample(draw_count, rng=rng)
        stats.update(_pointwise_kl(p, q, samples))

        if stats.n >= initial_samples and stats.n > 1:
            standard_error = np.sqrt(stats.sample_variance / stats.n)
            if standard_error <= target_standard_error:
                break

    return _result_from_stats(stats)


class _RunningStats:
    """Running mean and variance accumulator for pointwise estimates."""

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    @property
    def sample_variance(self) -> float:
        if self.n <= 1:
            return float("nan")
        return self.m2 / (self.n - 1)

    def update(self, values: npt.NDArray[np.float64]) -> None:
        batch_n = int(values.shape[0])
        if batch_n == 0:
            return
        batch_mean = float(np.mean(values))
        batch_m2 = float(np.sum((values - batch_mean) ** 2))
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * batch_n / total_n
        self.m2 += batch_m2 + delta * delta * self.n * batch_n / total_n
        self.n = total_n


def _pointwise_kl(
    p: Distribution, q: Distribution, samples: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    return np.asarray(p.logpdf(samples) - q.logpdf(samples), dtype=np.float64)


def _result_from_pointwise(pointwise_kl: npt.NDArray[np.float64]) -> DivergenceResult:
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


def _result_from_stats(stats: _RunningStats) -> DivergenceResult:
    value = float(stats.mean)
    num_samples = stats.n
    sample_variance = stats.sample_variance
    standard_error = (
        float(np.sqrt(sample_variance / num_samples)) if num_samples > 1 else float("nan")
    )

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
