from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._sampling import Draw, Stratified, stratified_mixture_samples
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture
from gmm_divergence.results import DivergenceResult, MonteCarloStatistics

if TYPE_CHECKING:
    from gmm_divergence.distributions._typing import GaussianLike
    from gmm_divergence.divergence._options import MonteCarlo


def kl_monte_carlo(
    p: GaussianLike, q: GaussianLike, /, *, estimator: MonteCarlo
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
    p : Gaussian or GaussianMixture
        Reference distribution to sample from.
    q : Gaussian or GaussianMixture
        Approximating distribution evaluated at the sampled points.
    estimator : MonteCarlo
        Explicit sampling and optional adaptive-sampling configuration.

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
    sampling = estimator.sampling

    if isinstance(sampling, Stratified):
        return _kl_monte_carlo_stratified(p, q, sampling=sampling)

    if estimator.target_standard_error is not None:
        return _adaptive_kl_monte_carlo(p, q, estimator)

    samples = sampling.sample(p)
    pointwise_kl = _pointwise_kl(p, q, samples)
    return _result_from_pointwise(pointwise_kl)


def _adaptive_kl_monte_carlo(
    p: GaussianLike, q: GaussianLike, estimator: MonteCarlo, /
) -> DivergenceResult:
    sampling = estimator.sampling
    if not isinstance(sampling, Draw):
        msg = "Adaptive MonteCarlo requires sampling.Draw."
        raise TypeError(msg)
    if estimator.max_samples is None or estimator.target_standard_error is None:
        msg = "Adaptive MonteCarlo configuration is incomplete."
        raise AssertionError(msg)
    batch_size = estimator.batch_size or sampling.n_samples
    rng = np.random.default_rng(sampling.rng)
    batches: list[npt.NDArray[np.float64]] = []
    drawn = 0
    while drawn < estimator.max_samples:
        count = min(batch_size, estimator.max_samples - drawn)
        batches.append(_pointwise_kl(p, q, p.sample(count, rng=rng)))
        drawn += count
        result = _result_from_pointwise(np.concatenate(batches))
        stats = result.monte_carlo_stats
        if (
            stats is not None
            and drawn >= sampling.n_samples
            and stats.standard_error <= estimator.target_standard_error
        ):
            return result
    return _result_from_pointwise(np.concatenate(batches))


def _kl_monte_carlo_stratified(
    p: GaussianLike, q: GaussianLike, /, *, sampling: Stratified
) -> DivergenceResult:
    p = GaussianMixture.from_components([p]) if isinstance(p, Gaussian) else p
    result = stratified_mixture_samples(p, sampling)
    pointwise_kl = _pointwise_kl(p, q, result.samples)
    weights = np.asarray(p.weights, dtype=np.float64)
    component_means = np.zeros_like(weights, dtype=np.float64)
    component_variances = np.zeros_like(weights, dtype=np.float64)

    for component_index, count in enumerate(result.counts):
        if count == 0:
            continue
        values = pointwise_kl[result.component_ids == component_index]
        component_means[component_index] = float(np.mean(values))
        if count > 1:
            component_variances[component_index] = float(np.var(values, ddof=1))

    value = float(np.dot(weights, component_means))
    variance_of_estimator = float(
        np.sum([
            weights[index] ** 2 * component_variances[index] / count
            for index, count in enumerate(result.counts)
            if count > 0
        ])
    )
    standard_error = float(np.sqrt(variance_of_estimator))
    sample_variance = float(variance_of_estimator * sampling.n_samples)
    return _monte_carlo_result(
        value=value,
        num_samples=sampling.n_samples,
        sample_variance=sample_variance,
        standard_error=standard_error,
        effective_sample_size=sampling.n_samples,
    )


def _pointwise_kl(
    p: GaussianLike, q: GaussianLike, samples: npt.ArrayLike
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

    return _monte_carlo_result(
        value=value,
        num_samples=num_samples,
        sample_variance=sample_variance,
        standard_error=standard_error,
        effective_sample_size=num_samples,
    )


def _monte_carlo_result(
    *,
    value: float,
    num_samples: int,
    sample_variance: float,
    standard_error: float,
    effective_sample_size: int,
) -> DivergenceResult:
    return DivergenceResult(
        value=value,
        method="monte_carlo",
        num_samples=num_samples,
        monte_carlo_stats=MonteCarloStatistics(
            sample_mean=value,
            sample_variance=sample_variance,
            standard_error=standard_error,
            effective_sample_size=effective_sample_size,
        ),
    )
