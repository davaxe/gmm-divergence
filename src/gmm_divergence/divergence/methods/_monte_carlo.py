"""Monte Carlo estimation respecting the batch's integration design."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._sampling import Draw, validate_sample_batch
from gmm_divergence.results import DivergenceResult, MonteCarloStatistics

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._typing import GaussianDistribution
    from gmm_divergence.divergence._options import MonteCarlo


def kl_monte_carlo(
    p: GaussianDistribution, q: GaussianDistribution, /, *, estimator: MonteCarlo
) -> DivergenceResult:
    """Estimate directed KL using source observations and their integration design.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution supplying observations.
    q : Gaussian or GaussianMixture
        Comparison distribution with the same feature dimension.
    estimator : MonteCarlo
        Explicit sample source and optional adaptive stopping configuration.

    Returns
    -------
    DivergenceResult
        Weighted or ordinary mean log-density ratio in nats. Includes evaluation
        counts, sample counts, supported standard error, and adaptive target status.

    Notes
    -----
    Equally weighted observations are assumed iid. Stratified variance requires
    at least two observations in every positive-weight component. Arbitrarily
    weighted batches without component IDs do not support standard errors.
    """
    if estimator.target_standard_error is not None:
        return _adaptive(p, q, estimator)
    batch = validate_sample_batch(p, estimator.sampling.sample(p))
    values = p.logpdf(batch.samples) - q.logpdf(batch.samples)
    weights = batch.integration_weights
    value = float(np.mean(values) if weights is None else np.dot(weights, values))
    standard_error = None
    if weights is None:
        standard_error = _iid_standard_error(values)
    elif batch.component_ids is not None:
        component_weights, _, _ = p.component_arrays()
        variances: list[float] = []
        for index, mass in enumerate(component_weights):
            if mass == 0:
                continue
            stratum = values[batch.component_ids == index]
            if len(stratum) < 2:
                break
            variances.append(float(mass**2 * np.var(stratum, ddof=1) / len(stratum)))
        else:
            standard_error = float(np.sqrt(sum(variances)))
    return _result(value, len(values), standard_error)


def _iid_standard_error(values: FloatArray) -> float | None:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None


def _result(
    value: float,
    n_samples: int,
    standard_error: float | None,
    *,
    target_reached: bool | None = None,
) -> DivergenceResult:
    return DivergenceResult(
        value=value,
        n_evaluations=n_samples,
        monte_carlo=MonteCarloStatistics(
            n_samples=n_samples, standard_error=standard_error, target_reached=target_reached
        ),
    )


def _adaptive(
    p: GaussianDistribution, q: GaussianDistribution, estimator: MonteCarlo
) -> DivergenceResult:
    sampling = estimator.sampling
    if not isinstance(sampling, Draw):
        msg = "adaptive sampling requires Draw."
        raise TypeError(msg)
    if estimator.max_samples is None or estimator.target_standard_error is None:
        msg = "adaptive estimator requires max_samples and target_standard_error."
        raise ValueError(msg)
    rng = np.random.default_rng(sampling.rng)
    values: list[FloatArray] = []
    drawn = 0
    batch_size = estimator.batch_size or sampling.n_samples
    standard_error = None
    pointwise = np.empty(0, dtype=np.float64)
    while drawn < estimator.max_samples:
        count = min(batch_size, estimator.max_samples - drawn)
        samples = p.sample(count, rng=rng)
        values.append(p.logpdf(samples) - q.logpdf(samples))
        drawn += count
        pointwise = np.concatenate(values)
        standard_error = _iid_standard_error(pointwise)
        if (
            drawn >= sampling.n_samples
            and standard_error is not None
            and standard_error <= estimator.target_standard_error
        ):
            return _result(float(np.mean(pointwise)), drawn, standard_error, target_reached=True)
    return _result(float(np.mean(pointwise)), drawn, standard_error, target_reached=False)
