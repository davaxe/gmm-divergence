from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from gmm_divergence import (
    DivergenceResult,
    FitResult,
    Gaussian,
    GaussianMixture,
    fit_mixture_weights,
    kl_divergence,
)
from gmm_divergence.divergence import MonteCarlo
from gmm_divergence.sampling import Samples

if TYPE_CHECKING:
    import numpy.typing as npt
    from pytest_benchmark.fixture import BenchmarkFixture


def _benchmark_mixture() -> GaussianMixture:
    return GaussianMixture.from_arrays(
        weights=[0.2, 0.3, 0.5],
        means=[[-2.0, 0.5], [0.0, -0.5], [2.0, 1.0]],
        covariances=[
            [[0.8, 0.1], [0.1, 0.5]],
            [[1.2, -0.2], [-0.2, 0.9]],
            [[0.6, 0.0], [0.0, 1.1]],
        ],
    )


def test_benchmark_gaussian_mixture_logpdf(benchmark: BenchmarkFixture) -> None:
    mixture = _benchmark_mixture()
    rng = np.random.default_rng(123)
    points = rng.normal(size=(500, 2))
    values = cast("npt.NDArray[np.float64]", benchmark(mixture.logpdf, points))
    assert values.shape == (500,)


def test_benchmark_monte_carlo_kl(benchmark: BenchmarkFixture) -> None:
    p = _benchmark_mixture()
    q = GaussianMixture.from_arrays(
        weights=[0.4, 0.6],
        means=[[-1.5, 0.25], [1.5, 0.75]],
        covariances=[[[1.0, 0.1], [0.1, 0.6]], [[0.9, -0.1], [-0.1, 1.4]]],
    )
    samples = p.sample(750, rng=321)

    result = cast(
        "DivergenceResult",
        benchmark(kl_divergence, p, q, method=MonteCarlo(sampling=Samples(samples))),
    )

    assert result.num_samples == 750


def test_benchmark_moment_matching_fit(benchmark: BenchmarkFixture) -> None:
    p = _benchmark_mixture()
    candidates = [
        Gaussian.from_arrays(mean=[-2.0, 0.5], covariance=[[0.8, 0.1], [0.1, 0.5]]),
        Gaussian.from_arrays(mean=[0.0, -0.5], covariance=[[1.2, -0.2], [-0.2, 0.9]]),
        Gaussian.from_arrays(mean=[2.0, 1.0], covariance=[[0.6, 0.0], [0.0, 1.1]]),
    ]

    result = cast(
        "FitResult", benchmark(fit_mixture_weights, p, candidates, objective="moment_matching")
    )

    assert result.converged is True
