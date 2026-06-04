from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from gmm_divergence import kl_divergence
from gmm_divergence.distribution import GaussianMixture

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def make_spd_covariances(
    rng: np.random.Generator, n_components: int, n_features: int
) -> npt.NDArray[np.float64]:
    covs = np.empty((n_components, n_features, n_features), dtype=np.float64)

    for k in range(n_components):
        matrix = rng.normal(size=(n_features, n_features)).astype(np.float64)
        cov = matrix @ np.transpose(matrix)
        cov += np.eye(n_features, dtype=np.float64) * np.float64(1e-3)
        covs[k] = cov

    return covs


def make_diag_covariances(
    rng: np.random.Generator, n_components: int, n_features: int
) -> npt.NDArray[np.float64]:
    return rng.uniform(low=0.1, high=2.0, size=(n_components, n_features))


def make_gmm(*, seed: int, n_components: int, n_features: int) -> GaussianMixture:
    rng = np.random.default_rng(seed)

    weights = rng.uniform(size=n_components)
    weights /= weights.sum()

    means = rng.normal(size=(n_components, n_features))
    covariances = make_spd_covariances(rng, n_components=n_components, n_features=n_features)
    return GaussianMixture.from_arrays(weights=weights, means=means, covariances=covariances)


@pytest.mark.benchmark(group="kl_monte_carlo")
@pytest.mark.parametrize("n_components", [1, 4, 16])
@pytest.mark.parametrize("n_features", [2, 8, 32])
@pytest.mark.parametrize("num_samples", [100, 1_000, 10_000])
def test_kl_monte_carlo_with_internal_sampling(
    benchmark: BenchmarkFixture, n_components: int, n_features: int, num_samples: int
) -> None:
    p = make_gmm(seed=1, n_components=n_components, n_features=n_features)
    q = make_gmm(seed=2, n_components=n_components, n_features=n_features)
    benchmark(kl_divergence, p, q, sampling=num_samples, method="monte_carlo")
