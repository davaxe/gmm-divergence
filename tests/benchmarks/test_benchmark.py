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
    rng: np.random.Generator,
    n_components: int,
    n_features: int,
    dtype: type[np.floating] = np.float64,
) -> npt.NDArray[np.floating]:
    covs = np.empty((n_components, n_features, n_features), dtype=dtype)

    for k in range(n_components):
        matrix = rng.normal(size=(n_features, n_features)).astype(dtype)
        cov = matrix @ matrix.T
        cov += np.eye(n_features, dtype=dtype) * dtype(1e-3)
        covs[k] = cov

    return covs


def make_diag_covariances(
    rng: np.random.Generator,
    n_components: int,
    n_features: int,
) -> npt.NDArray[np.float64]:
    return rng.uniform(
        low=0.1,
        high=2.0,
        size=(n_components, n_features),
    )


def make_gmm(
    *,
    seed: int,
    n_components: int,
    n_features: int,
    covariance_type: str,
) -> GaussianMixture[np.float64]:
    rng = np.random.default_rng(seed)

    weights = rng.uniform(size=n_components)
    weights /= weights.sum()

    means = rng.normal(size=(n_components, n_features))

    if covariance_type == "full":
        covariances = make_spd_covariances(
            rng,
            n_components=n_components,
            n_features=n_features,
        )
    elif covariance_type == "diag":
        covariances = make_diag_covariances(
            rng,
            n_components=n_components,
            n_features=n_features,
        )
    else:
        msg = f"Unsupported covariance_type: {covariance_type}"
        raise ValueError(msg)

    return GaussianMixture.create(
        weights=weights,
        means=means,
        covariances=covariances,
        covariance_type=covariance_type,
    )


@pytest.mark.benchmark(group="kl_monte_carlo")
@pytest.mark.parametrize("n_components", [1, 4, 16])
@pytest.mark.parametrize("n_features", [2, 8, 32])
@pytest.mark.parametrize("num_samples", [100, 1_000, 10_000])
@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_kl_monte_carlo_with_internal_sampling(
    benchmark: BenchmarkFixture,
    n_components: int,
    n_features: int,
    num_samples: int,
    covariance_type: str,
) -> None:
    p = make_gmm(
        seed=1,
        n_components=n_components,
        n_features=n_features,
        covariance_type=covariance_type,
    )
    q = make_gmm(
        seed=2,
        n_components=n_components,
        n_features=n_features,
        covariance_type=covariance_type,
    )

    benchmark(
        kl_divergence,
        p,
        q,
        num_samples=num_samples,
        method="monte_carlo",
    )
