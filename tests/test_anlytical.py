from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from gmm_divergence import kl_divergence
from gmm_divergence.distribution import GaussianMixture


def kl_normal(
    mean_p: npt.NDArray[np.float64],
    covariance_p: npt.NDArray[np.float64],
    mean_q: npt.NDArray[np.float64],
    covariance_q: npt.NDArray[np.float64],
) -> float:
    n_features = mean_p.shape[0]
    covariance_q_inv = np.linalg.inv(covariance_q)
    mean_delta = mean_q - mean_p

    return float(
        0.5
        * (
            np.trace(covariance_q_inv @ covariance_p)
            + mean_delta @ covariance_q_inv @ mean_delta
            - n_features
            + np.log(np.linalg.det(covariance_q) / np.linalg.det(covariance_p))
        )
    )


def test_mc_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.create(
        weights=[1.0],
        means=[mean_p],
        covariances=[covariance_p],
    )
    q = GaussianMixture.create(
        weights=[1.0],
        means=[mean_q],
        covariances=[covariance_q],
    )
    samples = np.random.default_rng(2027).multivariate_normal(
        mean=mean_p,
        cov=covariance_p,
        size=200_000,
    )
    estimate = kl_divergence(p, q, samples=samples, method="monte_carlo").value
    estimate_sample_internal = kl_divergence(p, q, num_samples=200_000, method="monte_carlo").value
    assert estimate == pytest.approx(
        kl_normal(mean_p, covariance_p, mean_q, covariance_q),
        abs=0.01,
    )
    assert estimate_sample_internal == pytest.approx(
        kl_normal(mean_p, covariance_p, mean_q, covariance_q),
        abs=0.01,
    )


def test_uncented_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.create(
        weights=[1.0],
        means=[mean_p],
        covariances=[covariance_p],
    )
    q = GaussianMixture.create(
        weights=[1.0],
        means=[mean_q],
        covariances=[covariance_q],
    )
    estimate = kl_divergence(p, q, method="unscented").value
    assert estimate == pytest.approx(
        kl_normal(mean_p, covariance_p, mean_q, covariance_q),
        abs=0.01,
    )


def test_gaussian_approximation_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.create(
        weights=[1.0],
        means=[mean_p],
        covariances=[covariance_p],
    )
    q = GaussianMixture.create(
        weights=[1.0],
        means=[mean_q],
        covariances=[covariance_q],
    )
    estimate = kl_divergence(p, q, method="gaussian_approximation").value
    assert estimate == pytest.approx(
        kl_normal(mean_p, covariance_p, mean_q, covariance_q),
        abs=0.01,
    )
