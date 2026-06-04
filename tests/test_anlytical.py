from __future__ import annotations

import numpy as np
import pytest

from gmm_divergence import kl_divergence
from gmm_divergence.distribution import Gaussian, GaussianMixture


def test_mc_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.from_arrays(weights=[1.0], means=[mean_p], covariances=[covariance_p])
    q = GaussianMixture.from_arrays(weights=[1.0], means=[mean_q], covariances=[covariance_q])
    samples = np.random.default_rng(2027).multivariate_normal(
        mean=mean_p, cov=covariance_p, size=200_000
    )
    estimate = kl_divergence(p, q, sampling=samples, method="monte_carlo").value
    estimate_sample_internal = kl_divergence(p, q, sampling=200_000, method="monte_carlo").value
    assert estimate == pytest.approx(
        kl_divergence(p.get_component(0), q.get_component(0), method="closed_form").value, abs=0.01
    )
    assert estimate_sample_internal == pytest.approx(
        kl_divergence(p.get_component(0), q.get_component(0), method="closed_form").value, abs=0.01
    )


def test_uncented_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.from_arrays(weights=[1.0], means=[mean_p], covariances=[covariance_p])
    q = GaussianMixture.from_arrays(weights=[1.0], means=[mean_q], covariances=[covariance_q])
    estimate = kl_divergence(p, q, method="unscented").value
    assert estimate == pytest.approx(
        kl_divergence(p.get_component(0), q.get_component(0), method="closed_form").value, abs=0.01
    )


def test_gaussian_approximation_converges_to_analytical_normal_kl() -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    p = GaussianMixture.from_arrays(weights=[1.0], means=[mean_p], covariances=[covariance_p])
    q = GaussianMixture.from_arrays(weights=[1.0], means=[mean_q], covariances=[covariance_q])
    estimate = kl_divergence(p, q, method="gaussian_approximation").value
    assert estimate == pytest.approx(
        kl_divergence(p.get_component(0), q.get_component(0), method="closed_form").value, abs=0.01
    )


@pytest.mark.parametrize(
    ("p_kind", "q_kind"),
    [("gaussian", "gaussian"), ("gaussian", "mixture"), ("mixture", "gaussian")],
)
@pytest.mark.parametrize("method", ["monte_carlo", "unscented", "gaussian_approximation"])
def test_kl_methods_accept_gaussian_and_mixture_inputs(
    p_kind: str, q_kind: str, method: str
) -> None:
    mean_p = np.array([0.2, -0.4], dtype=np.float64)
    covariance_p = np.array([[1.1, 0.25], [0.25, 0.8]], dtype=np.float64)
    mean_q = np.array([-0.1, 0.3], dtype=np.float64)
    covariance_q = np.array([[0.9, -0.1], [-0.1, 1.4]], dtype=np.float64)

    gaussian_p = Gaussian.from_arrays(mean=mean_p, covariance=covariance_p)
    gaussian_q = Gaussian.from_arrays(mean=mean_q, covariance=covariance_q)
    mixture_p = GaussianMixture.from_arrays(
        weights=[1.0], means=[mean_p], covariances=[covariance_p]
    )
    mixture_q = GaussianMixture.from_arrays(
        weights=[1.0], means=[mean_q], covariances=[covariance_q]
    )
    p = gaussian_p if p_kind == "gaussian" else mixture_p
    q = gaussian_q if q_kind == "gaussian" else mixture_q

    expected = kl_divergence(gaussian_p, gaussian_q, method="closed_form").value
    if method == "monte_carlo":
        samples = np.random.default_rng(2027).multivariate_normal(
            mean=mean_p, cov=covariance_p, size=200_000
        )
        estimate = kl_divergence(p, q, sampling=samples, method="monte_carlo").value
        assert estimate == pytest.approx(expected, abs=0.01)
    elif method == "unscented":
        estimate = kl_divergence(p, q, method="unscented").value
        assert estimate == pytest.approx(expected, abs=1e-12)
    else:
        estimate = kl_divergence(p, q, method="gaussian_approximation").value
        assert estimate == pytest.approx(expected, abs=1e-12)
