from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
import pytest
from scipy.integrate import quad

import gmm_divergence as gd
from gmm_divergence import Gaussian, GaussianMixture, kl_divergence
from gmm_divergence.divergence import MonteCarlo

if TYPE_CHECKING:
    from gmm_divergence.divergence import DivergenceSpec, KLMethod

EstimatorString: TypeAlias = Literal[
    "closed_form", "unscented", "variational", "gaussian_approximation"
]
EstimatorUnderTest: TypeAlias = EstimatorString | MonteCarlo


def _manual_gaussian_kl(p: Gaussian, q: Gaussian) -> float:
    inv_cov_q = np.linalg.inv(q.covariance)
    mean_delta = q.mean - p.mean
    trace_term = np.trace(inv_cov_q @ p.covariance)
    quadratic_term = mean_delta @ inv_cov_q @ mean_delta
    _, logdet_p = np.linalg.slogdet(p.covariance)
    _, logdet_q = np.linalg.slogdet(q.covariance)
    return float(0.5 * (trace_term + quadratic_term - p.dim + logdet_q - logdet_p))


def _quadrature_kl_1d(p: GaussianMixture, q: GaussianMixture) -> float:
    def integrand(x: float) -> float:
        point = np.array([[x]], dtype=np.float64)
        log_p = p.logpdf(point)[0]
        return float(np.exp(log_p) * (log_p - q.logpdf(point)[0]))

    value, error = quad(integrand, -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11, limit=300)
    assert error < 1e-8
    return float(value)


def test_closed_form_kl_matches_independent_gaussian_formula() -> None:
    p = Gaussian.from_arrays(mean=[0.5, -1.0], covariance=[[1.2, 0.2], [0.2, 0.7]])
    q = Gaussian.from_arrays(mean=[-0.25, 0.75], covariance=[[0.9, -0.1], [-0.1, 1.8]])
    result = kl_divergence(p, q, method="closed_form")
    assert result.method == "closed_form"
    assert result.num_samples is None
    assert result.value == pytest.approx(_manual_gaussian_kl(p, q), rel=1e-14, abs=1e-14)
    assert result.value > 0.0
    assert kl_divergence(q, p, method="closed_form").value != pytest.approx(result.value)


@pytest.mark.parametrize(
    ("method", "method_name"),
    [
        ("closed_form", "closed_form"),
        ("unscented", "unscented"),
        ("variational", "variational"),
        ("gaussian_approximation", "moment_matching"),
    ],
)
def test_gaussian_family_estimators_are_exact_for_single_gaussian_pairs(
    method: EstimatorString, method_name: str
) -> None:
    p = Gaussian.from_arrays(mean=[0.5, -1.0], covariance=[[1.2, 0.2], [0.2, 0.7]])
    q = Gaussian.from_arrays(mean=[-0.25, 0.75], covariance=[[0.9, -0.1], [-0.1, 1.8]])
    expected = kl_divergence(p, q, method="closed_form").value

    result = kl_divergence(p, q, method=cast("KLMethod", method), prefer_closed_form=False)

    assert result.method == method_name
    assert result.value == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_monte_carlo_kl_for_mixtures_matches_numerical_quadrature_reference() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.35, 0.65], means=[[-1.0], [1.25]], covariances=[[[0.35]], [[0.8]]]
    )
    q = GaussianMixture.from_arrays(
        weights=[0.55, 0.45], means=[[-0.4], [2.0]], covariances=[[[0.6]], [[1.1]]]
    )
    expected = _quadrature_kl_1d(p, q)

    result = kl_divergence(p, q, method=MonteCarlo(sampling=gd.sampling.Draw(100_000, rng=1234)))

    assert expected == pytest.approx(0.1129823940300846, rel=1e-12)
    assert result.method == "monte_carlo"
    assert result.num_samples == 100_000
    assert result.value == pytest.approx(expected, abs=0.015)
    assert result.value > 0.0


@pytest.mark.parametrize(
    "method",
    [
        MonteCarlo(sampling=gd.sampling.Draw(1_000, rng=7)),
        "unscented",
        "variational",
        "gaussian_approximation",
    ],
)
def test_kl_estimators_return_zero_for_identical_mixtures(method: EstimatorUnderTest) -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[0.35, 0.65], means=[[-1.0], [1.25]], covariances=[[[0.35]], [[0.8]]]
    )

    result = kl_divergence(mixture, mixture, method=cast("KLMethod", method))

    assert result.value == pytest.approx(0.0, abs=1e-14)


def test_monte_carlo_stratified_sampling_for_mixture_reference() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.8, 0.2], means=[[-1.0], [1.0]], covariances=[[[0.3]], [[0.7]]]
    )

    result = kl_divergence(p, p, method=MonteCarlo(sampling=gd.sampling.Stratified(10, rng=123)))

    assert result.method == "monte_carlo"
    assert result.num_samples == 10
    assert result.value == pytest.approx(0.0, abs=1e-14)
    assert result.monte_carlo_stats is not None
    assert result.monte_carlo_stats.effective_sample_size == 10


def test_monte_carlo_stratified_sampling_requires_samples_for_positive_components() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.8, 0.2], means=[[-1.0], [1.0]], covariances=[[[0.3]], [[0.7]]]
    )

    with pytest.raises(ValueError, match="at least one sample per positive-weight component"):
        _ = kl_divergence(p, p, method=MonteCarlo(sampling=gd.sampling.Stratified(1, rng=123)))


def test_gaussian_approximation_matches_closed_form_of_moment_matched_mixtures() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[-2.0], [1.5]], covariances=[[[0.5]], [[1.2]]]
    )
    q = GaussianMixture.from_arrays(
        weights=[0.6, 0.4], means=[[-1.0], [2.5]], covariances=[[[0.9]], [[0.7]]]
    )
    expected = kl_divergence(p.as_gaussian(), q.as_gaussian(), method="closed_form").value

    result = kl_divergence(p, q, method="gaussian_approximation")

    assert result.method == "moment_matching"
    assert result.value == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_monte_carlo_uses_precomputed_samples_without_resampling() -> None:
    p = Gaussian.from_arrays(mean=[0.0], covariance=[[1.0]])
    q = Gaussian.from_arrays(mean=[1.0], covariance=[[2.0]])
    samples = np.array([[-2.0], [-0.5], [0.0], [1.5], [3.0]], dtype=np.float64)
    expected = float(np.mean(p.logpdf(samples) - q.logpdf(samples)))

    result = kl_divergence(
        p, q, method=MonteCarlo(sampling=gd.sampling.Samples(samples)), prefer_closed_form=False
    )

    assert result.method == "monte_carlo"
    assert result.num_samples == samples.shape[0]
    assert result.value == pytest.approx(expected, rel=1e-15, abs=1e-15)


def test_kl_divergence_rejects_invalid_inputs_and_methods() -> None:
    p = Gaussian.from_arrays(mean=[0.0], covariance=[[1.0]])
    q_wrong_dim = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=np.eye(2))
    mixture = GaussianMixture.from_arrays(
        weights=[0.5, 0.5], means=[[-1.0], [1.0]], covariances=[[[1.0]], [[1.0]]]
    )

    with pytest.raises(ValueError, match="Gaussian-family distribution dimensions must match"):
        _ = kl_divergence(p, q_wrong_dim, method="closed_form")

    with pytest.raises(TypeError, match="requires p and q to be Gaussian"):
        _ = kl_divergence(mixture, p, method="closed_form")

    with pytest.raises(ValueError, match="Unknown KL method"):
        _ = kl_divergence(p, p, method=cast("KLMethod", cast("object", "not-a-method")))

    with pytest.raises(ValueError, match="must have shape"):
        _ = kl_divergence(
            p,
            p,
            method=MonteCarlo(sampling=gd.sampling.Samples(np.zeros((3, 2)))),
            prefer_closed_form=False,
        )

    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        _ = MonteCarlo(sampling=gd.sampling.Draw(0))


def test_monte_carlo_reports_standard_error() -> None:
    p = Gaussian.from_arrays(mean=[0.0], covariance=[[1.0]])
    q = Gaussian.from_arrays(mean=[1.0], covariance=[[2.0]])
    samples = np.array([[-2.0], [-0.5], [0.0], [1.5], [3.0]], dtype=np.float64)

    pointwise = p.logpdf(samples) - q.logpdf(samples)
    expected_value = float(np.mean(pointwise))
    expected_variance = float(np.var(pointwise, ddof=1))
    expected_se = float(np.sqrt(expected_variance / samples.shape[0]))

    result = kl_divergence(
        p, q, method=MonteCarlo(sampling=gd.sampling.Samples(samples)), prefer_closed_form=False
    )

    assert result.value == pytest.approx(expected_value)
    assert result.monte_carlo_stats is not None
    assert result.monte_carlo_stats.sample_variance == pytest.approx(expected_variance)
    assert result.monte_carlo_stats.standard_error == pytest.approx(expected_se)


def test_component_kl_matrix_matches_closed_form_component_pairs() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[-2.0], [1.5]], covariances=[[[0.5]], [[1.2]]]
    )
    q = GaussianMixture.from_arrays(
        weights=[0.6, 0.4], means=[[-1.0], [2.5]], covariances=[[[0.9]], [[0.7]]]
    )

    matrix = gd.component_kl_matrix(p, q)
    expected = np.array(
        [
            [
                kl_divergence(p.get_component(0), q.get_component(0), method="closed_form").value,
                kl_divergence(p.get_component(0), q.get_component(1), method="closed_form").value,
            ],
            [
                kl_divergence(p.get_component(1), q.get_component(0), method="closed_form").value,
                kl_divergence(p.get_component(1), q.get_component(1), method="closed_form").value,
            ],
        ],
        dtype=np.float64,
    )

    assert matrix.shape == (2, 2)
    assert matrix == pytest.approx(expected)


def test_component_kl_matrix_treats_gaussian_as_single_component() -> None:
    p = Gaussian.univariate(mean=0.0, variance=1.0)
    q = GaussianMixture.from_arrays(
        weights=[0.4, 0.6], means=[[-1.0], [2.0]], covariances=[[[0.5]], [[1.5]]]
    )

    matrix = gd.component_kl_matrix(p, q)

    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == pytest.approx(
        kl_divergence(p, q.get_component(0), method="closed_form").value
    )


def test_symmetric_kl_divergence_averages_both_directions() -> None:
    p = Gaussian.from_arrays(mean=[0.5, -1.0], covariance=[[1.2, 0.2], [0.2, 0.7]])
    q = Gaussian.from_arrays(mean=[-0.25, 0.75], covariance=[[0.9, -0.1], [-0.1, 1.8]])

    result = gd.symmetric_kl_divergence(p, q, method="closed_form")
    swapped = gd.symmetric_kl_divergence(q, p, method="closed_form")
    expected = 0.5 * (
        kl_divergence(p, q, method="closed_form").value
        + kl_divergence(q, p, method="closed_form").value
    )

    assert result.method == "symmetric_kl"
    assert result.num_samples is None
    assert result.value == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert swapped.value == pytest.approx(result.value, rel=1e-14, abs=1e-14)


def test_symmetric_kl_divergence_reports_total_monte_carlo_samples() -> None:
    p = Gaussian.univariate(mean=0.0, variance=1.0)
    q = Gaussian.univariate(mean=1.0, variance=2.0)
    samples = np.array([[-2.0], [-0.5], [0.0], [1.5], [3.0]], dtype=np.float64)

    result = gd.symmetric_kl_divergence(
        p, q, method=MonteCarlo(sampling=gd.sampling.Samples(samples)), prefer_closed_form=False
    )

    assert result.num_samples == 2 * samples.shape[0]


def test_jensen_shannon_divergence_is_zero_for_identical_mixtures() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[0.35, 0.65], means=[[-1.0], [1.25]], covariances=[[[0.35]], [[0.8]]]
    )
    samples = np.array([[-2.0], [-0.5], [0.0], [1.5], [3.0]], dtype=np.float64)

    result = gd.jensen_shannon_divergence(
        mixture, mixture, method=MonteCarlo(sampling=gd.sampling.Samples(samples))
    )

    assert result.method == "jensen_shannon"
    assert result.num_samples == 2 * samples.shape[0]
    assert result.value == pytest.approx(0.0, abs=1e-14)


def test_estimate_divergence_dispatches_existing_divergence_helpers() -> None:
    p = Gaussian.from_arrays(mean=[0.5], covariance=[[1.2]])
    q = Gaussian.from_arrays(mean=[-0.25], covariance=[[0.9]])

    kl = gd.estimate_divergence(p, q, divergence="kl")
    configured_kl = gd.estimate_divergence(
        p, q, divergence=gd.divergence.KLDivergence(method="closed_form")
    )
    symmetric = gd.estimate_divergence(
        p, q, divergence=gd.divergence.SymmetricKLDivergence(method="closed_form")
    )

    assert kl.value == pytest.approx(gd.kl_divergence(p, q, method="closed_form").value)
    assert configured_kl.value == pytest.approx(kl.value)
    assert symmetric.value == pytest.approx(
        gd.symmetric_kl_divergence(p, q, method="closed_form").value
    )


def test_estimate_divergence_dispatches_existing_jensen_shannon_helper() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[0.35, 0.65], means=[[-1.0], [1.25]], covariances=[[[0.35]], [[0.8]]]
    )
    samples = np.array([[-2.0], [-0.5], [0.0], [1.5], [3.0]], dtype=np.float64)
    method = MonteCarlo(sampling=gd.sampling.Samples(samples))

    result = gd.estimate_divergence(
        mixture, mixture, divergence=gd.divergence.JensenShannonDivergence(method=method)
    )

    assert result.method == "jensen_shannon"
    assert result.value == pytest.approx(
        gd.jensen_shannon_divergence(mixture, mixture, method=method).value
    )


def test_estimate_divergence_rejects_unknown_divergence() -> None:
    p = Gaussian.univariate(mean=0.0, variance=1.0)

    with pytest.raises(ValueError, match="Unknown divergence method"):
        _ = gd.estimate_divergence(
            p, p, divergence=cast("DivergenceSpec", cast("object", "not-a-divergence"))
        )
