from __future__ import annotations

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence import Gaussian, GaussianMixture, combine_gaussians


def test_gaussian_matches_manual_logpdf() -> None:
    gaussian = Gaussian.from_arrays(mean=[1.0, -2.0], covariance=np.diag([4.0, 9.0]))
    x = np.array([[1.0, -2.0], [3.0, 1.0], [-1.0, -5.0]], dtype=np.float64)

    diff = x - np.array([1.0, -2.0], dtype=np.float64)
    mahalanobis = diff[:, 0] ** 2 / 4.0 + diff[:, 1] ** 2 / 9.0
    expected = -0.5 * (2.0 * np.log(2.0 * np.pi) + np.log(4.0 * 9.0) + mahalanobis)

    assert gaussian.dim == 2
    assert np.array_equal(gaussian.covariance, np.diag([4.0, 9.0]))
    assert gaussian.logpdf(x) == pytest.approx(expected)
    assert not gaussian.mean.flags.writeable
    assert not gaussian.covariance.flags.writeable


def test_gaussian_mixture_normalizes_weights_and_matches_manual_logpdf() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[2.0, 1.0], means=[[-1.0], [2.0]], covariances=[[[0.5]], [[1.5]]]
    )
    x = np.array([[-2.0], [0.0], [3.0]], dtype=np.float64)

    components = [
        Gaussian.univariate(mean=-1.0, variance=0.5),
        Gaussian.univariate(mean=2.0, variance=1.5),
    ]
    component_logpdf = np.column_stack([component.logpdf(x) for component in components])
    expected = np.logaddexp.reduce(np.log(mixture.weights)[None, :] + component_logpdf, axis=1)

    assert mixture.dim == 1
    assert mixture.n_components == 2
    assert mixture.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert mixture.logpdf(x) == pytest.approx(expected)


def test_gaussian_mixture_sampling_is_seeded_and_has_expected_shape() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[0.2, 0.8],
        means=[[0.0, 0.0], [5.0, -1.0]],
        covariances=[np.eye(2), 0.5 * np.eye(2)],
    )

    first = mixture.sample(12, rng=123)
    second = mixture.sample(12, rng=123)
    assert first.shape == (12, 2)
    assert first.dtype == np.float64
    assert np.array_equal(first, second)

    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        _ = mixture.sample(0, rng=123)

    with pytest.raises(ValueError, match="x must have shape"):
        _ = mixture.logpdf(np.zeros((3, 3)))


def test_combine_gaussians_flattens_sources_and_records_component_mapping() -> None:
    gaussian = Gaussian.from_arrays(mean=[-2.0], covariance=[[0.5]])
    mixture = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[1.0], [3.0]], covariances=[[[1.0]], [[2.0]]]
    )

    combined = combine_gaussians([gaussian, mixture], weights=[0.4, 0.6], include_mapping=True)

    assert combined.mixture.weights == pytest.approx([0.4, 0.15, 0.45])
    assert combined.mixture.means[:, 0] == pytest.approx([-2.0, 1.0, 3.0])
    assert combined.mapping.source_of(0) == (0, 0)
    assert combined.mapping.source_of(2) == (1, 1)
    assert combined.mapping.component_of(1, 0) == 1
    with pytest.raises(ValueError, match="No component found"):
        _ = combined.mapping.component_of(2, 0)


def test_distribution_constructors_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="Mean must contain at least one feature"):
        _ = Gaussian.from_arrays(mean=[], covariance=[])

    with pytest.raises(ValueError, match="Covariance must be positive definite"):
        _ = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=[[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="Weights must be nonnegative"):
        _ = GaussianMixture.from_arrays(
            weights=[0.5, -0.5], means=[[0.0], [1.0]], covariances=[[[1.0]], [[1.0]]]
        )

    with pytest.raises(ValueError, match="Means must be a 2D array"):
        _ = GaussianMixture.from_arrays(weights=[1.0], means=[0.0], covariances=[[[1.0]]])


def test_gaussian_from_regularized_arrays_keeps_strict_constructor_explicit() -> None:
    covariance = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Covariance must be positive definite"):
        _ = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=covariance)

    gaussian = Gaussian.from_regularized_arrays(
        mean=[0.0, 1.0], covariance=covariance, regularization=gd.DiagonalLoading(eps=1e-3)
    )

    assert gaussian.covariance == pytest.approx(np.array([[1.001, 0.0], [0.0, 0.001]]))
    assert not gaussian.covariance.flags.writeable
    assert np.all(np.linalg.eigvalsh(gaussian.covariance) > 0.0)


def test_gaussian_mixture_from_regularized_arrays_regularizes_covariance_batch() -> None:
    covariances = np.array([[[1.0, 0.0], [0.0, 0.0]], [[2.0, 0.25], [0.25, 0.5]]], dtype=np.float64)

    mixture = GaussianMixture.from_regularized_arrays(
        weights=[2.0, 1.0],
        means=[[0.0, 0.0], [2.0, -1.0]],
        covariances=covariances,
        regularization=gd.EigenvalueClipping(min_eigenvalue=0.1),
    )

    assert mixture.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert mixture.covariances.shape == (2, 2, 2)
    assert not mixture.covariances.flags.writeable
    assert float(np.min(np.linalg.eigvalsh(mixture.covariances))) >= 0.1 - 1e-12
