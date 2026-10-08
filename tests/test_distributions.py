from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

import gmm_divergence as gd
from gmm_divergence import Gaussian, GaussianMixture, combine_gaussians

if TYPE_CHECKING:
    import numpy.typing as npt


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


@pytest.mark.parametrize("n_components", [1, 2])
def test_sklearn_conversion_preserves_distribution(n_components: int) -> None:
    mixture = GaussianMixture.from_arrays(
        weights=np.ones(n_components),
        means=[[0, 1], [3, -2]][:n_components],
        covariances=[[[2, 0.6], [0.6, 1]], [[1, -0.3], [-0.3, 2]]][:n_components],
    )
    converted = mixture.to_sklearn_gmm()
    points = np.array([[0, 1], [2, -1], [-3, 4]], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(converted.score_samples(points), dtype=np.float64), mixture.logpdf(points)
    )
    np.testing.assert_allclose(
        np.asarray(converted.precisions_, dtype=np.float64), np.linalg.inv(mixture.covariances)
    )
    probabilities = np.asarray(converted.predict_proba(points), dtype=np.float64)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1)
    sample_result = cast("tuple[npt.ArrayLike, npt.ArrayLike]", converted.sample(5))
    samples = np.asarray(sample_result[0], dtype=np.float64)
    labels = np.asarray(sample_result[1], dtype=np.intp)
    assert samples.shape == (5, 2)
    assert labels.shape == (5,)
    roundtrip = GaussianMixture.from_sklearn_gmm(converted)
    np.testing.assert_array_equal(roundtrip.weights, mixture.weights)
    np.testing.assert_array_equal(roundtrip.means, mixture.means)
    np.testing.assert_array_equal(roundtrip.covariances, mixture.covariances)
    assert not np.shares_memory(converted.means_, mixture.means)
    assert not np.shares_memory(converted.weights_, mixture.weights)
    assert not np.shares_memory(converted.covariances_, mixture.covariances)


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_from_sklearn_supports_all_covariance_types(
    covariance_type: Literal["full", "tied", "diag", "spherical"],
) -> None:
    rng = np.random.default_rng(0)
    points = np.concatenate([
        rng.multivariate_normal([-3, 1], [[1, 0.3], [0.3, 0.5]], size=100),
        rng.multivariate_normal([3, -1], [[0.5, -0.1], [-0.1, 1]], size=100),
    ])
    estimator = SklearnGaussianMixture(
        n_components=2, covariance_type=covariance_type, random_state=0
    ).fit(points)
    mixture = GaussianMixture.from_sklearn_gmm(estimator)
    assert mixture.covariances.shape == (2, 2, 2)
    np.testing.assert_allclose(
        mixture.logpdf(points), np.asarray(estimator.score_samples(points), dtype=np.float64)
    )
    assert not np.shares_memory(mixture.weights, estimator.weights_)
    assert not np.shares_memory(mixture.means, estimator.means_)
    assert not np.shares_memory(mixture.covariances, estimator.covariances_)


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

    with pytest.raises(ValueError, match="Covariance must be symmetric"):
        _ = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=[[1.0, 0.2], [0.0, 1.0]])


def test_distribution_construction_does_not_alias_input_or_expose_mutable_caches() -> None:
    means = np.array([[0.0], [1.0]], dtype=np.float64)
    mixture = GaussianMixture.from_arrays(
        weights=[0.5, 0.5], means=means, covariances=[[[1.0]], [[1.0]]]
    )
    means[0, 0] = 42.0

    assert means.flags.writeable
    assert mixture.means[0, 0] == pytest.approx(0.0)
    assert not mixture.chol().flags.writeable
    assert not mixture.log_dets().flags.writeable

    gaussian = Gaussian.univariate()
    assert not gaussian.chol().flags.writeable


def test_gaussian_construction_keeps_covariance_regularization_explicit() -> None:
    covariance = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Covariance must be positive definite"):
        _ = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=covariance)

    gaussian = Gaussian.from_arrays(
        mean=[0.0, 1.0],
        covariance=gd.covariance.regularize_covariance(
            covariance, regularizer=gd.covariance.DiagonalLoading(eps=1e-3)
        ),
    )

    assert gaussian.covariance == pytest.approx(np.array([[1.001, 0.0], [0.0, 0.001]]))
    assert not gaussian.covariance.flags.writeable
    assert np.all(np.linalg.eigvalsh(gaussian.covariance) > 0.0)


def test_gaussian_mixture_construction_with_regularized_covariance_batch() -> None:
    covariances = np.array([[[1.0, 0.0], [0.0, 0.0]], [[2.0, 0.25], [0.25, 0.5]]], dtype=np.float64)

    mixture = GaussianMixture.from_arrays(
        weights=[2.0, 1.0],
        means=[[0.0, 0.0], [2.0, -1.0]],
        covariances=gd.covariance.regularize_covariance(
            covariances,
            regularizer=gd.covariance.EigenvalueClipping(min_eigenvalue=0.1),
            batched=True,
        ),
    )

    assert mixture.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert mixture.covariances.shape == (2, 2, 2)
    assert not mixture.covariances.flags.writeable
    assert float(np.min(np.linalg.eigvalsh(mixture.covariances))) >= 0.1 - 1e-12
