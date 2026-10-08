from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence.fitting import ComponentStatistics, component_statistics

if TYPE_CHECKING:
    import numpy.typing as npt


def test_statistics_match_weighted_moments() -> None:
    stats = component_statistics([[0], [2], [4]], responsibilities=[[1, 0], [0.5, 0.5], [0, 1]])
    np.testing.assert_allclose(stats.weights, [0.5, 0.5])
    np.testing.assert_allclose(stats.means, [[2 / 3], [10 / 3]])
    np.testing.assert_allclose(stats.covariances, [[[8 / 9]], [[8 / 9]]])
    np.testing.assert_allclose(stats.soft_counts, [1.5, 1.5])
    np.testing.assert_allclose(stats.effective_sample_sizes, [1.8, 1.8])
    mixture = stats.to_gaussian_mixture()
    np.testing.assert_allclose(mixture.means, stats.means)
    assert gd.fitting.component_statistics is component_statistics


def test_reference_mixture_matches_explicit_responsibilities() -> None:
    mixture = gd.GaussianMixture.from_components([
        gd.Gaussian.univariate(-1),
        gd.Gaussian.univariate(1),
    ])
    points = np.array([[-2], [-1], [0], [1], [2]], dtype=np.float64)
    weights, means, covariances = (
        mixture.weights.copy(),
        mixture.means.copy(),
        mixture.covariances.copy(),
    )
    reference_stats = component_statistics(points, mixture=mixture)
    explicit_stats = component_statistics(points, responsibilities=mixture.responsibilities(points))
    for field in fields(ComponentStatistics):
        np.testing.assert_array_equal(
            getattr(reference_stats, field.name), getattr(explicit_stats, field.name)
        )
    np.testing.assert_array_equal(mixture.weights, weights)
    np.testing.assert_array_equal(mixture.means, means)
    np.testing.assert_array_equal(mixture.covariances, covariances)


def test_statistics_arrays_are_independent_and_read_only() -> None:
    points = np.array([[0], [2], [4]], dtype=np.float64)
    responsibilities = np.array([[1, 0], [0.5, 0.5], [0, 1]], dtype=np.float64)
    points_before, assignments_before = points.copy(), responsibilities.copy()
    stats = component_statistics(points, responsibilities=responsibilities)
    np.testing.assert_array_equal(points, points_before)
    np.testing.assert_array_equal(responsibilities, assignments_before)
    assert points.flags.writeable
    assert responsibilities.flags.writeable
    arrays = {field.name: getattr(stats, field.name).copy() for field in fields(stats)}
    copied = ComponentStatistics(**arrays)
    for name, original in arrays.items():
        value = getattr(copied, name)
        assert value.dtype == np.float64
        assert not value.flags.writeable
        assert not np.shares_memory(value, original)
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = 42
        original.flat[0] = 42
        assert value.flat[0] != 42


def test_regularization_is_only_applied_at_conversion() -> None:
    stats = component_statistics([[2, 3]], responsibilities=[[1]])
    np.testing.assert_array_equal(stats.covariances, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        _ = stats.to_gaussian_mixture()
    mixture = stats.to_gaussian_mixture(regularizer=gd.covariance.DiagonalLoading(eps=0.1))
    np.testing.assert_allclose(mixture.covariances, [0.1 * np.eye(2)])
    np.testing.assert_array_equal(stats.covariances, np.zeros((1, 2, 2)))


def test_exactly_one_assignment_source_required() -> None:
    mixture = gd.GaussianMixture.from_components([gd.Gaussian.univariate()])
    with pytest.raises(ValueError, match="Exactly one"):
        _ = component_statistics([[0]])
    with pytest.raises(ValueError, match="Exactly one"):
        _ = component_statistics([[0]], responsibilities=[[1]], mixture=mixture)


@pytest.mark.parametrize("points", [[], [[np.nan]], [[np.inf]], [1, 2], [[]]])
def test_invalid_observations_rejected(points: npt.ArrayLike) -> None:
    with pytest.raises(ValueError, match="nonempty, finite 2D"):
        _ = component_statistics(points, responsibilities=[[1]])


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ([[1]], "shape"),
        ([1, 1], "shape"),
        (np.empty((2, 0)), "shape"),
        ([[np.nan], [1]], "finite and nonnegative"),
        ([[-1, 2], [0.5, 0.5]], "finite and nonnegative"),
        ([[0.1, 0.1], [0.5, 0.5]], "sum to one"),
        ([[1, 0], [1, 0]], "zero responsibility mass"),
    ],
)
def test_invalid_assignments_rejected(assignments: npt.ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ = component_statistics([[0], [1]], responsibilities=assignments)
