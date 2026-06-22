from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence.covariance import (
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    LinearShrinkage,
    LowRank,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
    diagonal_loading,
    estimate_epsilon,
    regularize_covariance,
)

if TYPE_CHECKING:
    from gmm_divergence.covariance import CovarianceRegularizer, EpsilonMethod


def test_regularize_covariance_supports_string_methods_and_top_level_exports() -> None:
    covariance = [[1.0, 0.2], [0.2, 0.5]]

    regularized = gd.regularize_covariance(covariance)

    assert isinstance(gd.covariance, object)
    assert isinstance(gd.DiagonalLoading(), DiagonalLoading)
    assert isinstance(gd.RelativeToTrace(), RelativeToTrace)
    assert regularized.shape == (2, 2)
    assert regularized.dtype == np.float64
    assert regularized[0, 0] == pytest.approx(1.0 + 1e-6)
    assert regularized[1, 1] == pytest.approx(0.5 + 1e-6)


def test_estimate_epsilon_relative_trace_matches_mean_variance_scale() -> None:
    covariance = np.diag([2.0, 8.0]).astype(np.float64)

    eps = gd.estimate_epsilon(covariance, method=gd.RelativeToTrace(c=0.1))

    assert eps == pytest.approx(0.5)


def test_regularize_covariance_accepts_diagonal_loading_epsilon_strategy() -> None:
    covariance = np.diag([2.0, 8.0]).astype(np.float64)

    regularized = regularize_covariance(
        covariance, method=DiagonalLoading(eps=RelativeToTrace(c=0.1))
    )

    assert regularized == pytest.approx(np.diag([2.5, 8.5]))


def test_estimate_epsilon_target_condition_number_supports_batches() -> None:
    covariances = np.array(
        [[[1.0, 0.0], [0.0, 100.0]], [[2.0, 0.0], [0.0, 18.0]]], dtype=np.float64
    )

    eps = estimate_epsilon(covariances, method=TargetConditionNumber(kappa=10.0))

    assert eps == pytest.approx([10.0, 0.0])


def test_regularize_covariance_infers_batched_shape_and_applies_option_objects() -> None:
    covariances = np.array([[[1.0, 0.0], [0.0, 0.01]], [[0.5, 0.0], [0.0, 0.25]]], dtype=np.float64)

    regularized = regularize_covariance(covariances, method=EigenvalueClipping(min_eigenvalue=0.1))

    assert regularized.shape == covariances.shape
    assert float(np.min(np.linalg.eigvalsh(regularized))) >= 0.1 - 1e-12


def test_diagonal_loading_accepts_array_like_input_without_explicit_batch_flag() -> None:
    regularized = diagonal_loading([[1.0, 0.0], [0.0, 2.0]], eps=1e-3)

    assert regularized == pytest.approx(np.diag([1.001, 2.001]))


def test_regularize_covariance_lowrank_returns_positive_definite_matrix() -> None:
    covariance = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)

    regularized = regularize_covariance(covariance, method=LowRank(rank=1, eps=1e-4))

    eigenvalues = np.linalg.eigvalsh(regularized)
    assert regularized.shape == covariance.shape
    assert np.all(eigenvalues > 0.0)


def test_lowrank_supports_residual_variance_epsilon_strategy() -> None:
    covariance = np.diag([5.0, 2.0, 0.5]).astype(np.float64)

    regularized = regularize_covariance(covariance, method=LowRank(rank=1, eps=ResidualVariance()))

    assert regularized == pytest.approx(np.diag([6.25, 1.25, 1.25]))


def test_estimate_epsilon_rejects_residual_variance_without_rank() -> None:
    covariance = np.eye(2)

    with pytest.raises(ValueError, match=r"ResidualVariance.r must be provided"):
        _ = estimate_epsilon(covariance, method=ResidualVariance())


def test_estimate_epsilon_supports_residual_variance_rank_on_option() -> None:
    covariance = np.diag([5.0, 2.0, 0.5]).astype(np.float64)

    eps = estimate_epsilon(covariance, method=ResidualVariance(r=1))

    assert eps == pytest.approx(1.25)


def test_lowrank_rejects_mismatched_residual_variance_rank() -> None:
    covariance = np.diag([5.0, 2.0, 0.5]).astype(np.float64)

    with pytest.raises(ValueError, match=r"ResidualVariance.r must match"):
        _ = regularize_covariance(covariance, method=LowRank(rank=1, eps=ResidualVariance(r=2)))


def test_estimate_epsilon_rejects_unknown_methods() -> None:
    covariance = np.eye(2)

    with pytest.raises(ValueError, match="Unknown covariance epsilon heuristic method"):
        _ = estimate_epsilon(
            covariance, method=cast("EpsilonMethod", cast("object", "not-a-method"))
        )


def test_regularize_covariance_rejects_unknown_methods() -> None:
    covariance = np.eye(2)

    with pytest.raises(ValueError, match="Unknown covariance regularizer method"):
        _ = regularize_covariance(
            covariance, method=cast("CovarianceRegularizer", cast("object", "not-a-method"))
        )


def test_covariance_options_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="eps must be a nonnegative finite value"):
        _ = DiagonalLoading(eps=-1.0)

    with pytest.raises(ValueError, match=r"alpha must be a finite value in \[0, 1\]"):
        _ = LinearShrinkage(alpha=1.5)

    with pytest.raises(ValueError, match=r"alpha must be a finite value in \[0, 1\]"):
        _ = DiagonalShrinkage(alpha=-0.1)

    with pytest.raises(ValueError, match="min_eigenvalue must be a positive finite value"):
        _ = EigenvalueClipping(min_eigenvalue=0.0)

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        _ = LowRank(rank=0)
