from __future__ import annotations

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
    estimate_epsilon,
    regularize_covariance,
)


def test_regularize_covariance_requires_an_explicit_configuration() -> None:
    covariance = np.array([[1.0, 0.2], [0.2, 0.5]])
    regularized = regularize_covariance(covariance, regularizer=DiagonalLoading())
    assert regularized == pytest.approx(np.array([[1.000001, 0.2], [0.2, 0.500001]]))


def test_epsilon_heuristics_are_explicit() -> None:
    covariance = np.diag([2.0, 8.0])
    epsilon = estimate_epsilon(covariance, heuristic=RelativeToTrace(c=0.1))
    assert epsilon == pytest.approx(0.5)

    batched = np.array([[[1.0, 0.0], [0.0, 100.0]], [[2.0, 0.0], [0.0, 18.0]]])
    assert estimate_epsilon(batched, heuristic=TargetConditionNumber(kappa=10.0)) == pytest.approx([
        10.0,
        0.0,
    ])


@pytest.mark.parametrize(
    "regularizer",
    [
        DiagonalLoading(eps=RelativeToTrace(c=0.1)),
        DiagonalShrinkage(alpha=0.1),
        LinearShrinkage(alpha=0.1),
        EigenvalueClipping(min_eigenvalue=0.1),
        LowRank(rank=1, eps=1e-4),
    ],
)
def test_regularizers_accept_explicit_objects(
    regularizer: gd.covariance.CovarianceRegularizer,
) -> None:
    covariance = np.array([[3.0, 1.0], [1.0, 2.0]])
    original = covariance.copy()
    regularized = regularize_covariance(covariance, regularizer=regularizer)

    assert covariance == pytest.approx(original)
    assert not np.shares_memory(regularized, covariance)
    assert not regularized.flags.writeable
    assert regularized.shape == covariance.shape
    assert np.all(np.linalg.eigvalsh(regularized) > 0.0)


def test_residual_variance_uses_the_enclosing_low_rank_configuration() -> None:
    covariance = np.diag([5.0, 2.0, 0.5])
    regularized = regularize_covariance(
        covariance, regularizer=LowRank(rank=1, eps=ResidualVariance())
    )
    assert regularized == pytest.approx(np.diag([6.25, 1.25, 1.25]))


def test_regularized_distribution_constructors_require_a_regularizer() -> None:
    gaussian = gd.Gaussian.from_regularized_arrays(
        [0.0], [[0.0]], regularizer=DiagonalLoading(eps=1e-3)
    )
    mixture = gd.GaussianMixture.from_regularized_arrays(
        [1.0], [[0.0]], [[[0.0]]], regularizer=DiagonalLoading(eps=1e-3)
    )
    assert gaussian.covariance == pytest.approx(np.array([[1e-3]]))
    assert mixture.covariances == pytest.approx(np.array([[[1e-3]]]))
