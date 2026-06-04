from __future__ import annotations

import numpy as np
import pytest

from gmm_divergence import Gaussian, GaussianMixture


def test_gaussian_mixture_normalizes_weights() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[2.0, 3.0], means=[[0.0], [1.0]], covariances=[[[1.0]], [[2.0]]]
    )

    np.testing.assert_allclose(mixture.weights, [0.4, 0.6])
    assert not mixture.weights.flags.writeable


def test_gaussian_mixture_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _ = GaussianMixture.from_arrays(
            weights=[0.5, -0.5], means=[[0.0], [1.0]], covariances=[[[1.0]], [[2.0]]]
        )


def test_gaussian_expands_diagonal_covariance() -> None:
    gaussian = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=[1.0, 2.0])

    np.testing.assert_allclose(gaussian.covariance, np.diag([1.0, 2.0]))
    assert not gaussian.covariance.flags.writeable


def test_gaussian_rejects_non_positive_definite_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        _ = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=[[1.0, 2.0], [2.0, 1.0]])


def test_gaussian_mixture_rejects_non_symmetric_covariance() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        _ = GaussianMixture.from_arrays(
            weights=[1.0], means=[[0.0, 1.0]], covariances=[[[1.0, 0.5], [0.0, 1.0]]]
        )
