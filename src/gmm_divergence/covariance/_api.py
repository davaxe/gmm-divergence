# pyright: reportPrivateUsage=false
# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false, reportUnnecessaryComparison=false
"""Configuration-driven covariance regularization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gmm_divergence.covariance._options import (
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    IsotropicShrinkage,
    LowRank,
)
from gmm_divergence.covariance._regularize import (
    _diagonal_loading,
    _diagonal_shrinkage,
    _eigenvalue_clipping,
    _linear_shrinkage,
    _lowrank,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.covariance._options import CovarianceRegularizer


def regularize_covariance(
    covariance: npt.ArrayLike, *, regularizer: CovarianceRegularizer
) -> FloatArray:
    """Regularize a single covariance matrix or a batch of matrices.

    The shape determines batching. Returns an independent, read-only float64
    array of the same shape. Inputs may be singular, but the regularized
    matrices must be positive definite.

    Parameters
    ----------
    covariance : array-like, shape (n_features, n_features) or (n_matrices, n_features, n_features)
        Nonempty, finite, symmetric covariance matrix or batch. Symmetry is
        checked with NumPy's default allclose tolerance, then roundoff asymmetry
        is averaged out. Positive definiteness is checked after regularization.
    regularizer : CovarianceRegularizer
        Explicit loading, shrinkage, clipping, or low-rank configuration.
        Epsilon heuristics are resolved separately for each covariance.

    Returns
    -------
    ndarray
        Independent, read-only float64 array matching the input shape.

    Raises
    ------
    TypeError
        If regularizer is unsupported.

    Notes
    -----
    Invalid shapes, values, ranks, or non-positive-definite results raise
    ValueError through covariance validation. Regularization does not guarantee
    positive definiteness for every input and configuration.
    """
    match regularizer:
        case DiagonalLoading(epsilon=epsilon):
            return _diagonal_loading(covariance, epsilon=epsilon)
        case IsotropicShrinkage(strength=strength):
            return _linear_shrinkage(covariance, strength=strength)
        case DiagonalShrinkage(strength=strength):
            return _diagonal_shrinkage(covariance, strength=strength)
        case EigenvalueClipping(min_eigenvalue=min_eigenvalue):
            return _eigenvalue_clipping(covariance, min_eigenvalue=min_eigenvalue)
        case LowRank(rank=rank, epsilon=epsilon):
            return _lowrank(covariance, rank=rank, epsilon=epsilon)
        case _:
            msg = f"regularizer must be a CovarianceRegularizer, got {type(regularizer).__name__}."
            raise TypeError(msg)
