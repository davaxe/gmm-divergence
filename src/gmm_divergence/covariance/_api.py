"""Explicit covariance-regularization API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from gmm_divergence.covariance._options import (
    CovarianceRegularizer,
    DiagonalLoading,
    DiagonalShrinkage,
    EigenvalueClipping,
    LinearShrinkage,
    LowRank,
)
from gmm_divergence.covariance._regularize import (
    diagonal_loading,
    diagonal_shrinkage,
    eigenvalue_clipping,
    linear_shrinkage,
    lowrank,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import Covariance, Covariances


@overload
def regularize_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    regularizer: CovarianceRegularizer,
    batched: Literal[False] = False,
) -> Covariance: ...


@overload
def regularize_covariance(
    covariance: npt.ArrayLike, /, *, regularizer: CovarianceRegularizer, batched: Literal[True]
) -> Covariances: ...


@overload
def regularize_covariance(
    covariance: npt.ArrayLike, /, *, regularizer: CovarianceRegularizer, batched: None = None
) -> Covariance | Covariances: ...


def regularize_covariance(
    covariance: npt.ArrayLike, /, *, regularizer: CovarianceRegularizer, batched: bool | None = None
) -> Covariance | Covariances:
    """Regularize one covariance matrix or a batch with an explicit configuration.

    Parameters
    ----------
    covariance : array-like
        A matrix of shape ``(d, d)`` or a batch of shape ``(n, d, d)``.
    regularizer : CovarianceRegularizer
        Explicit regularization configuration, such as ``DiagonalLoading()``
        or ``LowRank(rank=2)``.
    batched : bool or None, default=None
        Whether the input is batched. If omitted, infer this from its rank.

    Notes
    -----
    The input is never modified. The returned covariance is an independent,
    read-only ``float64`` array.
    """
    match regularizer:
        case DiagonalLoading(eps=eps):
            return diagonal_loading(covariance, eps=eps, batched=batched)
        case LinearShrinkage(alpha=alpha):
            return linear_shrinkage(covariance, alpha=alpha, batched=batched)
        case DiagonalShrinkage(alpha=alpha):
            return diagonal_shrinkage(covariance, alpha=alpha, batched=batched)
        case EigenvalueClipping(min_eigenvalue=min_eigenvalue):
            return eigenvalue_clipping(covariance, min_eigenvalue=min_eigenvalue, batched=batched)
        case LowRank(rank=rank, eps=eps):
            return lowrank(covariance, rank=rank, eps=eps, batched=batched)
