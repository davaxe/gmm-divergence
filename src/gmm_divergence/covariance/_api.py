from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from gmm_divergence._core._dispatch import MethodSpec, Registry, cast_options
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

_REGISTRY = Registry(
    label="covariance regularizer",
    specs=(
        MethodSpec(name="diagonal_loading", option_type=DiagonalLoading, default=DiagonalLoading()),
        MethodSpec(name="linear_shrinkage", option_type=LinearShrinkage, default=LinearShrinkage()),
        MethodSpec(
            name="diagonal_shrinkage", option_type=DiagonalShrinkage, default=DiagonalShrinkage()
        ),
        MethodSpec(
            name="eigenvalue_clipping", option_type=EigenvalueClipping, default=EigenvalueClipping()
        ),
        MethodSpec(name="lowrank", option_type=LowRank, default=LowRank()),
    ),
)


@overload
def regularize_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    method: CovarianceRegularizer = "diagonal_loading",
    batched: Literal[False] = False,
    copy: bool = True,
) -> Covariance: ...


@overload
def regularize_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    method: CovarianceRegularizer = "diagonal_loading",
    batched: Literal[True],
    copy: bool = True,
) -> Covariances: ...


@overload
def regularize_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    method: CovarianceRegularizer = "diagonal_loading",
    batched: None = None,
    copy: bool = True,
) -> Covariance | Covariances: ...


def regularize_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    method: CovarianceRegularizer = "diagonal_loading",
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Regularize a covariance matrix or a batch of covariance matrices.

    The covariances are validated and regularized according to the specified
    method. The input can be either a single covariance matrix with shape ``(d,
    d)` or a batch of covariance matrices with shape `(n, d, d)``. If the
    batch flag is not explicitly provided, the shape of the input is used to infer whether it is
    batched or not.

    Parameters
    ----------
    covariance : array-like
        A covariance matrix with shape `(d, d)` or a batch with shape
        `(n, d, d)`.
    method : str or covariance regularizer configuration, default="diagonal_loading"
        Regularization method to apply. Passing a string uses the method's
        default configuration. `DiagonalLoading` and `LowRank` also accept
        epsilon heuristic strategy objects through their `eps` field.
    batched : bool or None, default=None
        Whether to interpret the input as batched. If `None`, the shape is
        inferred from the array rank.
    copy : bool, default=True
        Whether to regularize a copy of the input.

    Returns
    -------
    Covariance or batch of covariances
        Regularized covariance matrix or matrices.
    """
    spec, options = _REGISTRY.resolve(method)

    match spec.name:
        case "diagonal_loading":
            options = cast_options(options, DiagonalLoading)
            return diagonal_loading(covariance, eps=options.eps, batched=batched, copy=copy)
        case "linear_shrinkage":
            options = cast_options(options, LinearShrinkage)
            return linear_shrinkage(covariance, alpha=options.alpha, batched=batched, copy=copy)
        case "diagonal_shrinkage":
            options = cast_options(options, DiagonalShrinkage)
            return diagonal_shrinkage(covariance, alpha=options.alpha, batched=batched, copy=copy)
        case "eigenvalue_clipping":
            options = cast_options(options, EigenvalueClipping)
            return eigenvalue_clipping(
                covariance, min_eigenvalue=options.min_eigenvalue, batched=batched, copy=copy
            )
        case "lowrank":
            options = cast_options(options, LowRank)
            return lowrank(
                covariance, rank=options.rank, eps=options.eps, batched=batched, copy=copy
            )
        case _:
            msg = "Unhandled covariance regularizer registry entry."
            raise AssertionError(msg)
