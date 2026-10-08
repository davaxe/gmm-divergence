from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from gmm_divergence._core._numeric import symmetrize
from gmm_divergence._core._validation import (
    as_covariance,
    as_covariances,
    validate_positive_finite,
    validate_positive_int,
)
from gmm_divergence.covariance._epsilon import (
    EpsilonSpec,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
    epsilon_from_covariance,
)
from gmm_divergence.covariance._shape import validate_covariance_input

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import Covariance, Covariances, FloatArray


@overload
def diagonal_loading(
    covariance: npt.ArrayLike, *, eps: EpsilonSpec = 1e-6, batched: Literal[False] = False
) -> Covariance: ...


@overload
def diagonal_loading(
    covariance: npt.ArrayLike, eps: EpsilonSpec = 1e-6, *, batched: Literal[True]
) -> Covariances: ...


@overload
def diagonal_loading(
    covariance: npt.ArrayLike, *, eps: EpsilonSpec = 1e-6, batched: None = None
) -> Covariance | Covariances: ...


def diagonal_loading(
    covariance: npt.ArrayLike, eps: EpsilonSpec = 1e-6, *, batched: bool | None = None
) -> Covariance | Covariances:
    """Apply diagonal loading without modifying the input."""
    covariance_arr = _prepare_covariance(covariance, batched=batched)
    resolved_eps = _resolve_epsilon(covariance_arr, eps)
    _apply_diagonal_loading(covariance_arr, resolved_eps)
    return _finish_covariance(covariance_arr)


@overload
def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[False] = False
) -> Covariance: ...


@overload
def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[True]
) -> Covariances: ...


@overload
def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: None = None
) -> Covariance | Covariances: ...


def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: bool | None = None
) -> Covariance | Covariances:
    """Shrink a covariance toward an isotropic target without modifying the input."""
    covariance_arr = _prepare_covariance(covariance, batched=batched)
    if not 0 <= alpha <= 1.0:
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)
    dim = covariance_arr.shape[-1]
    scale = np.trace(covariance_arr, axis1=-2, axis2=-1) / dim
    shrinkage_target = np.eye(dim) * scale[..., None, None]
    covariance_arr += alpha * (shrinkage_target - covariance_arr)
    return _finish_covariance(covariance_arr)


@overload
def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[False] = False
) -> Covariance: ...


@overload
def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[True]
) -> Covariances: ...


@overload
def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: None = None
) -> Covariance | Covariances: ...


def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: bool | None = None
) -> Covariance | Covariances:
    """Shrink a covariance toward its diagonal without modifying the input."""
    covariance_arr = _prepare_covariance(covariance, batched=batched)
    if not 0 <= alpha <= 1.0:
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)
    diagonal = np.diagonal(covariance_arr, axis1=-2, axis2=-1)
    target = diagonal[..., :, None] * np.eye(covariance_arr.shape[-1])
    covariance_arr += alpha * (target - covariance_arr)
    return _finish_covariance(covariance_arr)


@overload
def eigenvalue_clipping(
    covariance: npt.ArrayLike, min_eigenvalue: float = 1e-6, *, batched: Literal[False] = False
) -> Covariance: ...


@overload
def eigenvalue_clipping(
    covariance: npt.ArrayLike, min_eigenvalue: float = 1e-6, *, batched: Literal[True]
) -> Covariances: ...


@overload
def eigenvalue_clipping(
    covariance: npt.ArrayLike, min_eigenvalue: float = 1e-6, *, batched: None = None
) -> Covariance | Covariances: ...


def eigenvalue_clipping(
    covariance: npt.ArrayLike, min_eigenvalue: float = 1e-6, *, batched: bool | None = None
) -> Covariance | Covariances:
    """Clip covariance eigenvalues from below without modifying the input."""
    _ = validate_positive_finite(min_eigenvalue, name="min_eigenvalue")
    covariance_arr = _prepare_covariance(covariance, batched=batched)
    eigvals, eigvecs = np.linalg.eigh(covariance_arr)
    clipped = np.maximum(eigvals, min_eigenvalue)
    covariance_arr = (eigvecs * clipped[..., None, :]) @ eigvecs.swapaxes(-1, -2)
    return _finish_covariance(covariance_arr)


@overload
def lowrank(
    covariance: npt.ArrayLike,
    rank: int,
    eps: EpsilonSpec = 1e-6,
    *,
    batched: Literal[False] = False,
) -> Covariance: ...


@overload
def lowrank(
    covariance: npt.ArrayLike, rank: int, eps: EpsilonSpec = 1e-6, *, batched: Literal[True]
) -> Covariances: ...


@overload
def lowrank(
    covariance: npt.ArrayLike, rank: int, eps: EpsilonSpec = 1e-6, *, batched: None = None
) -> Covariance | Covariances: ...


def lowrank(
    covariance: npt.ArrayLike, rank: int, eps: EpsilonSpec = 1e-6, *, batched: bool | None = None
) -> Covariance | Covariances:
    """Return a low-rank approximation without modifying the input."""
    _ = validate_positive_int(rank, name="rank")
    covariance_arr = _prepare_covariance(covariance, batched=batched)
    dim = covariance_arr.shape[-1]
    if rank > dim:
        msg = f"rank must not exceed covariance dimension {dim}, got {rank}."
        raise ValueError(msg)
    resolved_eps = _resolve_epsilon(covariance_arr, eps, rank=rank)
    eigvals, eigvecs = np.linalg.eigh(covariance_arr)
    indices = np.argsort(eigvals, axis=-1)[..., ::-1][..., :rank]
    top_values = np.take_along_axis(eigvals, indices, axis=-1)
    top_vectors = np.take_along_axis(eigvecs, indices[..., None, :], axis=-1)
    covariance_arr = (top_vectors * top_values[..., None, :]) @ top_vectors.swapaxes(-1, -2)
    _apply_diagonal_loading(covariance_arr, resolved_eps)
    return _finish_covariance(covariance_arr)


def _resolve_epsilon(
    covariance: FloatArray, eps: EpsilonSpec, *, rank: int | None = None
) -> float | FloatArray:
    if isinstance(eps, Real):
        resolved_eps = float(eps)
    else:
        if not isinstance(eps, (RelativeToTrace, TargetConditionNumber, ResidualVariance)):
            msg = (
                f"eps must be a nonnegative scalar or epsilon heuristic, got {type(eps).__name__}."
            )
            raise TypeError(msg)
        if isinstance(eps, ResidualVariance) and rank is not None:
            if eps.r is None:
                eps = ResidualVariance(c=eps.c, r=rank)
            elif eps.r != rank:
                msg = (
                    "ResidualVariance.r must match the enclosing low-rank rank, "
                    f"got ResidualVariance.r={eps.r} and rank={rank}."
                )
                raise ValueError(msg)
        resolved_eps = epsilon_from_covariance(covariance, eps)
    _validate_resolved_epsilon(resolved_eps)
    return resolved_eps


def _prepare_covariance(covariance: npt.ArrayLike, *, batched: bool | None) -> FloatArray:
    array = np.asarray(covariance, dtype=np.float64)
    _ = validate_covariance_input(array, batched=batched)
    return symmetrize(array)


def _finish_covariance(array: FloatArray) -> Covariance | Covariances:
    if array.ndim == 2:
        return as_covariance(array, n_features=array.shape[-1])
    return as_covariances(array, n_components=array.shape[0], n_features=array.shape[-1])


def _validate_resolved_epsilon(eps: float | FloatArray) -> None:
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim > 1:
        msg = f"Resolved epsilon must be a scalar or 1D array, got shape {eps_arr.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(eps_arr)):
        msg = "Resolved epsilon must contain only finite values."
        raise ValueError(msg)
    if np.any(eps_arr < 0.0):
        msg = "Resolved epsilon must be nonnegative."
        raise ValueError(msg)


def _apply_diagonal_loading(covariance: FloatArray, eps: float | FloatArray) -> None:
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim != 0 and eps_arr.shape != covariance.shape[:-2]:
        msg = f"Epsilon must be scalar or have shape {covariance.shape[:-2]}, got {eps_arr.shape}."
        raise ValueError(msg)
    idx = np.arange(covariance.shape[-1], dtype=np.intp)
    covariance[..., idx, idx] += eps_arr[..., None]
