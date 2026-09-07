from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

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
    estimate_epsilon,
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
    covariance_arr = _mutable_covariance_copy(covariance)
    shape_kind = _validate_and_symmetrize(covariance_arr, batched=batched)

    match shape_kind:
        case "single":
            resolved_eps = _resolve_epsilon(covariance_arr, eps, batched=False)
            _apply_diagonal_loading_single(covariance_arr, resolved_eps)
            return as_covariance(covariance_arr, n_features=covariance_arr.shape[0])
        case "batched":
            resolved_eps = _resolve_epsilon(covariance_arr, eps, batched=True)
            _apply_diagonal_loading_batched(covariance_arr, resolved_eps)
            return as_covariances(
                covariance_arr,
                n_components=covariance_arr.shape[0],
                n_features=covariance_arr.shape[1],
            )


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
    covariance_arr = _mutable_covariance_copy(covariance)
    if not (0 <= alpha <= 1.0):
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)

    shape_kind = _validate_and_symmetrize(covariance_arr, batched=batched)
    match shape_kind:
        case "single":
            d = covariance_arr.shape[0]
            shrinkage_target = np.eye(d) * np.trace(covariance_arr) / d
            covariance_arr += alpha * (shrinkage_target - covariance_arr)
            return as_covariance(covariance_arr, n_features=d)
        case "batched":
            n, d, _ = covariance_arr.shape
            shrinkage_target = (
                np.eye(d)
                * (np.trace(covariance_arr, axis1=1, axis2=2) / d)[:, np.newaxis, np.newaxis]
            )
            covariance_arr += alpha * (shrinkage_target - covariance_arr)
            return as_covariances(covariance_arr, n_components=n, n_features=d)


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
    covariance_arr = _mutable_covariance_copy(covariance)
    if not (0 <= alpha <= 1.0):
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)

    shape_kind = _validate_and_symmetrize(covariance_arr, batched=batched)
    match shape_kind:
        case "single":
            diag = np.diag(covariance_arr)
            covariance_arr += alpha * (np.diag(diag) - covariance_arr)
            return as_covariance(covariance_arr, n_features=covariance_arr.shape[0])
        case "batched":
            diag = np.einsum("nii->ni", covariance_arr)
            covariance_arr += alpha * (
                np.einsum("ni,ij->nij", diag, np.eye(covariance_arr.shape[1])) - covariance_arr
            )
            return as_covariances(
                covariance_arr,
                n_components=covariance_arr.shape[0],
                n_features=covariance_arr.shape[1],
            )


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
    covariance_arr = _mutable_covariance_copy(covariance)
    validate_positive_finite(min_eigenvalue, name="min_eigenvalue")
    shape_kind = _validate_and_symmetrize(covariance_arr, batched=batched)

    match shape_kind:
        case "single":
            eigvals, eigvecs = np.linalg.eigh(covariance_arr)
            clipped_eigvals = np.clip(eigvals, a_min=min_eigenvalue, a_max=None)
            covariance_arr[:] = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
            return as_covariance(covariance_arr, n_features=covariance_arr.shape[0])
        case "batched":
            eigvals, eigvecs = np.linalg.eigh(covariance_arr)
            clipped_eigvals = np.clip(eigvals, a_min=min_eigenvalue, a_max=None)
            covariance_arr[:] = np.einsum(
                "nij,nj,njk->nik", eigvecs, clipped_eigvals, eigvecs.transpose(0, 2, 1)
            )
            return as_covariances(
                covariance_arr,
                n_components=covariance_arr.shape[0],
                n_features=covariance_arr.shape[1],
            )


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
    covariance_arr = _mutable_covariance_copy(covariance)
    validate_positive_int(rank, name="rank")
    shape_kind = _validate_and_symmetrize(covariance_arr, batched=batched)
    dim = covariance_arr.shape[-1]
    if rank > dim:
        msg = f"rank must not exceed covariance dimension {dim}, got {rank}."
        raise ValueError(msg)

    match shape_kind:
        case "single":
            resolved_eps = _resolve_epsilon(covariance_arr, eps, batched=False, rank=rank)
            eigvals, eigvecs = np.linalg.eigh(covariance_arr)
            idx = np.argsort(eigvals)[::-1][:rank]
            lowrank_cov = eigvecs[:, idx] @ np.diag(eigvals[idx]) @ eigvecs[:, idx].T
            _apply_diagonal_loading_single(lowrank_cov, resolved_eps)
            return as_covariance(lowrank_cov, n_features=lowrank_cov.shape[0])
        case "batched":
            resolved_eps = _resolve_epsilon(covariance_arr, eps, batched=True, rank=rank)
            eigvals, eigvecs = np.linalg.eigh(covariance_arr)
            idx = np.argsort(eigvals, axis=1)[:, ::-1][:, :rank]
            top_eigvals = np.take_along_axis(eigvals, idx, axis=1)
            top_eigvecs = np.take_along_axis(eigvecs, idx[:, None, :], axis=2)
            lowrank_cov = (top_eigvecs * top_eigvals[:, None, :]) @ top_eigvecs.swapaxes(-1, -2)
            _apply_diagonal_loading_batched(lowrank_cov, resolved_eps)
            return as_covariances(
                lowrank_cov, n_components=lowrank_cov.shape[0], n_features=lowrank_cov.shape[1]
            )


def _resolve_epsilon(
    covariance: FloatArray, eps: EpsilonSpec, *, batched: bool, rank: int | None = None
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
        resolved_eps = estimate_epsilon(covariance, heuristic=eps, batched=batched)
    _validate_resolved_epsilon(resolved_eps)
    return resolved_eps


def _mutable_covariance_copy(covariance: npt.ArrayLike) -> FloatArray:
    return np.array(covariance, dtype=np.float64, copy=True)


def _validate_and_symmetrize(
    covariance: FloatArray, *, batched: bool | None
) -> Literal["single", "batched"]:
    shape_kind = validate_covariance_input(covariance, batched=batched)
    covariance[:] = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    return shape_kind


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


def _apply_diagonal_loading_single(covariance: FloatArray, eps: float | FloatArray) -> None:
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim != 0:
        msg = f"Single-covariance epsilon must be scalar, got shape {eps_arr.shape}."
        raise ValueError(msg)
    idx = np.arange(covariance.shape[0], dtype=np.intp)
    covariance[idx, idx] += float(eps_arr)


def _apply_diagonal_loading_batched(covariance: FloatArray, eps: float | FloatArray) -> None:
    idx = np.arange(covariance.shape[1], dtype=np.intp)
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 0:
        covariance[:, idx, idx] += float(eps_arr)
        return
    if eps_arr.shape != (covariance.shape[0],):
        msg = f"Batched epsilon must have shape ({covariance.shape[0]},), got {eps_arr.shape}."
        raise ValueError(msg)
    covariance[:, idx, idx] += eps_arr[:, None]
