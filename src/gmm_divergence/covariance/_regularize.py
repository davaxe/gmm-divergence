from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from gmm_divergence._core._validation import as_covariance, as_covariances
from gmm_divergence.covariance._epsilon import (
    EpsilonMethod,
    EpsilonSpec,
    ResidualVariance,
    estimate_epsilon,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import Covariance, Covariances, FloatArray


@overload
def diagonal_loading(
    covariance: npt.ArrayLike,
    *,
    eps: EpsilonSpec = 1e-6,
    batched: Literal[False] = False,
    copy: bool = True,
) -> Covariance: ...


@overload
def diagonal_loading(
    covariance: npt.ArrayLike, eps: EpsilonSpec = 1e-6, *, batched: Literal[True], copy: bool = True
) -> Covariances: ...


@overload
def diagonal_loading(
    covariance: npt.ArrayLike, *, eps: EpsilonSpec = 1e-6, batched: None = None, copy: bool = True
) -> Covariance | Covariances: ...


def diagonal_loading(
    covariance: npt.ArrayLike,
    eps: EpsilonSpec = 1e-6,
    *,
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Apply diagonal loading to a covariance matrix or batch of matrices."""
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    if copy:
        covariance_arr = covariance_arr.copy()

    match _check_covariance_shape(covariance_arr, batched=batched):
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
    covariance: npt.ArrayLike,
    alpha: float = 1e-6,
    *,
    batched: Literal[False] = False,
    copy: bool = True,
) -> Covariance: ...


@overload
def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[True], copy: bool = True
) -> Covariances: ...


@overload
def linear_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: None = None, copy: bool = True
) -> Covariance | Covariances: ...


def linear_shrinkage(
    covariance: npt.ArrayLike,
    alpha: float = 1e-6,
    *,
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Shrink a covariance toward an isotropic target."""
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    if not (0 <= alpha <= 1.0):
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)
    if copy:
        covariance_arr = covariance_arr.copy()

    match _check_covariance_shape(covariance_arr, batched=batched):
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
    covariance: npt.ArrayLike,
    alpha: float = 1e-6,
    *,
    batched: Literal[False] = False,
    copy: bool = True,
) -> Covariance: ...


@overload
def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: Literal[True], copy: bool = True
) -> Covariances: ...


@overload
def diagonal_shrinkage(
    covariance: npt.ArrayLike, alpha: float = 1e-6, *, batched: None = None, copy: bool = True
) -> Covariance | Covariances: ...


def diagonal_shrinkage(
    covariance: npt.ArrayLike,
    alpha: float = 1e-6,
    *,
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Shrink a covariance toward its diagonal."""
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    if not (0 <= alpha <= 1.0):
        msg = f"alpha must be in the range [0, 1], got {alpha}."
        raise ValueError(msg)
    if copy:
        covariance_arr = covariance_arr.copy()

    match _check_covariance_shape(covariance_arr, batched=batched):
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
    covariance: npt.ArrayLike,
    min_eigenvalue: float = 1e-6,
    *,
    batched: Literal[False] = False,
    copy: bool = True,
) -> Covariance: ...


@overload
def eigenvalue_clipping(
    covariance: npt.ArrayLike,
    min_eigenvalue: float = 1e-6,
    *,
    batched: Literal[True],
    copy: bool = True,
) -> Covariances: ...


@overload
def eigenvalue_clipping(
    covariance: npt.ArrayLike,
    min_eigenvalue: float = 1e-6,
    *,
    batched: None = None,
    copy: bool = True,
) -> Covariance | Covariances: ...


def eigenvalue_clipping(
    covariance: npt.ArrayLike,
    min_eigenvalue: float = 1e-6,
    *,
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Clip covariance eigenvalues from below."""
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    if min_eigenvalue <= 0.0 or not np.isfinite(min_eigenvalue):
        msg = f"min_eigenvalue must be a positive finite value, got {min_eigenvalue}."
        raise ValueError(msg)
    if copy:
        covariance_arr = covariance_arr.copy()

    match _check_covariance_shape(covariance_arr, batched=batched):
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
    copy: bool = True,
) -> Covariance: ...


@overload
def lowrank(
    covariance: npt.ArrayLike,
    rank: int,
    eps: EpsilonSpec = 1e-6,
    *,
    batched: Literal[True],
    copy: bool = True,
) -> Covariances: ...


@overload
def lowrank(
    covariance: npt.ArrayLike,
    rank: int,
    eps: EpsilonSpec = 1e-6,
    *,
    batched: None = None,
    copy: bool = True,
) -> Covariance | Covariances: ...


def lowrank(
    covariance: npt.ArrayLike,
    rank: int,
    eps: EpsilonSpec = 1e-6,
    *,
    batched: bool | None = None,
    copy: bool = True,
) -> Covariance | Covariances:
    """Approximate a covariance with a low-rank structure plus diagonal loading."""
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    if rank <= 0:
        msg = f"rank must be a positive integer, got {rank}."
        raise ValueError(msg)
    if copy:
        covariance_arr = covariance_arr.copy()

    match _check_covariance_shape(covariance_arr, batched=batched):
        case "single":
            resolved_eps = _resolve_epsilon(covariance_arr, eps, batched=False, rank=rank)
            covariance_arr = 0.5 * (covariance_arr + covariance_arr.T)
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
        eps = cast("EpsilonMethod", eps)
        if isinstance(eps, ResidualVariance) and rank is not None:
            if eps.r is None:
                eps = ResidualVariance(c=eps.c, r=rank)
            elif eps.r != rank:
                msg = (
                    "ResidualVariance.r must match the enclosing low-rank rank, "
                    f"got ResidualVariance.r={eps.r} and rank={rank}."
                )
                raise ValueError(msg)
        resolved_eps = estimate_epsilon(covariance, method=eps, batched=batched)
    _validate_resolved_epsilon(resolved_eps)
    return resolved_eps


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


def _check_covariance_shape(
    covariance: FloatArray, *, batched: bool | None = None
) -> Literal["single", "batched"]:
    if covariance.ndim == 2:
        if batched is True:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)

        if covariance.shape[0] != covariance.shape[1]:
            msg = f"Expected covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "single"

    if covariance.ndim == 3:
        if batched is False:
            msg = f"Expected single covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)

        if covariance.shape[1] != covariance.shape[2]:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "batched"

    msg = f"covariance must have shape (d, d) or (n, d, d), got {covariance.shape}."
    raise ValueError(msg)
