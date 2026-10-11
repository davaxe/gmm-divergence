from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import symmetrize
from gmm_divergence._core._validation import (
    as_covariance,
    as_covariances,
    validate_nonnegative_finite,
    validate_positive_finite,
    validate_positive_int,
    validate_unit_interval,
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


def _diagonal_loading(
    covariance: npt.ArrayLike, epsilon: EpsilonSpec = 1e-6
) -> Covariance | Covariances:
    """Apply diagonal loading without modifying the input."""
    covariance_arr = _prepare_covariance(covariance)
    resolved_eps = _resolve_epsilon(covariance_arr, epsilon)
    _apply_diagonal_loading(covariance_arr, resolved_eps)
    return _finish_covariance(covariance_arr)


def _linear_shrinkage(
    covariance: npt.ArrayLike, strength: float = 1e-2
) -> Covariance | Covariances:
    """Shrink a covariance toward an isotropic target without modifying the input."""
    covariance_arr = _prepare_covariance(covariance)
    _ = validate_unit_interval(strength, name="strength")
    n_features = covariance_arr.shape[-1]
    scale = np.trace(covariance_arr, axis1=-2, axis2=-1) / n_features
    shrinkage_target = np.eye(n_features) * scale[..., None, None]
    covariance_arr += strength * (shrinkage_target - covariance_arr)
    return _finish_covariance(covariance_arr)


def _diagonal_shrinkage(
    covariance: npt.ArrayLike, strength: float = 1e-2
) -> Covariance | Covariances:
    """Shrink a covariance toward its diagonal without modifying the input."""
    covariance_arr = _prepare_covariance(covariance)
    _ = validate_unit_interval(strength, name="strength")
    diagonal = np.diagonal(covariance_arr, axis1=-2, axis2=-1)
    target = diagonal[..., :, None] * np.eye(covariance_arr.shape[-1])
    covariance_arr += strength * (target - covariance_arr)
    return _finish_covariance(covariance_arr)


def _eigenvalue_clipping(
    covariance: npt.ArrayLike, min_eigenvalue: float = 1e-6
) -> Covariance | Covariances:
    """Clip covariance eigenvalues from below without modifying the input."""
    _ = validate_positive_finite(min_eigenvalue, name="min_eigenvalue")
    covariance_arr = _prepare_covariance(covariance)
    eigvals, eigvecs = np.linalg.eigh(covariance_arr)
    clipped = np.maximum(eigvals, min_eigenvalue)
    covariance_arr = (eigvecs * clipped[..., None, :]) @ eigvecs.swapaxes(-1, -2)
    return _finish_covariance(covariance_arr)


def _lowrank(
    covariance: npt.ArrayLike, rank: int, epsilon: EpsilonSpec = 1e-6
) -> Covariance | Covariances:
    """Return a low-rank approximation without modifying the input."""
    _ = validate_positive_int(rank, name="rank")
    covariance_arr = _prepare_covariance(covariance)
    n_features = covariance_arr.shape[-1]
    if rank > n_features:
        msg = f"rank must not exceed covariance dimension {n_features}, got {rank}."
        raise ValueError(msg)
    resolved_eps = _resolve_epsilon(covariance_arr, epsilon, rank=rank)
    eigvals, eigvecs = np.linalg.eigh(covariance_arr)
    indices = np.argsort(eigvals, axis=-1)[..., ::-1][..., :rank]
    top_values = np.take_along_axis(eigvals, indices, axis=-1)
    top_vectors = np.take_along_axis(eigvecs, indices[..., None, :], axis=-1)
    covariance_arr = (top_vectors * top_values[..., None, :]) @ top_vectors.swapaxes(-1, -2)
    _apply_diagonal_loading(covariance_arr, resolved_eps)
    return _finish_covariance(covariance_arr)


def _resolve_epsilon(
    covariance: FloatArray, epsilon: EpsilonSpec, *, rank: int | None = None
) -> float | FloatArray:
    if isinstance(epsilon, bool):
        msg = "epsilon must be a real scalar or epsilon heuristic."
        raise TypeError(msg)
    if isinstance(epsilon, Real):
        resolved_eps = validate_nonnegative_finite(float(epsilon), name="epsilon")
    else:
        if not isinstance(epsilon, (RelativeToTrace, TargetConditionNumber, ResidualVariance)):
            msg = (
                f"epsilon must be a nonnegative scalar or epsilon heuristic, got "
                f"{type(epsilon).__name__}."
            )
            raise TypeError(msg)
        if isinstance(epsilon, ResidualVariance) and rank is not None:
            if epsilon.rank is None:
                epsilon = ResidualVariance(scale=epsilon.scale, rank=rank)
            elif epsilon.rank != rank:
                msg = (
                    "ResidualVariance.rank must match the enclosing low-rank rank, "
                    f"got ResidualVariance.rank={epsilon.rank} and rank={rank}."
                )
                raise ValueError(msg)
        resolved_eps = epsilon_from_covariance(covariance, epsilon)
    _validate_resolved_epsilon(resolved_eps)
    return resolved_eps


def _prepare_covariance(covariance: npt.ArrayLike) -> FloatArray:
    array = np.asarray(covariance, dtype=np.float64)
    _ = validate_covariance_input(array)
    return symmetrize(array)


def _finish_covariance(array: FloatArray) -> Covariance | Covariances:
    if array.ndim == 2:
        return as_covariance(array, n_features=array.shape[-1])
    return as_covariances(array, n_components=array.shape[0], n_features=array.shape[-1])


def _validate_resolved_epsilon(epsilon: float | FloatArray) -> None:
    eps_arr = np.asarray(epsilon, dtype=np.float64)
    if eps_arr.ndim > 1:
        msg = f"Resolved epsilon must be a scalar or 1D array, got shape {eps_arr.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(eps_arr)):
        msg = "Resolved epsilon must contain only finite values."
        raise ValueError(msg)
    if np.any(eps_arr < 0.0):
        msg = "Resolved epsilon must be nonnegative."
        raise ValueError(msg)


def _apply_diagonal_loading(covariance: FloatArray, epsilon: float | FloatArray) -> None:
    eps_arr = np.asarray(epsilon, dtype=np.float64)
    if eps_arr.ndim != 0 and eps_arr.shape != covariance.shape[:-2]:
        msg = f"Epsilon must be scalar or have shape {covariance.shape[:-2]}, got {eps_arr.shape}."
        raise ValueError(msg)
    idx = np.arange(covariance.shape[-1], dtype=np.intp)
    covariance[..., idx, idx] += eps_arr[..., None]


__all__ = [
    "_diagonal_loading",
    "_diagonal_shrinkage",
    "_eigenvalue_clipping",
    "_linear_shrinkage",
    "_lowrank",
]
