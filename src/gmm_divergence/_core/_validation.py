from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._numeric import symmetrize

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import Covariance, Covariances, FloatArray, Weights


def as_weights(
    weights: npt.ArrayLike,
    /,
    *,
    expected_length: int | None = None,
    normalize: bool = True,
    name: str = "Weights",
    writable: bool = False,
) -> Weights:
    """Return validated nonnegative finite weights."""
    weights_arr = np.asarray(weights, dtype=np.float64)

    if weights_arr.ndim != 1:
        msg = f"{name} must be a 1D array."
        raise ValueError(msg)

    if expected_length is not None and weights_arr.shape[0] != expected_length:
        msg = f"{name} must have length {expected_length}, got {weights_arr.shape[0]}."
        raise ValueError(msg)

    if weights_arr.shape[0] == 0:
        msg = f"{name} must contain at least one value."
        raise ValueError(msg)

    if not np.all(np.isfinite(weights_arr)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    negative_tolerance = 1e-12
    if np.any(weights_arr < -negative_tolerance):
        msg = f"{name} must be nonnegative."
        raise ValueError(msg)
    weights_arr = np.maximum(weights_arr, 0.0)

    weight_sum = float(np.sum(weights_arr))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        msg = f"{name} must sum to a positive finite value."
        raise ValueError(msg)

    weights_arr = weights_arr / weight_sum if normalize else weights_arr.copy()
    weights_arr.setflags(write=writable)
    return weights_arr


def as_covariance(
    covariance: npt.ArrayLike,
    /,
    *,
    n_features: int,
    name: str = "Covariance",
    writable: bool = False,
) -> Covariance:
    """Return a validated symmetric positive-definite covariance matrix."""
    return _as_covariance_array(covariance, (n_features, n_features), name=name, writable=writable)


def as_covariances(
    covariances: npt.ArrayLike,
    /,
    *,
    n_components: int,
    n_features: int,
    name: str = "Covariances",
    writable: bool = False,
) -> Covariances:
    """Return a validated stack of symmetric positive-definite covariances."""
    return _as_covariance_array(
        covariances, (n_components, n_features, n_features), name=name, writable=writable
    )


def _as_covariance_array(
    values: npt.ArrayLike, shape: tuple[int, ...], *, name: str, writable: bool
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        msg = f"{name} must have shape {shape}, got {array.shape}."
        raise ValueError(msg)
    _validate_symmetric(array, name=name)
    array = symmetrize(array)
    _validate_covariance_values(array, name=name)
    array.setflags(write=writable)
    return array


def as_probabilities(values: npt.ArrayLike, /, *, name: str = "Weights") -> FloatArray:
    """Validate a probability vector with sum tolerance rtol=1e-7, atol=1e-8.

    Values must be finite and nonnegative. No normalization or clipping is applied.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        msg = f"{name} must be a nonempty probability vector."
        raise ValueError(msg)
    _validate_probabilities(array, name=name)
    return readonly_copy(array, dtype=np.float64)


def as_probability_rows(values: npt.ArrayLike, /, *, name: str) -> FloatArray:
    """Validate nonempty probability rows using the same policy as probability vectors."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        msg = f"{name} must be a nonempty 2D array of probability rows."
        raise ValueError(msg)
    _validate_probabilities(array, name=name)
    return readonly_copy(array, dtype=np.float64)


def _validate_probabilities(array: FloatArray, *, name: str) -> None:
    if not np.isfinite(array).all() or (array < 0).any():
        msg = f"{name} must be finite and nonnegative."
        raise ValueError(msg)
    if not np.allclose(array.sum(axis=-1), 1.0, rtol=1e-7, atol=1e-8):
        msg = f"{name} must sum to one along the last axis."
        raise ValueError(msg)


def as_points(
    points: npt.ArrayLike,
    /,
    *,
    n_features: int,
    name: str = "Points",
    writable: bool = False,
    require_nonempty: bool = False,
) -> FloatArray:
    """Return validated points with shape `(n_points, n_features)`."""
    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.ndim == 1:
        points_arr = points_arr[None, :]

    if points_arr.ndim != 2 or points_arr.shape[1] != n_features:
        msg = f"{name} must have shape (n_points, {n_features}), got {points_arr.shape}."
        raise ValueError(msg)

    if require_nonempty and points_arr.shape[0] == 0:
        msg = f"{name} must contain at least one sample."
        raise ValueError(msg)

    if not np.all(np.isfinite(points_arr)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    points_arr = np.array(points_arr, dtype=np.float64, copy=True)
    points_arr.setflags(write=writable)
    return points_arr


def as_sample_batches(
    samples: npt.ArrayLike,
    /,
    *,
    n_distributions: int,
    n_features: int,
    name: str = "Sample batches",
    writable: bool = False,
) -> FloatArray:
    """Return validated sample batches with shape `(n_distributions, n_samples, n_features)`."""
    samples_arr = np.asarray(samples, dtype=np.float64)
    if samples_arr.ndim != 3 or samples_arr.shape[0] != n_distributions:
        msg = (
            f"{name} must have shape ({n_distributions}, n_samples, {n_features}), "
            f"got {samples_arr.shape}."
        )
        raise ValueError(msg)

    if samples_arr.shape[2] != n_features:
        msg = f"{name} must have feature dimension {n_features}, got {samples_arr.shape[2]}."
        raise ValueError(msg)

    if samples_arr.shape[1] == 0:
        msg = f"{name} must contain at least one sample per distribution."
        raise ValueError(msg)

    if not np.all(np.isfinite(samples_arr)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    samples_arr = np.array(samples_arr, dtype=np.float64, copy=True)
    samples_arr.setflags(write=writable)
    return samples_arr


def validate_positive_int(value: object, /, *, name: str) -> int:
    """Validate a positive integer option."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer, got {value}."
        raise ValueError(msg)
    return int(value)


def validate_positive_finite(value: float, /, *, name: str) -> float:
    """Validate a positive finite scalar option."""
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be a positive finite value, got {value}."
        raise ValueError(msg)
    return value


def validate_nonnegative_finite(value: float, /, *, name: str) -> float:
    """Validate a nonnegative finite scalar option."""
    if not isfinite(value) or value < 0.0:
        msg = f"{name} must be a nonnegative finite value, got {value}."
        raise ValueError(msg)
    return value


def validate_unit_interval(value: float, /, *, name: str) -> float:
    """Validate a finite scalar in the closed unit interval."""
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        msg = f"{name} must be a finite value in [0, 1], got {value}."
        raise ValueError(msg)
    return value


def _validate_covariance_values(covariance: FloatArray, /, *, name: str) -> None:
    if not np.all(np.isfinite(covariance)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    try:
        _ = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        msg = f"{name} must be positive definite."
        raise ValueError(msg) from exc


def _validate_symmetric(covariance: FloatArray, /, *, name: str) -> None:
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2)):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
