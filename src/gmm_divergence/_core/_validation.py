from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

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

    weights_arr = cast("Weights", weights_arr / weight_sum if normalize else weights_arr.copy())
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
    covariance_arr = np.asarray(covariance, dtype=np.float64)
    full_shape = (n_features, n_features)
    if covariance_arr.shape != full_shape:
        msg = f"{name} must have shape {full_shape}, got {covariance_arr.shape}."
        raise ValueError(msg)
    covariance_arr = 0.5 * (covariance_arr + covariance_arr.T)
    _validate_covariance_values(covariance_arr, name=name)
    covariance_arr.setflags(write=writable)
    return covariance_arr


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
    covariances_arr = np.asarray(covariances, dtype=np.float64)
    full_shape = (n_components, n_features, n_features)
    if covariances_arr.shape != full_shape:
        msg = f"{name} must have shape {full_shape}, got {covariances_arr.shape}."
        raise ValueError(msg)
    covariances_arr = 0.5 * (covariances_arr + np.swapaxes(covariances_arr, -1, -2))
    _validate_covariance_values(covariances_arr, name=name)
    covariances_arr.setflags(write=writable)
    return covariances_arr


def _validate_covariance_values(covariance: FloatArray, /, *, name: str) -> None:
    if not np.all(np.isfinite(covariance)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2)):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)

    try:
        _ = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        msg = f"{name} must be positive definite."
        raise ValueError(msg) from exc
