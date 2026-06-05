from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import scipy

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.typing import Covariance, Covariances, FloatArray, Weights


def logsumexp(a: FloatArray, axis: int = -1) -> FloatArray:
    """Compute log-sum-exp."""
    return np.asarray(scipy.special.logsumexp(a, axis=axis), dtype=np.float64)


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
    diag_shape = (n_features,)

    if covariance_arr.shape == diag_shape:
        covariance_arr = np.diag(covariance_arr).astype(np.float64)
    elif covariance_arr.shape != full_shape:
        msg = (
            f"{name} must have shape {full_shape} or diagonal shape {diag_shape}"
            f", got {covariance_arr.shape}."
        )
        raise ValueError(msg)
    else:
        covariance_arr = cast("Covariance", covariance_arr.copy())

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
    diag_shape = (n_components, n_features)

    if covariances_arr.shape == diag_shape:
        full_covariances = np.zeros(full_shape, dtype=np.float64)
        diagonal = np.arange(n_features)
        full_covariances[:, diagonal, diagonal] = covariances_arr
        covariances_arr = cast("Covariances", full_covariances)
    elif covariances_arr.shape != full_shape:
        msg = (
            f"{name} must have shape {full_shape} or diagonal shape {diag_shape}, "
            f"got {covariances_arr.shape}."
        )
        raise ValueError(msg)
    else:
        covariances_arr = cast("Covariances", covariances_arr.copy())

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


def resolve_samples(
    distribution: Distribution,
    samples: npt.ArrayLike | int,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Return provided samples or draw samples from the distribution."""
    dim = distribution.dim
    if not isinstance(samples, int):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != dim:
            msg = f"Expected samples of shape (n_samples, n_features), got {samples.shape}"
            raise ValueError(msg)
        return samples
    return distribution.sample(n_samples=samples or 10_000, rng=rng)


def pairwise_kl(means: FloatArray, covariances: Covariances) -> FloatArray:
    """Compute pairwise KL divergence between Gaussian components.

    Parameters
    ----------
    means : FloatArray
        An array of shape (N, D) containing N mean vectors of dimension D.
    covariances : Covariances
        An array of shape (N, D, D) containing N covariance matrices of
        dimension D x D.

    Returns
    -------
    FloatArray
        An array of shape (N, N) where the entry at (i, j) is the KL divergence
        from the i-th Gaussian component to the j-th Gaussian component.
    """
    dim = means.shape
    inv_cov = np.linalg.inv(covariances)
    _, logdet = np.linalg.slogdet(covariances)  # (N,)
    logdet_term = logdet[None, :] - logdet[:, None]  # (N, N)
    trace_term = np.einsum("jkl,ilk->ij", inv_cov, covariances)  # (N, N)
    diff = means[None, :, :] - means[:, None, :]  # (N, N, D)
    quad_term = np.einsum("ijd,jde,ije->ij", diff, inv_cov, diff)  # (N, N)
    return 0.5 * (logdet_term - dim + trace_term + quad_term)
