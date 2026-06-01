from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, cast

import numpy as np
import numpy.typing as npt

from gmm_divergence.typing import CovarianceType, PrecisionT


@dataclass(frozen=True, slots=True)
class GaussianMixture(Generic[PrecisionT]):
    weights: npt.NDArray[PrecisionT]
    """Weight array of shape (n_components,)."""
    means: npt.NDArray[PrecisionT]
    """Mean array of shape (n_components, n_features)."""
    covariances: npt.NDArray[PrecisionT]
    """Covariance array shape depends on the covariance type."""
    covariance_type: CovarianceType = "full"

    _chol: npt.NDArray[PrecisionT] | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factors."""

    def __post_init__(self) -> None:
        """Validate the shapes of weights, means, and covariances."""
        if self.weights.ndim != 1:
            msg = "Weights must be a 1D array."
            raise ValueError(msg)

        if self.means.ndim != 2:
            msg = "Means must be a 2D array."
            raise ValueError(msg)

        n_components = self.weights.shape[0]
        n_features = self.means.shape[1]

        if self.weights.shape[0] != self.means.shape[0]:
            msg = "Number of components in weights and means must match."
            raise ValueError(msg)

        if self.covariance_type == "full":
            expected_shape = (n_components, n_features, n_features)
            if self.covariances.shape != expected_shape:
                msg = (
                    "For covariance_type='full', covariances must have shape "
                    f"{expected_shape}, got {self.covariances.shape}."
                )
                raise ValueError(msg)
        elif self.covariance_type == "diag":
            expected_shape = (n_components, n_features)
            if self.covariances.shape != expected_shape:
                msg = (
                    "For covariance_type='diag', covariances must have shape "
                    f"{expected_shape}, got {self.covariances.shape}."
                )
                raise ValueError(msg)
        else:
            msg = f"Unsupported covariance_type: {self.covariance_type!r}."
            raise ValueError(msg)

        self.weights.setflags(write=False)
        self.means.setflags(write=False)
        self.covariances.setflags(write=False)

    @classmethod
    def create(
        cls,
        weights: npt.ArrayLike,
        means: npt.ArrayLike,
        covariances: npt.ArrayLike,
        covariance_type: CovarianceType = "full",
        dtype: type[PrecisionT] = np.float64,
    ) -> GaussianMixture[PrecisionT]:
        """Create a Gaussian mixture from array-like parameters."""
        weights = np.asarray(weights, dtype=dtype)
        means = np.asarray(means, dtype=dtype)
        covariances = np.asarray(covariances, dtype=dtype)
        return cls(
            weights=weights,
            means=means,
            covariances=covariances,
            covariance_type=covariance_type,
        )

    @property
    def dtype(self) -> type[PrecisionT]:
        """Data type of the Gaussian mixture parameters."""
        return self.means.dtype.type

    @property
    def n_components(self) -> int:
        """Number of components in the Gaussian mixture."""
        return self.weights.shape[0]

    def logpdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT]:
        """Evaluate the mixture log-density at one or more sample points."""
        x = np.asarray(x, dtype=self.means.dtype)
        if x.ndim == 1:
            x = x[None, :]
        if self.covariance_type == "full":
            return gmm_logpdf_full(x=x, gmm=self)
        if self.covariance_type == "diag":
            return gmm_logpdf_diag(x=x, gmm=self)
        msg = f"Unsupported covariance type: {self.covariance_type}"
        raise ValueError(msg)

    def pdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT]:
        """Evaluate the mixture density at one or more sample points."""
        return np.exp(self.logpdf(x))

    def chol(self) -> npt.NDArray[PrecisionT]:
        """Compute or retrieve the cached Cholesky factors of the covariance matrices."""
        if self._chol is not None:
            return self._chol

        if self.covariance_type != "full":
            chol = np.sqrt(self.covariances, dtype=self.dtype)
        else:
            chol = np.linalg.cholesky(self.covariances).astype(self.dtype)

        object.__setattr__(self, "_chol", chol)
        return chol

    def sample(
        self, n_samples: int, *, rng: np.random.Generator | int | None = None
    ) -> npt.NDArray[PrecisionT]:
        """Draw samples from the Gaussian mixture."""
        return sample_gmm(self, n_samples=n_samples, rng=rng)


def sample_gmm(
    gmm: GaussianMixture[PrecisionT],
    /,
    n_samples: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[PrecisionT]:
    """Draw samples from a Gaussian mixture."""
    rng = np.random.default_rng(rng)
    weights = gmm.weights / np.sum(gmm.weights)

    component_ids = rng.choice(
        gmm.n_components,
        size=n_samples,
        p=weights,
    )

    if gmm.covariance_type == "diag":
        stds = np.sqrt(gmm.covariances[component_ids])
        eps = rng.standard_normal(size=gmm.means[component_ids].shape)
        samples = gmm.means[component_ids] + eps * stds
    elif gmm.covariance_type == "full":
        chol = gmm.chol()
        eps = rng.standard_normal(size=gmm.means[component_ids].shape)
        samples = gmm.means[component_ids] + np.einsum(
            "nij,nj->ni",
            chol[component_ids],
            eps,
        )
    else:
        msg = f"Unsupported covariance type: {gmm.covariance_type}"
        raise ValueError(msg)

    return samples.astype(gmm.means.dtype, copy=False)


def _logsumexp(a: npt.NDArray[PrecisionT], axis: int = -1) -> npt.NDArray[PrecisionT]:
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(
        a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)),
        axis=axis,
    )


def gmm_logpdf_full(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the log-density of a full-covariance Gaussian mixture."""
    if gmm.covariance_type != "full":
        msg = f"Expected full covariance type, got {gmm.covariance_type}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=gmm.dtype)
    weights = gmm.weights / np.sum(gmm.weights)

    n_samples, n_features = x.shape
    n_components = weights.shape[0]

    log_weights = np.log(weights)
    chol = gmm.chol()
    log_probs = np.empty((n_samples, n_components), dtype=gmm.dtype)
    constant = n_features * np.log(2.0 * np.pi)

    for k in range(n_components):
        diff: npt.NDArray[PrecisionT] = x - gmm.means[k]
        y = cast("npt.NDArray[PrecisionT]", np.linalg.solve(chol[k], diff.T))
        mahal = np.sum(y * y, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[k])))
        log_gaussian = -0.5 * (constant + log_det + mahal)
        log_probs[:, k] = log_weights[k] + log_gaussian

    return _logsumexp(log_probs, axis=1)


def gmm_pdf_full(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the density of a full-covariance Gaussian mixture."""
    return np.exp(gmm_logpdf_full(x, gmm))


def gmm_logpdf_diag(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the log-density of a diagonal-covariance Gaussian mixture."""
    if gmm.covariance_type != "diag":
        msg = f"Expected diagonal covariance type, got {gmm.covariance_type}"
        raise ValueError(msg)

    x = np.asarray(x, dtype=gmm.dtype)
    weights = gmm.weights / np.sum(gmm.weights)
    means = gmm.means
    variances = gmm.covariances

    _n_samples, n_features = x.shape
    log_weights = np.log(weights)

    diff = x[:, None, :] - means[None, :, :]
    mahal = np.sum((diff * diff) / variances[None, :, :], axis=2)
    log_det = np.sum(np.log(variances), axis=1)
    constant = n_features * np.log(2.0 * np.pi)
    log_gaussian = -0.5 * (constant + log_det[None, :] + mahal)
    log_probs = log_weights[None, :] + log_gaussian

    return _logsumexp(log_probs, axis=1)


def gmm_pdf_diag(
    x: npt.ArrayLike,
    gmm: GaussianMixture[PrecisionT],
) -> npt.NDArray[PrecisionT]:
    """Evaluate the density of a diagonal-covariance Gaussian mixture."""
    return np.exp(gmm_logpdf_diag(x, gmm))
