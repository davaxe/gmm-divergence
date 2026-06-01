from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from gmm_divergence.gmm.density import gmm_logpdf_diag, gmm_logpdf_full

PrecisionT = TypeVar("PrecisionT", np.float32, np.float64, default=np.float64)

CovarianceType = Literal["full", "diag"]


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
    """Cached Cholesky factors, shape (n_components, n_features, n_features)."""

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
        if self.covariance_type != "full":
            msg = "Cholesky factors are only defined for full covariance type."
            raise ValueError(msg)

        if self._chol is None:
            object.__setattr__(self, "_chol", np.linalg.cholesky(self.covariances))
        return cast("npt.NDArray[PrecisionT]", self._chol)
