from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from gmm_divergence.distribution.base import Distribution
from gmm_divergence.typing import PrecisionT

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True, repr=False)
class Gaussian(Distribution[PrecisionT]):
    mean: npt.NDArray[PrecisionT]
    """Mean array of shape (n_features,)."""

    covariance: npt.NDArray[PrecisionT]
    """Covariance array of shape (n_features, n_features)."""

    _chol: npt.NDArray[PrecisionT] | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factor."""

    @classmethod
    def from_arrays(
        cls,
        mean: npt.ArrayLike,
        covariance: npt.ArrayLike,
        dtype: type[PrecisionT] = np.float64,
    ) -> Gaussian[PrecisionT]:
        """Create a Gaussian instance from array-like inputs."""
        mean = np.asarray(mean, dtype=dtype)
        covariance = np.asarray(covariance, dtype=dtype)
        return cls(mean=mean, covariance=covariance)

    def __post_init__(self) -> None:
        """Validate the shapes of mean and covariance."""
        if self.mean.ndim != 1 and not (self.mean.ndim == 2 and self.mean.shape[1] == 1):
            msg = "Mean must be a 1D array."
            raise ValueError(msg)

        object.__setattr__(self, "mean", self.mean.flatten())
        n_features = self.mean.shape[0]

        full_shape = (n_features, n_features)
        diag_shape = (n_features,)

        if self.covariance.shape == diag_shape:
            object.__setattr__(
                self,
                "covariance",
                np.diag(self.covariance).astype(self.dtype),
            )
        elif self.covariance.shape != full_shape:
            msg = (
                "Covariance must have shape "
                f"{full_shape} or diagonal shape {diag_shape}, "
                f"got {self.covariance.shape}."
            )
            raise ValueError(msg)

        self.mean.setflags(write=False)
        self.covariance.setflags(write=False)

    @property
    @override
    def dim(self) -> int:
        """Return the dimensionality of the Gaussian."""
        return self.mean.shape[0]

    @property
    @override
    def dtype(self) -> type[PrecisionT]:
        """Data type of the Gaussian parameters."""
        return self.mean.dtype.type

    def chol(self) -> npt.NDArray[PrecisionT]:
        """Compute or retrieve the Cholesky factor of the covariance."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariance).astype(self.dtype)
        object.__setattr__(self, "_chol", chol)
        return chol

    @override
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> npt.NDArray[PrecisionT]:
        """Draw samples from the Gaussian."""
        rng = np.random.default_rng(rng)
        return rng.multivariate_normal(
            mean=self.mean,
            cov=self.covariance,
            size=n_samples,
        ).astype(self.dtype)

    @override
    def logpdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT]:
        """Evaluate the log-density of the Gaussian at given points."""
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim == 1:
            x = x[None, :]

        d = self.mean.shape[0]
        chol = self.chol()
        diff = x - self.mean
        y = np.linalg.solve(chol, diff.T).T
        mahalanobis = np.sum(y**2, axis=-1)
        log_det = 2 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dim={self.dim}, "
            f"dtype={self.dtype.__name__}, "
            f"mean_shape={self.mean.shape}, "
            f"covariance_shape={self.covariance.shape}"
            f")"
        )
