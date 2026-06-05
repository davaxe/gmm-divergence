from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from typing_extensions import override

from gmm_divergence._core._validation import as_covariance
from gmm_divergence.distributions._base import GaussianComponentArrays, GaussianFamily

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import Covariance, FloatArray


@dataclass(frozen=True, slots=True, repr=False)
class Gaussian(GaussianFamily):
    mean: FloatArray
    """Mean array of shape (n_features,)."""

    covariance: Covariance
    """Covariance array of shape (n_features, n_features)."""

    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factor."""

    @classmethod
    def from_arrays(cls, mean: npt.ArrayLike, covariance: npt.ArrayLike) -> Gaussian:
        """Create a Gaussian instance from array-like inputs."""
        return cls(mean=cast("FloatArray", mean), covariance=cast("Covariance", covariance))

    def __post_init__(self) -> None:
        """Validate the shapes of mean and covariance."""
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float64))

        if self.mean.ndim != 1 and not (self.mean.ndim == 2 and self.mean.shape[1] == 1):
            msg = "Mean must be a 1D array."
            raise ValueError(msg)

        object.__setattr__(self, "mean", self.mean.flatten())
        if not np.all(np.isfinite(self.mean)):
            msg = "Mean must contain only finite values."
            raise ValueError(msg)
        n_features = self.mean.shape[0]
        if n_features == 0:
            msg = "Mean must contain at least one feature."
            raise ValueError(msg)
        object.__setattr__(
            self, "covariance", as_covariance(self.covariance, n_features=n_features)
        )

        self.mean.setflags(write=False)

    def chol(self) -> FloatArray:
        """Compute or retrieve the Cholesky factor of the covariance."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariance).astype(np.float64)
        object.__setattr__(self, "_chol", chol)
        return chol

    @override
    def sample(self, n_samples: int, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw samples from the Gaussian."""
        rng = np.random.default_rng(rng)
        return rng.multivariate_normal(mean=self.mean, cov=self.covariance, size=n_samples).astype(
            np.float64
        )

    @override
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian at given points."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        d = self.mean.shape[0]
        chol = self.chol()
        diff = x - self.mean
        rhs = cast("FloatArray", diff.T)
        y = cast("FloatArray", np.linalg.solve(chol, rhs).T)
        mahalanobis = np.sum(y**2, axis=-1)
        log_det = 2 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

    @override
    def component_arrays(self) -> GaussianComponentArrays:
        """Return the mean and covariance as component arrays."""
        return np.array([1.0]), self.mean[None, :], self.covariance[None, :, :]

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dim={self.dim}, "
            f"mean_shape={self.mean.shape}, "
            f"covariance_shape={self.covariance.shape}"
            f")"
        )
