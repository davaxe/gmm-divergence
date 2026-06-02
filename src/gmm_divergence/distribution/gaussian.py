from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from gmm_divergence.distribution.base import Distribution

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence.typing import FloatArray


@dataclass(frozen=True, slots=True, repr=False)
class Gaussian(Distribution):
    mean: FloatArray
    """Mean array of shape (n_features,)."""

    covariance: FloatArray
    """Covariance array of shape (n_features, n_features)."""

    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factor."""

    @classmethod
    def from_arrays(
        cls,
        mean: npt.ArrayLike,
        covariance: npt.ArrayLike,
    ) -> Gaussian:
        """Create a Gaussian instance from array-like inputs."""
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        return cls(mean=mean, covariance=covariance)

    def __post_init__(self) -> None:
        """Validate the shapes of mean and covariance."""
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float64))
        object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=np.float64))

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
                np.diag(self.covariance).astype(np.float64),
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

    def chol(self) -> FloatArray:
        """Compute or retrieve the Cholesky factor of the covariance."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariance).astype(np.float64)
        object.__setattr__(self, "_chol", chol)
        return chol

    @override
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> FloatArray:
        """Draw samples from the Gaussian."""
        rng = np.random.default_rng(rng)
        return rng.multivariate_normal(
            mean=self.mean,
            cov=self.covariance,
            size=n_samples,
        ).astype(np.float64)

    @override
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian at given points."""
        x = np.asarray(x, dtype=np.float64)
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
            f"mean_shape={self.mean.shape}, "
            f"covariance_shape={self.covariance.shape}"
            f")"
        )
