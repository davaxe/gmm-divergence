from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

import numpy as np

from gmm_divergence.typing import CovarianceType, PrecisionT

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Gaussian(Generic[PrecisionT]):
    mean: npt.NDArray[PrecisionT]
    """Mean array of shape (n_features,)."""
    covariance: npt.NDArray[PrecisionT]
    """Covariance array shape depends on the covariance type."""
    covariance_type: CovarianceType = "full"

    _chol: npt.NDArray[PrecisionT] | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factor."""

    def __post_init__(self) -> None:
        """Validate the shapes of mean and covariance."""
        if self.mean.ndim != 1:
            msg = "Mean must be a 1D array."
            raise ValueError(msg)

        n_features = self.mean.shape[0]

        if self.covariance_type == "full":
            expected_shape = (n_features, n_features)
            if self.covariance.shape != expected_shape:
                msg = (
                    "For covariance_type='full', covariance must have shape "
                    f"{expected_shape}, got {self.covariance.shape}."
                )
                raise ValueError(msg)
        elif self.covariance_type == "diag":
            expected_shape = (n_features,)
            if self.covariance.shape != expected_shape:
                msg = (
                    "For covariance_type='diag', covariance must have shape "
                    f"{expected_shape}, got {self.covariance.shape}."
                )
                raise ValueError(msg)
        else:
            msg = f"Unsupported covariance_type: {self.covariance_type!r}."
            raise ValueError(msg)

    @property
    def dtype(self) -> type[PrecisionT]:
        """Data type of the Gaussian parameters."""
        return self.mean.dtype.type

    def chol(self) -> npt.NDArray[PrecisionT]:
        """Compute or retrieve the Cholesky factor of the covariance."""
        if self._chol is not None:
            return self._chol

        if self.covariance_type == "full":
            chol = np.linalg.cholesky(self.covariance).astype(self.dtype)
        else:
            chol = np.sqrt(self.covariance, dtype=self.dtype)

        object.__setattr__(self, "_chol", chol)
        return chol

    def sample(
        self, n_samples: int, rng: np.random.Generator | int | None = None
    ) -> npt.NDArray[PrecisionT]:
        """Generate samples from the Gaussian distribution."""
        rng = np.random.default_rng(rng)
        return rng.multivariate_normal(
            mean=self.mean,
            cov=self.covariance if self.covariance_type == "full" else np.diag(self.covariance),
            size=n_samples,
        ).astype(self.dtype)

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

    def pdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT]:
        """Evaluate the density of the Gaussian at given points."""
        return np.exp(self.logpdf(x))
