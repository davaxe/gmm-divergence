"""An immutable multivariate Gaussian distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence._core._numeric import logdet_from_cholesky
from gmm_divergence._core._validation import (
    as_covariance,
    as_points,
    validate_positive_finite,
    validate_positive_int,
)
from gmm_divergence.covariance import regularize_covariance

if TYPE_CHECKING:
    from gmm_divergence._core._types import Covariance, Covariances, FloatArray, Weights
    from gmm_divergence.covariance import CovarianceRegularizer


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class Gaussian:
    """Gaussian with mean `(n_features,)` and positive-definite covariance.

    Array-like inputs are copied into read-only float64 arrays. An optional
    regularizer is applied before validating positive definiteness.

    Parameters
    ----------
    mean : array-like, shape (n_features,)
        Nonempty, finite feature means.
    covariance : array-like, shape (n_features, n_features)
        Finite, symmetric covariance. Must be positive definite after any
        regularization.
    regularizer : CovarianceRegularizer or None, default=None
        Optional covariance regularization applied before construction.
    """

    mean: FloatArray
    covariance: Covariance
    _chol: FloatArray | None = field(default=None, repr=False)
    _log_det: float | None = field(default=None, repr=False)

    def __init__(
        self,
        mean: npt.ArrayLike,
        covariance: npt.ArrayLike,
        *,
        regularizer: CovarianceRegularizer | None = None,
    ) -> None:
        mean_array = np.array(mean, dtype=np.float64, copy=True)
        if mean_array.ndim != 1 or mean_array.size == 0:
            msg = "mean must be a nonempty 1D array."
            raise ValueError(msg)
        if not np.isfinite(mean_array).all():
            msg = "mean must contain only finite values."
            raise ValueError(msg)
        if regularizer is not None:
            covariance = regularize_covariance(covariance, regularizer=regularizer)
        covariance_array = as_covariance(covariance, n_features=mean_array.size)
        mean_array.setflags(write=False)
        object.__setattr__(self, "mean", mean_array)
        object.__setattr__(self, "covariance", covariance_array)
        object.__setattr__(self, "_chol", None)
        object.__setattr__(self, "_log_det", None)

    @classmethod
    def from_samples(
        cls, samples: npt.ArrayLike, *, regularizer: CovarianceRegularizer | None = None
    ) -> Gaussian:
        """Estimate a Gaussian from finite observations `(n_samples, n_features)`.

        At least two observations are required. Covariance uses the unbiased
        estimate (divisor `n_samples - 1`). Singular covariance requires an
        explicit regularizer.

        Parameters
        ----------
        samples : array-like, shape (n_samples, n_features)
            Finite observations with at least two rows and one feature. For
            one feature, provide a column matrix rather than a 1D vector.
        regularizer : CovarianceRegularizer or None, default=None
            Optional regularization for the empirical covariance.

        Returns
        -------
        Gaussian
            Distribution with the sample mean and unbiased sample covariance.
        """
        observations = np.asarray(samples, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] == 0:
            msg = "samples must have shape (n_samples >= 2, n_features >= 1)."
            raise ValueError(msg)
        if not np.isfinite(observations).all():
            msg = "samples must contain only finite values."
            raise ValueError(msg)
        covariance = np.atleast_2d(np.cov(observations, rowvar=False, bias=False))
        return cls(np.mean(observations, axis=0), covariance, regularizer=regularizer)

    @classmethod
    def univariate(cls, mean: float = 0.0, variance: float = 1.0) -> Gaussian:
        """Create a one-feature Gaussian from mean and variance.

        Parameters
        ----------
        mean : float, default=0.0
            Finite feature mean.
        variance : float, default=1.0
            Finite, strictly positive feature variance.

        Returns
        -------
        Gaussian
            Gaussian with one feature.
        """
        _ = validate_positive_finite(variance, name="variance")
        return cls([mean], [[variance]])

    @classmethod
    def isotropic(cls, mean: npt.ArrayLike, *, variance: float = 1.0) -> Gaussian:
        """Create a Gaussian with the same variance along every feature.

        Parameters
        ----------
        mean : array-like, shape (n_features,)
            Nonempty, finite feature means.
        variance : float, default=1.0
            Finite, strictly positive variance; covariance is `variance * I`.

        Returns
        -------
        Gaussian
            Gaussian with independent features and equal marginal variances.
        """
        _ = validate_positive_finite(variance, name="variance")
        mean_array = np.asarray(mean, dtype=np.float64)
        if mean_array.ndim != 1 or mean_array.size == 0:
            msg = "mean must be a nonempty 1D array."
            raise ValueError(msg)
        return cls(mean_array, np.eye(mean_array.size) * variance)

    @classmethod
    def standard(cls, n_features: int) -> Gaussian:
        """Create a zero-mean Gaussian with identity covariance.

        Parameters
        ----------
        n_features : int
            Positive number of features.

        Returns
        -------
        Gaussian
            Standard multivariate Gaussian of the requested dimension.
        """
        n_features = validate_positive_int(n_features, name="n_features")
        return cls(np.zeros(n_features), np.eye(n_features))

    @property
    def n_features(self) -> int:
        """Number of features in one observation."""
        return self.mean.size

    def cholesky(self) -> FloatArray:
        """Return the cached, read-only lower covariance factor, shape (n_features, n_features)."""
        if self._chol is None:
            chol = np.linalg.cholesky(self.covariance).astype(np.float64)
            chol.setflags(write=False)
            object.__setattr__(self, "_chol", chol)
            return chol
        return self._chol

    def log_determinant(self) -> float:
        """Return the cached natural logarithm of the covariance determinant."""
        if self._log_det is None:
            value = float(logdet_from_cholesky(self.cholesky()))
            object.__setattr__(self, "_log_det", value)
            return value
        return self._log_det

    def sample(self, n_samples: int, *, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw independent observations from this Gaussian.

        Parameters
        ----------
        n_samples : int
            Positive number of observations to draw.
        rng : numpy.random.Generator, int, or None, default=None
            Random source. An integer restarts a reproducible stream on each
            call; a generator advances its state. None uses fresh entropy.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Newly allocated float64 observations.
        """
        n_samples = validate_positive_int(n_samples, name="n_samples")
        return np.random.default_rng(rng).multivariate_normal(
            mean=self.mean, cov=self.covariance, size=n_samples
        )

    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate natural-log densities at one or more points.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_points, n_features)
            Finite observations. A 1D input represents one observation.

        Returns
        -------
        ndarray, shape (n_points,)
            Log densities, including a length-one array for a single point.
        """
        points = as_points(x, n_features=self.n_features, name="x")
        whitened = np.linalg.solve(self.cholesky(), (points - self.mean).T).T
        return -0.5 * (
            self.n_features * np.log(2 * np.pi)
            + self.log_determinant()
            + np.sum(whitened**2, axis=-1)
        )

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate densities at one or more points.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_points, n_features)
            Finite observations. A 1D input represents one observation.

        Returns
        -------
        ndarray, shape (n_points,)
            Densities, including a length-one array for a single point.
        """
        return np.exp(self.logpdf(x))

    def component_arrays(self) -> tuple[Weights, FloatArray, Covariances]:
        """Return weight, mean, and covariance arrays for one Gaussian component.

        Shapes are (1,), (1, n_features), and (1, n_features, n_features).

        The weight array is newly allocated; the other arrays are read-only
        views of this Gaussian's parameters.
        """
        return np.array([1.0]), self.mean[None, :], self.covariance[None, :, :]

    @override
    def __repr__(self) -> str:
        return f"Gaussian(n_features={self.n_features})"
