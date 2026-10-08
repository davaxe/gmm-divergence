from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import numpy.typing as npt
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from typing_extensions import override

from gmm_divergence._core._numeric import logdet_from_cholesky, logsumexp
from gmm_divergence._core._validation import (
    as_covariances,
    as_points,
    as_weights,
    validate_positive_int,
)
from gmm_divergence.covariance import regularize_covariance
from gmm_divergence.distributions._gaussian import Gaussian

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from gmm_divergence._core._types import Covariance, Covariances, FloatArray, Weights
    from gmm_divergence.covariance import CovarianceRegularizer


@dataclass(frozen=True, slots=True)
class MixtureDiagnostics:
    """Diagnostics for a Gaussian mixture model."""

    n_components: int
    """Number of components in the mixture."""
    dim: int
    """Dimensionality of the mixture."""
    weight_sum: float
    """Sum of the weights of the components."""
    max_weight: float
    """Maximum weight among the components."""
    min_weight: float
    """Minimum weight among the components."""
    weights_entropy: float
    """Entropy of the weights of the components in nats."""
    covar_condition_numbers: list[float]
    """Condition numbers of the covariance matrices of the components."""


@dataclass(frozen=True, slots=True, repr=False)
class GaussianMixture:
    weights: Weights
    """Weight array of shape (n_components,)."""
    means: FloatArray
    """Mean array of shape (n_components, n_features)."""
    covariances: Covariances
    """Covariance array of shape (n_components, n_features, n_features)."""
    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    _log_dets: FloatArray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_arrays(
        cls,
        weights: npt.ArrayLike,
        means: npt.ArrayLike,
        covariances: npt.ArrayLike,
        regularizer: CovarianceRegularizer | None = None,
    ) -> GaussianMixture:
        """Create a Gaussian mixture from array-like parameters."""
        return cls(
            weights=np.asarray(weights),
            means=np.asarray(means),
            covariances=np.asarray(covariances)
            if regularizer is None
            else regularize_covariance(covariances, regularizer=regularizer, batched=True),
        )

    @classmethod
    def from_sklearn_gmm(cls, gmm: SklearnGaussianMixture) -> GaussianMixture:
        """Convert a fitted sklearn mixture, expanding covariances to full matrices."""
        means = np.array(gmm.means_, dtype=np.float64)
        weights = as_weights(gmm.weights_)
        sklearn_covariances = np.array(gmm.covariances_, dtype=np.float64)
        n_components, n_features = means.shape
        covariance_type = cast("str", gmm.get_params()["covariance_type"])
        match covariance_type:
            case "full":
                covariances = sklearn_covariances
            case "tied":
                covariances = np.repeat(sklearn_covariances[None, :, :], n_components, axis=0)
            case "diag":
                covariances = sklearn_covariances[:, :, None] * np.eye(n_features)
            case "spherical":
                covariances = sklearn_covariances[:, None, None] * np.eye(n_features)
            case _:
                msg = f"Unsupported sklearn covariance type: {covariance_type!r}."
                raise ValueError(msg)
        return cls.from_arrays(weights=weights, means=means, covariances=covariances)

    def to_sklearn_gmm(self) -> SklearnGaussianMixture:
        """Convert to a sklearn mixture ready for evaluation and sampling.

        Parameters are copied without fitting data. Optimization history and
        convergence diagnostics are unavailable because no fitting is performed.
        """
        mixture = SklearnGaussianMixture(n_components=self.n_components, covariance_type="full")
        mixture.weights_ = self.weights.copy()
        mixture.means_ = self.means.copy()
        mixture.covariances_ = self.covariances.copy()
        # Sklearn uses upper triangular factors U such that precision = U @ U.T.
        inverse_chol = np.linalg.solve(
            self.chol(), np.broadcast_to(np.eye(self.dim), self.covariances.shape)
        )
        precision_chol = np.swapaxes(inverse_chol, -1, -2)
        mixture.precisions_cholesky_ = precision_chol
        mixture.precisions_ = precision_chol @ np.swapaxes(precision_chol, -1, -2)
        mixture.n_features_in_ = self.dim
        return mixture

    @classmethod
    def from_samples(cls, x: npt.ArrayLike, n_components: int) -> GaussianMixture:
        """Fit a Gaussian mixture to sample data using sklearn's EM algorithm.

        This uses default configuration to fit the GMM. For full control over
        the fitting process, use sklearn's GaussianMixture directly and then
        convert to this class using `from_sklearn_gmm`.
        """
        samples = np.asarray(x, dtype=np.float64)
        sklearn_gmm = SklearnGaussianMixture(n_components=n_components).fit(samples)
        return cls.from_sklearn_gmm(sklearn_gmm)

    @classmethod
    def from_components(
        cls, components: Sequence[Gaussian], weights: npt.ArrayLike | None = None
    ) -> GaussianMixture:
        """Create a Gaussian mixture from a sequence of Gaussian components and optional weights."""
        means = np.array([comp.mean for comp in components], dtype=np.float64)
        covariances = np.array([comp.covariance for comp in components], dtype=np.float64)
        if weights is None:
            weights = np.ones(len(components), dtype=np.float64) / len(components)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        return cls(weights=weights, means=means, covariances=covariances)

    def __post_init__(self) -> None:
        """Validate the shapes of weights, means, and covariances."""
        object.__setattr__(self, "means", np.array(self.means, dtype=np.float64, copy=True))

        if self.means.ndim != 2:
            msg = "Means must be a 2D array."
            raise ValueError(msg)

        if not np.all(np.isfinite(self.means)):
            msg = "Means must contain only finite values."
            raise ValueError(msg)

        if self.means.shape[1] == 0:
            msg = "Means must contain at least one feature."
            raise ValueError(msg)

        n_features = self.means.shape[1]
        object.__setattr__(
            self, "weights", as_weights(self.weights, expected_length=self.means.shape[0])
        )
        n_components = self.weights.shape[0]
        object.__setattr__(
            self,
            "covariances",
            as_covariances(self.covariances, n_components=n_components, n_features=n_features),
        )

        self.means.setflags(write=False)

    @property
    def n_components(self) -> int:
        """Number of components in the Gaussian mixture."""
        return self.weights.shape[0]

    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian mixture at given points."""
        return gmm_logpdf(x=x, gmm=self)

    def chol(self) -> FloatArray:
        """Compute or retrieve the cached Cholesky factors."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariances).astype(np.float64)
        chol.setflags(write=False)
        object.__setattr__(self, "_chol", chol)
        return chol

    def log_dets(self) -> FloatArray:
        """Compute or retrieve the cached log-determinants of the covariances."""
        if self._log_dets is not None:
            return self._log_dets

        chol = self.chol()
        log_dets = logdet_from_cholesky(chol)
        log_dets.setflags(write=False)
        object.__setattr__(self, "_log_dets", log_dets)
        return log_dets

    def sample(self, n_samples: int, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw samples from the Gaussian mixture."""
        return sample_gmm(self, n_samples=n_samples, rng=rng)

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the density of the Gaussian mixture at given points."""
        return np.exp(self.logpdf(x))

    def get_component(self, index: int) -> Gaussian:
        """Return the Gaussian component at the specified index."""
        if index < 0 or index >= self.n_components:
            msg = f"Component index {index} is out of bounds for {self.n_components} components."
            raise IndexError(msg)

        return Gaussian(mean=self.means[index], covariance=self.covariances[index])

    def select_components(self, indices: npt.ArrayLike) -> GaussianMixture:
        """Return a new Gaussian mixture containing only the specified components."""
        indices = np.asarray(indices, dtype=np.intp)
        if np.any(indices < 0) or np.any(indices >= self.n_components):
            msg = f"Component indices must be in the range [0, {self.n_components})."
            raise IndexError(msg)

        return GaussianMixture(
            weights=as_weights(self.weights[indices], expected_length=indices.shape[0]),
            means=self.means[indices],
            covariances=self.covariances[indices],
        )

    def responsibilities(self, x: npt.ArrayLike) -> FloatArray:
        """Compute the responsibilities of each component for the given points.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Points at which to compute responsibilities.

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            The responsibilities of each component for each point, normalized to
            sum to 1 across components.
        """
        x = as_points(x, n_features=self.dim, name="x")
        if self.n_components == 1:
            return np.ones((x.shape[0], 1), dtype=np.float64)

        log_terms = _component_logpdf(self, x) + np.log(self.weights)[None, :]
        return np.exp(log_terms - logsumexp(log_terms, axis=1)[:, None])

    @override
    def __repr__(self) -> str:
        weight_sum = float(np.sum(self.weights))
        return (
            f"{type(self).__name__}("
            f"n_components={self.n_components}, "
            f"dim={self.dim}, "
            f"weight_sum={weight_sum:.6g}, "
            f"means_shape={self.means.shape}, "
            f"covariances_shape={self.covariances.shape}"
            f")"
        )

    @overload
    def as_gaussian(self, *, require_single: Literal[True]) -> Gaussian | None: ...

    @overload
    def as_gaussian(self, *, require_single: Literal[False] = False) -> Gaussian: ...

    def as_gaussian(self, *, require_single: bool = False) -> Gaussian | None:
        """Return a Gaussian approximation of the mixture using moment matching."""
        if require_single and self.n_components > 1:
            return None
        if self.n_components == 1:
            return self.get_component(0)
        mean, covariance = self.moments()
        return Gaussian(mean=mean.astype(np.float64, copy=False), covariance=covariance)

    def component_arrays(self) -> tuple[Weights, FloatArray, Covariances]:
        """Return the weights, means, and covariances as arrays."""
        return self.weights, self.means, self.covariances

    @property
    def dim(self) -> int:
        """Dimensionality of the Gaussian mixture."""
        return self.means.shape[1]

    def __iter__(self) -> Iterator[tuple[float, Gaussian]]:
        """Iterate over the Gaussian components of the mixture."""
        for k in range(self.n_components):
            yield self.weights[k], self.get_component(k)

    def diagnostics(self) -> MixtureDiagnostics:
        """Return diagnostics for the Gaussian mixture."""
        weight_sum = float(np.sum(self.weights))
        max_weight = float(np.max(self.weights))
        min_weight = float(np.min(self.weights))
        weights_entropy = float(-np.sum(self.weights * np.log(self.weights + 1e-12)))
        covar_condition_numbers = [float(np.linalg.cond(cov)) for cov in self.covariances]
        return MixtureDiagnostics(
            n_components=self.n_components,
            dim=self.dim,
            weight_sum=weight_sum,
            max_weight=max_weight,
            min_weight=min_weight,
            weights_entropy=weights_entropy,
            covar_condition_numbers=covar_condition_numbers,
        )

    def moments(self) -> tuple[FloatArray, Covariance]:
        """Return the mean and covariance of the Gaussian mixture."""
        weights, means, covariances = self.component_arrays()
        mean = np.sum(weights[:, None] * means, axis=0)
        mean_delta = means - mean
        covariance = np.sum(
            weights[:, None, None]
            * (covariances + mean_delta[:, :, None] * mean_delta[:, None, :]),
            axis=0,
        )
        covariance = 0.5 * (covariance + covariance.T)
        return mean.astype(np.float64, copy=False), covariance


def sample_gmm(
    gmm: GaussianMixture, /, n_samples: int, *, rng: np.random.Generator | int | None = None
) -> FloatArray:
    """Draw samples from a Gaussian mixture."""
    n_samples = validate_positive_int(n_samples, name="n_samples")
    rng = np.random.default_rng(rng)

    component_ids = rng.choice(gmm.n_components, size=n_samples, p=gmm.weights)
    chol = gmm.chol()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)
    samples = gmm.means[component_ids] + np.einsum("nij,nj->ni", chol[component_ids], eps)
    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the log-density of a Gaussian mixture without an explicit Python loop."""
    x = as_points(x, n_features=gmm.dim, name="x")

    log_terms = _component_logpdf(gmm, x) + np.log(gmm.weights)[None, :]
    return logsumexp(log_terms, axis=1)


def _component_logpdf(gmm: GaussianMixture, x: FloatArray) -> FloatArray:
    """Evaluate component densities at validated points using cached factors."""
    diff = x[None, :, :] - gmm.means[:, None, :]
    rhs = np.swapaxes(diff, 1, 2)
    whitened = np.linalg.solve(gmm.chol(), rhs)
    mahalanobis = np.sum(whitened**2, axis=1)
    constant = gmm.dim * np.log(2.0 * np.pi)
    return (-0.5 * (constant + gmm.log_dets()[:, None] + mahalanobis)).T


def gmm_pdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the density of a Gaussian mixture."""
    return np.exp(gmm_logpdf(x, gmm))
