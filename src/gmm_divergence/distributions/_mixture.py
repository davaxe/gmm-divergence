from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from typing_extensions import override

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._numeric import logdet_from_cholesky, logsumexp
from gmm_divergence._core._validation import (
    as_covariances,
    as_points,
    as_weights,
    validate_positive_int,
)
from gmm_divergence.covariance import regularize_covariance
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mapping import MixtureMapping, index_array, validate_index

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from gmm_divergence._core._types import Covariance, Covariances, FloatArray, Weights
    from gmm_divergence.covariance import CovarianceRegularizer


@dataclass(frozen=True, slots=True)
class MixtureDiagnostics:
    """Diagnostics for a Gaussian mixture model."""

    n_components: int
    """Number of components in the mixture."""
    n_features: int
    """Dimensionality of the mixture."""
    weight_sum: float
    """Sum of the weights of the components."""
    max_weight: float
    """Maximum weight among the components."""
    min_weight: float
    """Minimum weight among the components."""
    weight_entropy: float
    """Entropy of the weights of the components in nats."""
    covariance_condition_numbers: FloatArray
    """Condition numbers of the covariance matrices of the components."""

    def __post_init__(self) -> None:
        """Own the diagnostic array independently of its caller."""
        object.__setattr__(
            self,
            "covariance_condition_numbers",
            readonly_copy(self.covariance_condition_numbers, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, repr=False, init=False, eq=False)
class GaussianMixture:
    weights: Weights
    """Weight array of shape (n_components,)."""
    means: FloatArray
    """Mean array of shape (n_components, n_features)."""
    covariances: Covariances
    """Covariance array of shape (n_components, n_features, n_features)."""
    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    _log_dets: FloatArray | None = field(default=None, init=False, repr=False)
    _aggregate_moments: tuple[FloatArray, Covariance] | None = field(
        default=None, init=False, repr=False
    )

    mapping: MixtureMapping | None = None
    """Optional immediate-source provenance, one row per flattened component."""

    def __init__(
        self,
        weights: npt.ArrayLike,
        means: npt.ArrayLike,
        covariances: npt.ArrayLike,
        *,
        regularizer: CovarianceRegularizer | None = None,
        mapping: MixtureMapping | None = None,
    ) -> None:
        means_array = np.array(means, dtype=np.float64, copy=True)
        if means_array.ndim != 2 or 0 in means_array.shape or not np.all(np.isfinite(means_array)):
            msg = "means must be a nonempty finite array of shape (n_components, n_features)."
            raise ValueError(msg)
        weights_array = as_weights(weights, expected_length=means_array.shape[0])
        if regularizer is not None:
            covariances = regularize_covariance(covariances, regularizer=regularizer)
        covariance_array = as_covariances(
            covariances, n_components=means_array.shape[0], n_features=means_array.shape[1]
        )
        if mapping is not None and len(mapping.source_index) != len(weights_array):
            msg = "mapping must contain one row per component."
            raise ValueError(msg)
        means_array.setflags(write=False)
        object.__setattr__(self, "weights", weights_array)
        object.__setattr__(self, "means", means_array)
        object.__setattr__(self, "covariances", covariance_array)
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "_chol", None)
        object.__setattr__(self, "_log_dets", None)
        object.__setattr__(self, "_aggregate_moments", None)

    @classmethod
    def from_sklearn(cls, gmm: SklearnGaussianMixture) -> GaussianMixture:
        """Convert a fitted sklearn mixture, expanding covariances to full matrices."""
        if not all(hasattr(gmm, name) for name in ("means_", "weights_", "covariances_")):
            msg = "gmm must be fitted before conversion with from_sklearn."
            raise ValueError(msg)
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
        return cls(weights=weights, means=means, covariances=covariances)

    def to_sklearn(self) -> SklearnGaussianMixture:
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
            self.cholesky(), np.broadcast_to(np.eye(self.n_features), self.covariances.shape)
        )
        precision_chol = np.swapaxes(inverse_chol, -1, -2)
        mixture.precisions_cholesky_ = precision_chol
        mixture.precisions_ = precision_chol @ np.swapaxes(precision_chol, -1, -2)
        mixture.n_features_in_ = self.n_features
        return mixture

    @classmethod
    def from_components(
        cls, components: Sequence[Gaussian], *, weights: npt.ArrayLike | None = None
    ) -> GaussianMixture:
        """Create a Gaussian mixture from a sequence of Gaussian components and optional weights."""
        if not components:
            msg = "components must contain at least one Gaussian."
            raise ValueError(msg)
        means = np.array([comp.mean for comp in components], dtype=np.float64)
        covariances = np.array([comp.covariance for comp in components], dtype=np.float64)
        if weights is None:
            weights = np.ones(len(components), dtype=np.float64) / len(components)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        return cls(weights=weights, means=means, covariances=covariances)

    @property
    def n_components(self) -> int:
        """Number of components in the Gaussian mixture."""
        return self.weights.shape[0]

    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian mixture at given points."""
        return gmm_logpdf(x=x, gmm=self)

    def cholesky(self) -> FloatArray:
        """Compute or retrieve the cached Cholesky factors."""
        if self._chol is not None:
            return self._chol

        cholesky = np.linalg.cholesky(self.covariances).astype(np.float64)
        cholesky.setflags(write=False)
        object.__setattr__(self, "_chol", cholesky)
        return cholesky

    def log_determinants(self) -> FloatArray:
        """Compute or retrieve the cached log-determinants of the covariances."""
        if self._log_dets is not None:
            return self._log_dets

        cholesky = self.cholesky()
        log_determinants = logdet_from_cholesky(cholesky)
        log_determinants.setflags(write=False)
        object.__setattr__(self, "_log_dets", log_determinants)
        return log_determinants

    def sample(self, n_samples: int, *, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw samples from the Gaussian mixture."""
        return sample_gmm(self, n_samples=n_samples, rng=rng)

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the density of the Gaussian mixture at given points."""
        return np.exp(self.logpdf(x))

    def get_component(self, index: int) -> Gaussian:
        """Return the Gaussian component at the specified index."""
        validate_index(index, self.n_components)

        return Gaussian(mean=self.means[index], covariance=self.covariances[index])

    def select_components(self, indices: npt.ArrayLike) -> GaussianMixture:
        """Return a new Gaussian mixture containing only the specified components."""
        indices = index_array(indices, name="indices")
        if np.any(indices < 0) or np.any(indices >= self.n_components):
            msg = f"Component indices must be in the range [0, {self.n_components})."
            raise IndexError(msg)

        return GaussianMixture(
            weights=as_weights(self.weights[indices], expected_length=indices.shape[0]),
            means=self.means[indices],
            covariances=self.covariances[indices],
            mapping=None
            if self.mapping is None
            else MixtureMapping(
                source_index=self.mapping.source_index[indices],
                local_component_index=self.mapping.local_component_index[indices],
            ),
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
        x = as_points(x, n_features=self.n_features, name="x")
        if self.n_components == 1:
            return np.ones((x.shape[0], 1), dtype=np.float64)

        with np.errstate(divide="ignore"):
            log_terms = _component_logpdf(self, x) + np.log(self.weights)[None, :]
        return np.exp(log_terms - logsumexp(log_terms, axis=1)[:, None])

    @override
    def __repr__(self) -> str:
        weight_sum = float(np.sum(self.weights))
        return (
            f"{type(self).__name__}("
            f"n_components={self.n_components}, "
            f"n_features={self.n_features}, "
            f"weight_sum={weight_sum:.6g}, "
            f"means_shape={self.means.shape}, "
            f"covariances_shape={self.covariances.shape}"
            f")"
        )

    def as_gaussian(self) -> Gaussian:
        """Convert an exactly single-component mixture to a Gaussian."""
        if self.n_components != 1:
            msg = (
                "as_gaussian requires exactly one component; use moment_match for an approximation."
            )
            raise ValueError(msg)
        return self.get_component(0)

    def moment_match(self) -> Gaussian:
        """Approximate the mixture with a Gaussian sharing its aggregate moments."""
        return Gaussian(self.mean, self.covariance)

    def component_arrays(self) -> tuple[Weights, FloatArray, Covariances]:
        """Return the weights, means, and covariances as arrays."""
        return self.weights, self.means, self.covariances

    @property
    def n_features(self) -> int:
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
        weight_entropy = float(-np.sum(self.weights * np.log(self.weights + 1e-12)))
        covariance_condition_numbers = readonly_copy(
            np.linalg.cond(self.covariances), dtype=np.float64
        )
        return MixtureDiagnostics(
            n_components=self.n_components,
            n_features=self.n_features,
            weight_sum=weight_sum,
            max_weight=max_weight,
            min_weight=min_weight,
            weight_entropy=weight_entropy,
            covariance_condition_numbers=covariance_condition_numbers,
        )

    def _moments(self) -> tuple[FloatArray, Covariance]:
        """Return the mean and covariance of the Gaussian mixture."""
        if self._aggregate_moments is not None:
            return self._aggregate_moments
        weights, means, covariances = self.component_arrays()
        mean = np.sum(weights[:, None] * means, axis=0)
        mean_delta = means - mean
        covariance = np.sum(
            weights[:, None, None]
            * (covariances + mean_delta[:, :, None] * mean_delta[:, None, :]),
            axis=0,
        )
        covariance = 0.5 * (covariance + covariance.T)
        moments = (
            readonly_copy(mean, dtype=np.float64),
            readonly_copy(covariance, dtype=np.float64),
        )
        object.__setattr__(self, "_aggregate_moments", moments)
        return moments

    @property
    def mean(self) -> FloatArray:
        """Aggregate mean, shape (n_features,), independently owned and read-only."""
        return self._moments()[0]

    @property
    def covariance(self) -> Covariance:
        """Aggregate covariance, shape (n_features, n_features), read-only."""
        return self._moments()[1]


def sample_gmm(
    gmm: GaussianMixture, /, n_samples: int, *, rng: np.random.Generator | int | None = None
) -> FloatArray:
    """Draw samples from a Gaussian mixture."""
    n_samples = validate_positive_int(n_samples, name="n_samples")
    rng = np.random.default_rng(rng)

    component_ids = rng.choice(gmm.n_components, size=n_samples, p=gmm.weights)
    cholesky = gmm.cholesky()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)
    samples = gmm.means[component_ids] + np.einsum("nij,nj->ni", cholesky[component_ids], eps)
    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the log-density of a Gaussian mixture without an explicit Python loop."""
    x = as_points(x, n_features=gmm.n_features, name="x")

    with np.errstate(divide="ignore"):
        log_terms = _component_logpdf(gmm, x) + np.log(gmm.weights)[None, :]
    return logsumexp(log_terms, axis=1)


def _component_logpdf(gmm: GaussianMixture, x: FloatArray) -> FloatArray:
    """Evaluate component densities at validated points using cached factors."""
    diff = x[None, :, :] - gmm.means[:, None, :]
    rhs = np.swapaxes(diff, 1, 2)
    whitened = np.linalg.solve(gmm.cholesky(), rhs)
    mahalanobis = np.sum(whitened**2, axis=1)
    constant = gmm.n_features * np.log(2.0 * np.pi)
    return (-0.5 * (constant + gmm.log_determinants()[:, None] + mahalanobis)).T


def gmm_pdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the density of a Gaussian mixture."""
    return np.exp(gmm_logpdf(x, gmm))
