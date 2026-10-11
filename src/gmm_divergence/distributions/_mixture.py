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
    """Immutable mixture of full-covariance Gaussian components.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        Finite, nonnegative coefficients with positive total mass. Normalized
        to sum to one; zero-weight components are retained.
    means : array-like, shape (n_components, n_features)
        Finite component means, with at least one component and feature.
    covariances : array-like, shape (n_components, n_features, n_features)
        Finite, symmetric component covariances. Each must be positive definite
        after any regularization.
    regularizer : CovarianceRegularizer or None, default=None
        Optional regularization applied to component covariances.
    mapping : MixtureMapping or None, default=None
        Optional provenance with one row per component.

    Notes
    -----
    Parameter arrays are copied and made read-only. `means` and `covariances`
    describe components; `mean` and `covariance` describe the full mixture.
    """

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
        """Convert a fitted scikit-learn mixture to full covariance matrices.

        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Fitted estimator using full, tied, diagonal, or spherical covariance.

        Returns
        -------
        GaussianMixture
            Independent distribution with the same component order and weights.

        Raises
        ------
        ValueError
            If the estimator is unfitted or its covariance type is unsupported.
        """
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

        Returns
        -------
        sklearn.mixture.GaussianMixture
            Estimator with independent parameter arrays and full covariances.
            Mixture provenance is not transferred.
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
        """Create a mixture from fixed Gaussian components.

        Parameters
        ----------
        components : sequence of Gaussian
            Nonempty sequence of components with equal feature dimensions.
        weights : array-like, shape (n_components,), or None, default=None
            Finite, nonnegative coefficients with positive total mass. They
            are normalized; None assigns equal component weights.

        Returns
        -------
        GaussianMixture
            Independent mixture preserving the supplied component order.
        """
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
        """Evaluate natural-log mixture densities at one or more points.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_points, n_features)
            Finite observations. A 1D input represents one observation.

        Returns
        -------
        ndarray, shape (n_points,)
            Log densities, including a length-one array for a single point.
        """
        return gmm_logpdf(x=x, gmm=self)

    def cholesky(self) -> FloatArray:
        """Return cached, read-only lower factors, shape (n_components, d, d)."""
        if self._chol is not None:
            return self._chol

        cholesky = np.linalg.cholesky(self.covariances).astype(np.float64)
        cholesky.setflags(write=False)
        object.__setattr__(self, "_chol", cholesky)
        return cholesky

    def log_determinants(self) -> FloatArray:
        """Return cached covariance log-determinants, shape (n_components,).

        The returned float64 array is read-only and uses natural logarithms.
        """
        if self._log_dets is not None:
            return self._log_dets

        cholesky = self.cholesky()
        log_determinants = logdet_from_cholesky(cholesky)
        log_determinants.setflags(write=False)
        object.__setattr__(self, "_log_dets", log_determinants)
        return log_determinants

    def sample(self, n_samples: int, *, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw independent observations from the mixture.

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
            Newly allocated float64 observations in random component order.
        """
        return sample_gmm(self, n_samples=n_samples, rng=rng)

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate mixture densities at one or more points.

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

    def get_component(self, index: int) -> Gaussian:
        """Return a Gaussian component independently of the mixture.

        Parameters
        ----------
        index : int
            Component position in `[0, n_components)`. Negative indices and
            booleans are not accepted.

        Returns
        -------
        Gaussian
            Gaussian with copied parameters from the requested component.

        Notes
        -----
        Index validation raises TypeError for noninteger positions and
        IndexError for positions outside the component range.
        """
        validate_index(index, self.n_components)

        return Gaussian(mean=self.means[index], covariance=self.covariances[index])

    def select_components(self, indices: npt.ArrayLike) -> GaussianMixture:
        """Select components in the requested order and normalize their weights.

        Parameters
        ----------
        indices : array-like, shape (n_selected,)
            Nonempty vector of integer positions in `[0, n_components)`.
            Repeated positions are allowed. Selected weights must have positive
            total mass.

        Returns
        -------
        GaussianMixture
            Independent mixture in the requested order, retaining mapping rows.

        Raises
        ------
        IndexError
            If a component position is out of range.

        Notes
        -----
        Input validation raises TypeError for a noninteger dtype and ValueError
        for an invalid index shape or zero selected mass.
        """
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
        x : array-like, shape (n_features,) or (n_samples, n_features)
            Finite points at which to compute responsibilities. A 1D input
            represents one observation.

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
        """Convert an exactly single-component mixture to a Gaussian.

        Returns
        -------
        Gaussian
            Exact distribution with copied component parameters.

        Raises
        ------
        ValueError
            If the mixture has multiple components. Use `moment_match` for an
            approximation in that case.
        """
        if self.n_components != 1:
            msg = (
                "as_gaussian requires exactly one component; use moment_match for an approximation."
            )
            raise ValueError(msg)
        return self.get_component(0)

    def moment_match(self) -> Gaussian:
        """Approximate the mixture with a Gaussian sharing its aggregate moments.

        Returns
        -------
        Gaussian
            Gaussian with the mixture mean and covariance, including the
            covariance contributed by differences between component means.
        """
        return Gaussian(self.mean, self.covariance)

    def component_arrays(self) -> tuple[Weights, FloatArray, Covariances]:
        """Return read-only weight, mean, and covariance parameter arrays.

        Shapes are `(n_components,)`, `(n_components, n_features)`, and
        `(n_components, n_features, n_features)`. These are the stored arrays,
        not copies or aggregate moments.
        """
        return self.weights, self.means, self.covariances

    @property
    def n_features(self) -> int:
        """Dimensionality of the Gaussian mixture."""
        return self.means.shape[1]

    def __iter__(self) -> Iterator[tuple[float, Gaussian]]:
        """Yield `(weight, Gaussian)` pairs in stored component order."""
        for k in range(self.n_components):
            yield self.weights[k], self.get_component(k)

    def diagnostics(self) -> MixtureDiagnostics:
        """Summarize component weights and covariance conditioning.

        Returns
        -------
        MixtureDiagnostics
            Component and feature counts, weight sum and extrema, weight entropy
            in nats, and independent read-only per-component condition numbers.
        """
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
    """Draw n_samples independent observations from gmm, with shape (N, D).

    n_samples must be positive. rng accepts a generator, integer seed, or None;
    integer seeds restart a stream, generators advance, and None uses fresh
    entropy. Returns a newly allocated float64 observation matrix.
    """
    n_samples = validate_positive_int(n_samples, name="n_samples")
    rng = np.random.default_rng(rng)

    component_ids = rng.choice(gmm.n_components, size=n_samples, p=gmm.weights)
    cholesky = gmm.cholesky()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)
    samples = gmm.means[component_ids] + np.einsum("nij,nj->ni", cholesky[component_ids], eps)
    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate mixture log densities without a Python loop over components.

    x is a finite point (D,) or point matrix (N, D) matching gmm.n_features.
    Returns an (N,) float64 array, including length one for a single point.
    """
    x = as_points(x, n_features=gmm.n_features, name="x")

    with np.errstate(divide="ignore"):
        log_terms = _component_logpdf(gmm, x) + np.log(gmm.weights)[None, :]
    return logsumexp(log_terms, axis=1)


def _component_logpdf(gmm: GaussianMixture, x: FloatArray) -> FloatArray:
    """Evaluate component log densities at validated points using cached factors.

    x has shape (n_points, n_features). Returns float64 log densities with shape
    (n_points, n_components), excluding component weights.
    """
    diff = x[None, :, :] - gmm.means[:, None, :]
    rhs = np.swapaxes(diff, 1, 2)
    whitened = np.linalg.solve(gmm.cholesky(), rhs)
    mahalanobis = np.sum(whitened**2, axis=1)
    constant = gmm.n_features * np.log(2.0 * np.pi)
    return (-0.5 * (constant + gmm.log_determinants()[:, None] + mahalanobis)).T


def gmm_pdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate mixture densities at finite points matching gmm.n_features.

    x has shape (D,) or (N, D). Returns float64 densities with shape (N,),
    including a length-one array for a single point.
    """
    return np.exp(gmm_logpdf(x, gmm))
