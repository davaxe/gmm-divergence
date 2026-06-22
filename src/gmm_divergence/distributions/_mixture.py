from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence._core._numeric import logsumexp
from gmm_divergence._core._validation import (
    as_covariances,
    as_points,
    as_positive_sample_count,
    as_weights,
)
from gmm_divergence.distributions._base import (
    GaussianComponentArrays,
    GaussianFamily,
    gaussian_family_moments,
)
from gmm_divergence.distributions._gaussian import Gaussian

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from gmm_divergence._core._types import Covariances, FloatArray, Weights


@dataclass(frozen=True, slots=True, repr=False)
class GaussianMixture(GaussianFamily):
    weights: Weights
    """Weight array of shape (n_components,)."""

    means: FloatArray
    """Mean array of shape (n_components, n_features)."""

    covariances: Covariances
    """Covariance array of shape (n_components, n_features, n_features)."""

    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factors."""

    _log_dets: FloatArray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_arrays(
        cls, weights: npt.ArrayLike, means: npt.ArrayLike, covariances: npt.ArrayLike
    ) -> GaussianMixture:
        """Create a Gaussian mixture from array-like parameters."""
        return cls(
            weights=cast("Weights", weights),
            means=cast("FloatArray", means),
            covariances=cast("Covariances", covariances),
        )

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
        object.__setattr__(self, "means", np.asarray(self.means, dtype=np.float64))

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

    @override
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian mixture at given points."""
        x = as_points(x, n_features=self.dim, name="x")

        return gmm_logpdf(x=x, gmm=self)

    def chol(self) -> FloatArray:
        """Compute or retrieve the cached Cholesky factors."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariances).astype(np.float64)
        object.__setattr__(self, "_chol", chol)
        return chol

    def log_dets(self) -> FloatArray:
        """Compute or retrieve the cached log-determinants of the covariances."""
        if self._log_dets is not None:
            return self._log_dets

        chol = self.chol()
        log_dets = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)
        object.__setattr__(self, "_log_dets", log_dets)
        return log_dets

    @override
    def sample(self, n_samples: int, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Draw samples from the Gaussian mixture."""
        return sample_gmm(self, n_samples=n_samples, rng=rng)

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
        mean, covariance = gaussian_family_moments(self)
        return Gaussian(mean=mean, covariance=covariance)

    @override
    def component_arrays(self) -> GaussianComponentArrays:
        """Return the weights, means, and covariances as arrays."""
        return self.weights, self.means, self.covariances

    def __iter__(self) -> Iterator[tuple[float, Gaussian]]:
        """Iterate over the Gaussian components of the mixture."""
        for k in range(self.n_components):
            yield self.weights[k], self.get_component(k)


def sample_gmm(
    gmm: GaussianMixture, /, n_samples: int, *, rng: np.random.Generator | int | None = None
) -> FloatArray:
    """Draw samples from a Gaussian mixture."""
    n_samples = as_positive_sample_count(n_samples)
    rng = np.random.default_rng(rng)

    component_ids = rng.choice(gmm.n_components, size=n_samples, p=gmm.weights)
    chol = gmm.chol()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)
    samples = gmm.means[component_ids] + np.einsum("nij,nj->ni", chol[component_ids], eps)
    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the log-density of a Gaussian mixture without an explicit Python loop."""
    x = as_points(x, n_features=gmm.dim, name="x")

    _, n_features = x.shape
    log_weights = np.log(gmm.weights)  # (K,)
    chol = gmm.chol()  # (K, D, D)
    log_dets = gmm.log_dets()  # (K,)
    diff = x[None, :, :] - gmm.means[:, None, :]  # (K, N, D)
    rhs = np.swapaxes(diff, 1, 2)  # (K, D, N)
    y = np.linalg.solve(chol, rhs)  # (K, D, N)
    mahal = np.sum(y * y, axis=1)  # (K, N)
    constant = n_features * np.log(2.0 * np.pi)
    log_gaussian = -0.5 * (constant + log_dets[:, None] + mahal)  # (K, N)
    log_probs = log_weights[:, None] + log_gaussian  # (K, N)
    return logsumexp(log_probs.T, axis=1)  # (N,)


def gmm_pdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the density of a Gaussian mixture."""
    return np.exp(gmm_logpdf(x, gmm))
