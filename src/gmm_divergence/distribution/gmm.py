from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence.distribution.base import GaussianComponentArrays, GaussianFamily
from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.utils import as_covariances, as_weights, logsumexp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gmm_divergence.typing import Covariances, FloatArray, Weights


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

    @property
    def n_components(self) -> int:
        """Number of components in the Gaussian mixture."""
        return self.weights.shape[0]

    @override
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the log-density of the Gaussian mixture at given points."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        return gmm_logpdf(x=x, gmm=self)

    def chol(self) -> FloatArray:
        """Compute or retrieve the cached Cholesky factors."""
        if self._chol is not None:
            return self._chol

        chol = np.linalg.cholesky(self.covariances).astype(np.float64)
        object.__setattr__(self, "_chol", chol)
        return chol

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
        mean = np.sum(self.weights[:, None] * self.means, axis=0)
        diff = self.means - mean
        cov = np.sum(
            self.weights[:, None, None] * (self.covariances + diff[:, :, None] * diff[:, None, :]),
            axis=0,
        )
        return Gaussian(mean=mean, covariance=0.5 * (cov + cov.T))

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
    rng = np.random.default_rng(rng)

    component_ids = rng.choice(gmm.n_components, size=n_samples, p=gmm.weights)
    chol = gmm.chol()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)
    samples = gmm.means[component_ids] + np.einsum("nij,nj->ni", chol[component_ids], eps)
    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the log-density of a Gaussian mixture."""
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        x = x[None, :]

    n_samples, n_features = x.shape
    n_components = gmm.weights.shape[0]

    log_weights = np.log(gmm.weights)
    chol = gmm.chol()
    log_probs = np.empty((n_samples, n_components), dtype=np.float64)
    constant = n_features * np.log(2.0 * np.pi)

    for k in range(n_components):
        diff = x - gmm.means[k]
        y = cast("npt.NDArray[np.float64]", np.linalg.solve(chol[k], diff.T))
        mahal = np.sum(y * y, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[k])))
        log_gaussian = -0.5 * (constant + log_det + mahal)
        log_probs[:, k] = log_weights[k] + log_gaussian

    return logsumexp(log_probs, axis=1)


def gmm_pdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the density of a Gaussian mixture."""
    return np.exp(gmm_logpdf(x, gmm))
