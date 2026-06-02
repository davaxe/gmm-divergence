from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence.distribution.base import Distribution
from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.utils import logsumexp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence.typing import FloatArray


@dataclass(frozen=True, slots=True, repr=False)
class GaussianMixture(Distribution):
    weights: FloatArray
    """Weight array of shape (n_components,)."""

    means: FloatArray
    """Mean array of shape (n_components, n_features)."""

    covariances: FloatArray
    """Covariance array of shape (n_components, n_features, n_features)."""

    _chol: FloatArray | None = field(default=None, init=False, repr=False)
    """Cached Cholesky factors."""

    def __post_init__(self) -> None:
        """Validate the shapes of weights, means, and covariances."""
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=np.float64))
        object.__setattr__(self, "means", np.asarray(self.means, dtype=np.float64))
        object.__setattr__(self, "covariances", np.asarray(self.covariances, dtype=np.float64))

        if self.weights.ndim != 1:
            msg = "Weights must be a 1D array."
            raise ValueError(msg)

        if self.means.ndim != 2:
            msg = "Means must be a 2D array."
            raise ValueError(msg)

        n_components = self.weights.shape[0]
        n_features = self.means.shape[1]

        if self.weights.shape[0] != self.means.shape[0]:
            msg = "Number of components in weights and means must match."
            raise ValueError(msg)

        full_shape = (n_components, n_features, n_features)
        diag_shape = (n_components, n_features)

        if self.covariances.shape == diag_shape:
            covariances = np.zeros(full_shape, dtype=np.float64)
            diagonal = np.arange(n_features)
            covariances[:, diagonal, diagonal] = self.covariances
            object.__setattr__(self, "covariances", covariances)
        elif self.covariances.shape != full_shape:
            msg = (
                "Covariances must have shape "
                f"{full_shape} or diagonal shape {diag_shape}, "
                f"got {self.covariances.shape}."
            )
            raise ValueError(msg)

        self.weights.setflags(write=False)
        self.means.setflags(write=False)
        self.covariances.setflags(write=False)

    @classmethod
    def from_arrays(
        cls,
        weights: npt.ArrayLike,
        means: npt.ArrayLike,
        covariances: npt.ArrayLike,
    ) -> GaussianMixture:
        """Create a Gaussian mixture from array-like parameters."""
        weights = np.asarray(weights, dtype=np.float64)
        means = np.asarray(means, dtype=np.float64)
        covariances = np.asarray(covariances, dtype=np.float64)
        return cls(weights=weights, means=means, covariances=covariances)

    @classmethod
    def from_distributions(
        cls,
        weights: npt.ArrayLike,
        distributions: Sequence[Gaussian | GaussianMixture],
    ) -> GaussianMixture:
        """Create a Gaussian mixture by combining Gaussians and/or mixtures."""
        if not distributions:
            msg = "At least one distribution is required."
            raise ValueError(msg)

        weights = np.asarray(weights, dtype=np.float64)

        if weights.ndim != 1:
            msg = "Weights must be a 1D array."
            raise ValueError(msg)

        if len(weights) != len(distributions):
            msg = "Number of weights must match number of distributions."
            raise ValueError(msg)

        dims = {d.dim for d in distributions}
        if len(dims) != 1:
            msg = "All distributions must have the same dimensionality."
            raise ValueError(msg)

        weights_list: list[float] = (weights / weights.sum()).tolist()

        def as_mixture(
            d: Gaussian | GaussianMixture,
        ) -> GaussianMixture:
            if isinstance(d, GaussianMixture):
                return d
            return cls.from_arrays(
                weights=[1.0],
                means=[d.mean],
                covariances=[d.covariance],
            )

        mixtures = [as_mixture(d) for d in distributions]
        return cls.from_arrays(
            weights=np.concatenate([
                w * m.weights for w, m in zip(weights_list, mixtures, strict=True)
            ]),
            means=np.concatenate([m.means for m in mixtures]),
            covariances=np.concatenate([m.covariances for m in mixtures]),
        )

    @property
    @override
    def dim(self) -> int:
        """Return the dimensionality of the Gaussian mixture."""
        return self.means.shape[1]

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
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> FloatArray:
        """Draw samples from the Gaussian mixture."""
        return sample_gmm(self, n_samples=n_samples, rng=rng)

    def get_component(self, index: int) -> Gaussian:
        """Return the Gaussian component at the specified index."""
        if index < 0 or index >= self.n_components:
            msg = f"Component index {index} is out of bounds for {self.n_components} components."
            raise IndexError(msg)

        return Gaussian(
            mean=self.means[index],
            covariance=self.covariances[index],
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


def sample_gmm(
    gmm: GaussianMixture,
    /,
    n_samples: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Draw samples from a Gaussian mixture."""
    rng = np.random.default_rng(rng)

    weights = gmm.weights / np.sum(gmm.weights)

    component_ids = rng.choice(
        gmm.n_components,
        size=n_samples,
        p=weights,
    )

    chol = gmm.chol()
    eps = rng.standard_normal(size=gmm.means[component_ids].shape)

    samples = gmm.means[component_ids] + np.einsum(
        "nij,nj->ni",
        chol[component_ids],
        eps,
    )

    return samples.astype(np.float64, copy=False)


def gmm_logpdf(x: npt.ArrayLike, gmm: GaussianMixture) -> FloatArray:
    """Evaluate the log-density of a Gaussian mixture."""
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        x = x[None, :]

    weights = gmm.weights / np.sum(gmm.weights)
    n_samples, n_features = x.shape
    n_components = weights.shape[0]

    log_weights = np.log(weights)
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


def gmm_pdf(
    x: npt.ArrayLike,
    gmm: GaussianMixture,
) -> FloatArray:
    """Evaluate the density of a Gaussian mixture."""
    return np.exp(gmm_logpdf(x, gmm))
