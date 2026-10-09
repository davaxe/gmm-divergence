from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._validation import as_probability_rows
from gmm_divergence.covariance import regularize_covariance
from gmm_divergence.distributions import GaussianMixture

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
from gmm_divergence.covariance import CovarianceRegularizer


@dataclass(frozen=True, slots=True)
class ComponentStatistics:
    """Responsibility-weighted empirical component statistics."""

    weights: FloatArray
    means: FloatArray
    covariances: FloatArray
    soft_counts: FloatArray
    effective_sample_sizes: FloatArray

    def __post_init__(self) -> None:
        for name in ("weights", "means", "covariances", "soft_counts", "effective_sample_sizes"):
            array = readonly_copy(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, array)

    def to_gaussian_mixture(
        self, *, regularizer: CovarianceRegularizer | None = None
    ) -> GaussianMixture:
        """Convert statistics, optionally regularizing covariances without modifying them.

        Singular empirical covariances require regularization for conversion.
        """
        return GaussianMixture.from_arrays(
            weights=self.weights,
            means=self.means,
            covariances=self.covariances
            if regularizer is None
            else regularize_covariance(self.covariances, regularizer=regularizer, batched=True),
        )


def component_statistics(
    x: npt.ArrayLike,
    /,
    *,
    responsibilities: npt.ArrayLike | None = None,
    mixture: GaussianMixture | None = None,
) -> ComponentStatistics:
    r"""Compute empirical component statistics from fixed responsibilities.

    For observations $x_i$ and responsibilities $r_{ik}$, compute

    $$
    N_k = \sum_i r_{ik}, \qquad \hat{\pi}_k = N_k/N,
    \qquad \hat{\mu}_k = \frac{\sum_i r_{ik}x_i}{N_k},
    $$

    $$
    \hat{\Sigma}_k = \frac{\sum_i r_{ik}(x_i-\hat{\mu}_k)
    (x_i-\hat{\mu}_k)^\top}{N_k},
    \qquad N_{\mathrm{eff},k} = \frac{N_k^2}{\sum_i r_{ik}^2}.
    $$

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Nonempty, finite observations.
    responsibilities : array-like, shape (n_samples, n_components), optional
        Finite, nonnegative assignments summing to one per observation.
        Exactly one of `responsibilities` or `mixture` must be provided.
    mixture : GaussianMixture, optional
        Reference mixture used to compute responsibilities without refitting.

    Returns
    -------
    ComponentStatistics
        Independent, read-only `weights`, `means`, `covariances`, `soft_counts`,
        and `effective_sample_sizes`, preserving component order.

    Notes
    -----
    This performs a single moment-estimation step, not an iterative GMM fit.
    Invalid inputs and components with zero responsibility mass raise `ValueError`;
    empty components are rejected rather than dropped to preserve alignment.

    Covariances use weighted maximum-likelihood estimates without an
    unbiasedness correction and may be singular. Apply optional regularization
    through `ComponentStatistics.to_gaussian_mixture(regularizer=...)`.
    Small soft counts may give unreliable estimates; regularization cannot
    compensate for insufficient data.

    Kish effective sample size measures concentration of responsibility weights,
    not component mass or the number of independent observations.
    """
    x, r = _validate_component_statistics_inputs(
        x, responsibilities=responsibilities, mixture=mixture
    )

    n_samples, n_features = x.shape
    n_components = r.shape[1]
    counts = r.sum(axis=0)
    sample_probabilities = counts / n_samples
    component_means = (r.T @ x) / counts[:, None]
    component_covariances = np.empty((n_components, n_features, n_features), dtype=np.float64)

    for k in range(n_components):
        diff = x - component_means[k]

        covariance = ((r[:, k, None] * diff).T @ diff) / counts[k]
        component_covariances[k] = 0.5 * (covariance + covariance.T)

    effective_sample_sizes = counts**2 / np.sum(r**2, axis=0)
    return ComponentStatistics(
        weights=sample_probabilities,
        means=component_means,
        covariances=component_covariances,
        soft_counts=counts,
        effective_sample_sizes=effective_sample_sizes,
    )


def _validate_component_statistics_inputs(
    x: npt.ArrayLike, *, responsibilities: npt.ArrayLike | None, mixture: GaussianMixture | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate observations and resolve component responsibilities."""
    if (responsibilities is None) == (mixture is None):
        msg = "Exactly one of 'responsibilities' or 'mixture' must be provided."
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        msg_0 = "x must be a nonempty, finite 2D array."
        raise ValueError(msg_0)

    r = np.asarray(
        mixture.responsibilities(x) if mixture is not None else responsibilities, dtype=np.float64
    )

    if r.ndim != 2 or r.shape[0] != len(x) or r.shape[1] == 0:
        msg_1 = "responsibilities must have shape (n_samples, n_components)."
        raise ValueError(msg_1)

    r = as_probability_rows(r, name="responsibilities")

    empty = np.flatnonzero(r.sum(axis=0) == 0)
    if empty.size:
        msg_4 = f"Components with zero responsibility mass: {empty.tolist()}."
        raise ValueError(msg_4)

    return x, r
