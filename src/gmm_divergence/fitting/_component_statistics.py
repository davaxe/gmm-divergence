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
    """Responsibility-weighted empirical statistics with read-only arrays.

    Attributes
    ----------
    weights : ndarray, shape (n_components,)
        Soft counts divided by the number of observations.
    means : ndarray, shape (n_components, n_features)
        Responsibility-weighted empirical means.
    covariances : ndarray, shape (n_components, n_features, n_features)
        Weighted maximum-likelihood covariances, which may be singular.
    soft_counts : ndarray, shape (n_components,)
        Sum of responsibilities for each component.
    effective_sample_sizes : ndarray, shape (n_components,)
        Kish concentration measure, soft_counts squared divided by the sum of
        squared responsibilities; not a count of independent observations.
    """

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

        Parameters
        ----------
        regularizer : CovarianceRegularizer or None, default=None
            Optional configuration applied to each empirical covariance.

        Returns
        -------
        GaussianMixture
            Independent mixture preserving empirical component order.

        Notes
        -----
        Construction raises ValueError if a covariance is not positive definite
        after any regularization.
        """
        return GaussianMixture(
            weights=self.weights,
            means=self.means,
            covariances=self.covariances
            if regularizer is None
            else regularize_covariance(self.covariances, regularizer=regularizer),
        )


def component_statistics(
    x: npt.ArrayLike,
    /,
    *,
    responsibilities: npt.ArrayLike | None = None,
    mixture: GaussianMixture | None = None,
) -> ComponentStatistics:
    r"""Compute empirical component statistics from fixed responsibilities.

    For $N$ observations $x_n\in\mathbb{R}^d$ and responsibilities $r_{nk}$
    over $K$ components, with $r_{nk}\geq0$ and $\sum_{k=1}^{K}r_{nk}=1$, compute

    $$
    N_k = \sum_{n=1}^{N}r_{nk}, \qquad \widehat\pi_k=N_k/N,
    \qquad \widehat\mu_k=\frac{\sum_{n=1}^{N}r_{nk}x_n}{N_k},
    $$

    $$
    \widehat\Sigma_k=\frac{\sum_{n=1}^{N}r_{nk}(x_n-\widehat\mu_k)
    (x_n-\widehat\mu_k)^\top}{N_k},
    \qquad N_{\mathrm{eff},k}=\frac{N_k^2}{\sum_{n=1}^{N}r_{nk}^2}.
    $$

    Here $N_k$ is the soft count, $\widehat\pi_k$ the empirical component
    probability, $\widehat\mu_k$ and $\widehat\Sigma_k$ its estimated mean and
    covariance, and $N_{\mathrm{eff},k}$ the Kish concentration measure.

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

    _validate_mixture(mixture)

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


def _validate_mixture(mixture: object) -> None:
    if mixture is not None and not isinstance(mixture, GaussianMixture):
        msg = "mixture must be a GaussianMixture."
        raise TypeError(msg)
