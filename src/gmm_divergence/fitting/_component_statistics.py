from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    """Responsibility-weighted empirical statistics with read-only arrays."""

    weights: FloatArray
    """Soft counts divided by the number of observations.
    Shape (n_components,).
    """
    means: FloatArray
    """Responsibility-weighted empirical means.
    Shape (n_components, n_features).
    """
    covariances: FloatArray
    """Weighted maximum-likelihood covariances, which may be singular.
    Shape (n_components, n_features, n_features).
    """
    soft_counts: FloatArray
    """Sum of responsibilities for each component.
    Shape (n_components,).
    """
    effective_sample_sizes: FloatArray
    """Kish concentration measure, soft_counts squared divided by the sum of
    squared responsibilities; not a count of independent observations.
    Shape (n_components,).
    """

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
    empty_component_policy: Literal["raise", "reference"] = "raise",
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
    if (responsibilities is None) == (mixture is None):
        msg = "Exactly one of 'responsibilities' or 'mixture' must be provided."
        raise ValueError(msg)

    if empty_component_policy == "reference" and mixture is None:
        msg_0 = "empty_component_policy='reference' requires mixture."
        raise ValueError(msg_0)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        msg = "x must be a nonempty, finite 2D array."
        raise ValueError(msg)

    r = np.asarray(
        mixture.responsibilities(x) if mixture is not None else responsibilities, dtype=np.float64
    )
    if r.ndim != 2 or r.shape[0] != len(x) or r.shape[1] == 0:
        msg = "responsibilities must have shape (n_samples, n_components)."
        raise ValueError(msg)
    r = as_probability_rows(r, name="responsibilities")

    n_samples, n_features = x.shape
    n_components = r.shape[1]
    counts = r.sum(axis=0)
    empty = np.flatnonzero(counts == 0)

    if empty.size and empty_component_policy == "raise":
        msg = f"Components with zero responsibility mass: {empty.tolist()}."
        raise ValueError(msg)

    if empty_component_policy == "reference" and mixture is not None:
        means = mixture.means.copy()
        covariances = mixture.covariances.copy()
    else:
        means = np.empty((n_components, n_features), dtype=np.float64)
        covariances = np.empty((n_components, n_features, n_features), dtype=np.float64)

    effective_sample_sizes = np.zeros(n_components, dtype=np.float64)

    for k in np.flatnonzero(counts > 0):
        # Scale before normalization to handle extremely small responsibilities.
        column = r[:, k]
        scaled = column / column.max()
        weights = scaled / scaled.sum()

        means[k] = weights @ x
        centered = x - means[k]
        covariance = (centered * weights[:, None]).T @ centered
        covariances[k] = 0.5 * (covariance + covariance.T)

        # Equivalent to counts[k]**2 / sum(r[:, k]**2), without underflow.
        effective_sample_sizes[k] = 1.0 / (weights @ weights)

    return ComponentStatistics(
        weights=counts / n_samples,
        means=means,
        covariances=covariances,
        soft_counts=counts,
        effective_sample_sizes=effective_sample_sizes,
    )
