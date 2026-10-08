from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

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

    def to_gaussian_mixture(
        self, *, regularizer: CovarianceRegularizer | None = None
    ) -> GaussianMixture:
        """Convert the component statistics to a Gaussian mixture."""
        return GaussianMixture.from_arrays(
            weights=self.weights,
            means=self.means,
            covariances=self.covariances
            if regularizer is None
            else regularize_covariance(self.covariances, regularizer=regularizer, batched=True),
        )


def component_statistics(
    x: npt.ArrayLike,
    *,
    responsibilities: npt.ArrayLike | None = None,
    mixture: GaussianMixture | None = None,
    regularizer: CovarianceRegularizer | None = None,
) -> ComponentStatistics:
    r"""Compute responsibility-weighted statistics for mixture components.

    Given a collection of observations and their soft component assignments,
    estimate the component probabilities, means, covariance matrices, and
    effective sample sizes.

    Responsibilities can either be supplied explicitly or computed from
    a reference Gaussian mixture model (GMM). Exactly one of
    `responsibilities` or `mixture` must be provided.

    For observations

    $$
    X = \{x_i\}_{i=1}^{N}, \qquad x_i \in \mathbb{R}^{d},
    $$

    and responsibilities

    $$
    r_{ik} = P(k \mid x_i),
    \qquad
    \sum_{k=1}^{K} r_{ik} = 1,
    $$

    the following statistics are computed for each component
    $k \in \{1,\ldots,K\}$.

    **Effective component counts (soft counts)**

    $$
    N_k = \sum_{i=1}^{N} r_{ik}.
    $$

    **Component probabilities**

    $$
    \hat{\pi}_k = \frac{N_k}{N}.
    $$

    **Responsibility-weighted component means**

    $$
    \hat{\mu}_k =
    \frac{1}{N_k}
    \sum_{i=1}^{N} r_{ik} x_i.
    $$

    **Responsibility-weighted component covariances**

    $$
    \hat{\Sigma}_k =
    \frac{1}{N_k}
    \sum_{i=1}^{N}
    r_{ik}
    (x_i - \hat{\mu}_k)
    (x_i - \hat{\mu}_k)^\top.
    $$

    Covariances are estimated using the weighted maximum-likelihood
    estimator, without an unbiasedness correction.

    **Kish effective sample sizes**

    $$
    N_{\mathrm{eff},k} =
    \frac{
        \left(\sum_{i=1}^{N} r_{ik}\right)^2
    }{
        \sum_{i=1}^{N} r_{ik}^{2}
    }.
    $$

    The Kish effective sample size characterizes the concentration of
    responsibility weights. It should not be interpreted as the number
    of independent observations or as a measure of total component mass.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Observations from which component statistics are estimated.
        All values must be finite.

    responsibilities : array-like, shape (n_samples, n_components), optional
        Soft assignments of observations to mixture components.

        Responsibilities must be finite, nonnegative, and sum to one
        across components for every observation.

        Mutually exclusive with `mixture`.

    mixture : GaussianMixture, optional
        Reference Gaussian mixture used to calculate responsibilities
        through `mixture.responsibilities(x)`.

        The reference mixture is not modified or refitted.

        Mutually exclusive with `responsibilities`.

    regularizer : CovarianceRegularizer, optional
        Optional covariance regularization applied to the estimated
        component covariance matrices.

        If None, the empirical maximum-likelihood covariance matrices
        are returned without regularization.

    Returns
    -------
    ComponentStatistics
        Container with the following attributes:

        - `sample_probabilities` : ndarray, shape (n_components,)
            Estimated component probabilities $\\hat{\\pi}_k$.

        - `component_means` : ndarray, shape (n_components, n_features)
            Responsibility-weighted component means $\\hat{\\mu}_k$.

        - `component_covariances` : ndarray,
          shape (n_components, n_features, n_features)
            Responsibility-weighted component covariance matrices
            $\\hat{\\Sigma}_k$.

        - `effective_sample_counts` : ndarray, shape (n_components,)
            Total responsibility mass $N_k$ assigned to each component.

        - `effective_sample_sizes` : ndarray, shape (n_components,)
            Kish effective sample sizes $N_{\\mathrm{eff},k}$.

    Notes
    -----
    This function performs a single responsibility-weighted moment
    estimation step. It does not fit a GMM through expectation-maximization
    or update the supplied reference mixture.

    When responsibilities are obtained from a shared reference GMM,
    the resulting statistics describe how a particular sample population
    occupies and differs within the reference components.

    The estimates can be used to construct a population-specific GMM:

    $$
    \hat{p}(x) =
    \sum_{k=1}^{K}
    \hat{\pi}_k
    \mathcal{N}
    \left(
        x; \hat{\mu}_k, \hat{\Sigma}_k
    \right).
    $$

    Component indices remain aligned with the supplied responsibilities
    or reference mixture, enabling component-wise comparisons between
    independently characterized sample populations.

    Components with very small responsibility mass may produce unreliable
    estimates. Furthermore, empirical covariance matrices may be singular
    or poorly conditioned, particularly when the effective sample size
    is small relative to the feature dimension.

    Covariance regularization can improve numerical conditioning but
    does not compensate for insufficient statistical information.

    Examples
    --------
    Compute statistics using a reference GMM:

    ```python
    >>> stats = component_statistics(X, mixture=gmm)
    >>> stats.sample_probabilities.shape
    (K,)
    >>> stats.component_means.shape
    (K, D)
    ```

    Compute statistics from precomputed responsibilities:

    ```python
    >>> R = gmm.responsibilities(X)
    >>> stats = component_statistics(X, responsibilities=R)
    ```
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
    if regularizer is not None:
        component_covariances = regularize_covariance(
            component_covariances, regularizer=regularizer, batched=True
        )

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

    if not np.isfinite(r).all() or (r < 0).any():
        msg_2 = "responsibilities must be finite and nonnegative."
        raise ValueError(msg_2)

    if not np.allclose(r.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        msg_3 = "Each row of responsibilities must sum to one."
        raise ValueError(msg_3)

    empty = np.flatnonzero(r.sum(axis=0) == 0)
    if empty.size:
        msg_4 = f"Components with zero responsibility mass: {empty.tolist()}."
        raise ValueError(msg_4)

    return x, r
