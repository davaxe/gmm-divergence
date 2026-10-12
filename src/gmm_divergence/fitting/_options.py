"""Options for fitting candidate-mixture weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._sampling import BatchSampleSpec, Draw, SampleSpec
from gmm_divergence._core._validation import validate_nonnegative_finite, validate_unit_interval
from gmm_divergence._core._validation import validate_positive_finite as _validate_positive_float
from gmm_divergence._core._validation import validate_positive_int as _validate_positive_int


@dataclass(frozen=True, slots=True, kw_only=True)
class SimplexSLSQP:
    r"""SLSQP optimizer over simplex-constrained weights.

    The optimizer works directly with the mixture weights
    $\mathbf{w}\in\Delta_L$ for $L$ retained candidates, where

    $$
    \Delta_L =
    \left\{
        \mathbf{w}\in\mathbb{R}^{L}:
        w_i\geq0,\ \sum_{i=1}^{L}w_i=1
    \right\}.
    $$

    This keeps the optimization variables interpretable but requires the
    optimizer to enforce the simplex equality and bound constraints.

    Parameters
    ----------
    tol : float, default=1e-8
        Finite, positive convergence tolerance passed to scipy.optimize.minimize.
    max_iterations : int, default=1000
        Positive iteration budget.
    min_weight : float, default=1e-12
        Finite, nonnegative lower bound per retained candidate. The product of
        min_weight and the number of retained candidates must not exceed one.
    initial_weights : array-like or None, default=None
        Finite, nonnegative 1D coefficients with positive total mass and one
        entry per retained candidate. Normalized at fitting time. None starts
        uniformly. Inputs are copied and made read-only.
    """

    tol: float = 1e-8
    """Optimizer convergence tolerance."""
    max_iterations: int = 1000
    """Maximum number of optimizer iterations."""
    min_weight: float = 1e-12
    """Lower bound for each retained candidate weight.

    Ideally, this would be zero, but SLSQP can fail when weights are exactly
    zero. Setting it to small positive values avoids numerical issues while
    still allowing negligible candidate weights. This bound does not remove
    candidates or their Gaussian components from the fitted mixture."""
    initial_weights: npt.ArrayLike | None = None
    """Optional initial simplex weights for the active candidates."""

    def __post_init__(self) -> None:
        _ = _validate_positive_float(self.tol, name="tol")
        _ = _validate_positive_int(self.max_iterations, name="max_iterations")
        _ = validate_nonnegative_finite(self.min_weight, name="min_weight")
        _freeze_optional_vector(self, "initial_weights")


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardKL:
    r"""Forward KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{KL}}\!\left(p \,\|\, q_{\mathbf{w}}\right)
    =
    \mathbb{E}_{X \sim p}
    \left[
        \log p(X) - \log q_{\mathbf{w}}(X)
    \right],
    $$

    where $p$ is the target density and
    $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$ combines $L$ fixed candidate densities
    with nonnegative weights summing to one. $X$ denotes a random observation
    from the first distribution. Since $\log p(X)$ does
    not depend on $\mathbf{w}$, its contribution is cached. Reported values
    include this constant and estimate KL in nats.

    Parameters
    ----------
    target_sampling : SampleSpec, default=Draw()
        Source of fixed target observations. Draw defaults to 10000 samples;
        Samples reuses target observations; Stratified retains integration mass.
    """

    target_sampling: SampleSpec = field(default_factory=Draw)
    """Sampling specification for the expectation under target.

    Use `sampling.Draw(...)` to draw fresh samples, `sampling.Samples(...)` for
    precomputed reference samples, or `sampling.Stratified(...)` when `p` is a
    Gaussian-family distribution and fixed per-component counts are desired.
    """

    def __post_init__(self) -> None:
        _validate_sampling(self.target_sampling, name="target_sampling", operation="sample")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReverseKL:
    r"""Reverse KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, p\right)
    =
    \mathbb{E}_{X \sim q_{\mathbf{w}}}
    \left[
        \log q_{\mathbf{w}}(X) - \log p(X)
    \right].
    $$

    Here $p$ is the target density and $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$
    combines $L$ fixed candidate densities with nonnegative weights summing to
    one. $X$ is a random observation under the indicated distribution.
    The implementation evaluates a fixed-sample estimator using samples from
    each candidate mixture $q_i$. The reference density $p$ is evaluated at
    those samples, so no samples from $p$ are required.

    Parameters
    ----------
    candidate_sampling : BatchSampleSpec, default=Draw()
        Fixed batch per candidate. Draw defaults to 10000 observations per
        retained candidate. SampleBatches must cover every original candidate;
        filtering and reordering are applied to those batches during preparation.
    """

    candidate_sampling: BatchSampleSpec = field(default_factory=Draw)
    """Sampling specification for fixed batches from each q_i.

    Use `sampling.Draw(...)` to draw one batch per candidate distribution,
    `sampling.Stratified(...)` for fixed per-component counts, or
    `sampling.SampleBatches(...)` to provide those batches directly.
    """

    def __post_init__(self) -> None:
        _validate_sampling(
            self.candidate_sampling, name="candidate_sampling", operation="sample_batches"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BidirectionalKL:
    r"""Bidirectional KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing a weighted combination of
    forward and reverse KL objectives:

    $$
    \beta\,D_{\mathrm{KL}}\!\left(p \,\|\, q_{\mathbf{w}}\right)
    +
    (1-\beta)\,D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, p\right),
    \qquad \beta\in[0,1].
    $$

    Here $p$ is the target density, $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$ is
    the combined candidate density, and $\beta$ is `forward_weight`.
    Values of $\beta$ closer to one emphasize coverage of $p$ by
    $q_{\mathbf{w}}$, while values closer to zero emphasize the reverse-KL
    objective.

    Parameters
    ----------
    target_sampling : SampleSpec, default=Draw()
        Fixed observations from the target for the forward term. Not consumed
        when forward_weight is zero.
    candidate_sampling : BatchSampleSpec, default=Draw()
        Fixed batch per candidate for the reverse term. Precomputed batches
        cover original candidates. Not consumed when forward_weight is one.
    forward_weight : float, default=0.5
        Finite weight in [0, 1], represented by beta in the formula. The
        reverse term receives 1 - forward_weight.
    """

    target_sampling: SampleSpec = field(default_factory=Draw)
    """Sampling specification for the forward term under p."""
    candidate_sampling: BatchSampleSpec = field(default_factory=Draw)
    """Sampling specification for the reverse term under each q_i.

    Use `sampling.Draw(...)` to draw one batch per candidate distribution,
    `sampling.Stratified(...)` for fixed per-component counts, or
    `sampling.SampleBatches(...)` to provide those batches directly.
    """
    forward_weight: float = 0.5
    """Weight assigned to the forward KL term."""

    def __post_init__(self) -> None:
        _validate_sampling(self.target_sampling, name="target_sampling", operation="sample")
        _validate_sampling(
            self.candidate_sampling, name="candidate_sampling", operation="sample_batches"
        )
        _ = validate_unit_interval(self.forward_weight, name="forward_weight")


@dataclass(frozen=True, slots=True, kw_only=True)
class JensenShannon:
    r"""Jensen-Shannon fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{JS}}\!\left(p, q_{\mathbf{w}}\right)
    =
    \frac{1}{2}D_{\mathrm{KL}}\!\left(p \,\|\, m_{\mathbf{w}}\right)
    +
    \frac{1}{2}D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, m_{\mathbf{w}}\right),
    $$

    where

    $$
    m_{\mathbf{w}} = \frac{1}{2}p + \frac{1}{2}q_{\mathbf{w}}.
    $$

    Here $p$ is the target density and $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$
    combines fixed candidate densities with nonnegative weights summing to one.
    The exact divergence is symmetric and bounded by $\log 2$ nats. Fitting
    evaluates its sampled approximation using fixed observations
    from `p` and each candidate mixture `q_i` during optimization.

    Parameters
    ----------
    target_sampling : SampleSpec, default=Draw()
        Fixed observations from the target for its midpoint-divergence term.
    candidate_sampling : BatchSampleSpec, default=Draw()
        Fixed batch per candidate for the fitted mixture's midpoint term.
        Precomputed batches cover every original candidate.

    Notes
    -----
    Values estimate divergence in nats, not its square-root distance. Finite
    samples may produce estimates outside the exact [0, log(2)] range.
    """

    target_sampling: SampleSpec = field(default_factory=Draw)
    """Sampling specification for the term under p."""
    candidate_sampling: BatchSampleSpec = field(default_factory=Draw)
    """Sampling specification for fixed batches from each q_i.

    Use `sampling.Draw(...)` to draw one batch per candidate distribution,
    `sampling.Stratified(...)` for fixed per-component counts, or
    `sampling.SampleBatches(...)` to provide those batches directly.
    """

    def __post_init__(self) -> None:
        _validate_sampling(self.target_sampling, name="target_sampling", operation="sample")
        _validate_sampling(
            self.candidate_sampling, name="candidate_sampling", operation="sample_batches"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MomentMatching:
    r"""Moment-matching fitting objective configuration.

    Fits weights by matching moments of $q_{\mathbf{w}}$ to moments of the
    reference distribution $p$ instead of optimizing a sampled KL estimate.
    The first-moment objective compares mixture means,

    $$
    \left\|\mu_{q_{\mathbf{w}}} - \mu_p\right\|_2^2,
    $$

    where $\mu_p$ and $\mu_{q_{\mathbf{w}}}$ are the aggregate means of target
    $p$ and combined density $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$. With raw
    second moments $M_{2,p}=\Sigma_p+\mu_p\mu_p^\top$ and
    $M_{2,q_{\mathbf{w}}}=\Sigma_{q_{\mathbf{w}}}
    +\mu_{q_{\mathbf{w}}}\mu_{q_{\mathbf{w}}}^\top$, the optional objective is

    $$
    \|\mu_{q_{\mathbf{w}}}-\mu_p\|_2^2
    +\|M_{2,q_{\mathbf{w}}}-M_{2,p}\|_{\mathrm{F}}^2.
    $$

    Here $\Sigma$ denotes aggregate covariance and the Frobenius norm squares
    and sums every matrix entry, matching the flattened array implementation.

    Parameters
    ----------
    fit_second_moments : bool, default=False
        Include the flattened raw second moment `covariance + outer(mean, mean)`
        alongside the mean. These are raw moments, not covariances alone.

    Notes
    -----
    No samples are drawn. The value is the squared Euclidean residual of the
    concatenated raw moments; it is not a divergence in nats and depends on
    feature scale.
    """

    fit_second_moments: bool = False
    """Whether to include raw second moments alongside the feature means."""


Optimizer: TypeAlias = SimplexSLSQP
FitObjective: TypeAlias = ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching


def validate_optimizer(optimizer: object) -> Optimizer:
    """Reject unsupported optimizers before any preparation work."""
    if not isinstance(optimizer, SimplexSLSQP):
        msg = "optimizer must be a valid optimizer."
        raise TypeError(msg)
    return optimizer


def _freeze_optional_vector(instance: object, name: str) -> None:
    value = getattr(instance, name)
    if value is None:
        return
    vector = readonly_copy(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        msg = f"{name} must be a finite 1D array."
        raise ValueError(msg)
    object.__setattr__(instance, name, vector)


def _validate_sampling(sampling: object, *, name: str, operation: str) -> None:
    if not callable(getattr(sampling, operation, None)):
        msg = f"{name} must implement {operation} and return SampleBatch records."
        raise TypeError(msg)
