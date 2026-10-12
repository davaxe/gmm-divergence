"""Explicit divergence-estimator configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from gmm_divergence._core._sampling import Draw, SampleSpec
from gmm_divergence._core._validation import validate_positive_finite, validate_positive_int


@dataclass(frozen=True, slots=True, kw_only=True)
class MonteCarlo:
    """Estimate KL with an explicit sample source.

    Set `target_standard_error` to enable adaptive sampling. Adaptive sampling
    is intentionally restricted to `Draw` because it needs additional samples.

    Parameters
    ----------
    sampling : SampleSpec, default=Draw()
        Source of observations from the first distribution in a directed KL
        call. Draw defaults to 10000 observations. Precomputed samples must
        originate from that same distribution.
    target_standard_error : float or None, default=None
        Finite, positive stopping threshold in nats. None uses one fixed batch.
        Adaptive runs check the estimated standard error after drawing at least
        sampling.n_samples observations.
    max_samples : int or None, default=None
        Required positive budget for adaptive runs, at least sampling.n_samples.
        Unused budgets are rejected when target_standard_error is None.
    batch_size : int or None, default=None
        Positive adaptive draw size; None uses sampling.n_samples. The final
        batch is shortened to respect max_samples.

    Notes
    -----
    Standard error is available for equally weighted observations with at least
    two samples, or stratified batches with at least two observations in every
    positive-weight component. Arbitrary weighted batches have no uncertainty
    estimate. Precomputed observations must satisfy the assumed sampling design.
    Adaptive results report whether the target was reached; the estimated error
    does not bound approximation bias or guarantee an absolute error.
    """

    sampling: SampleSpec = field(default_factory=Draw)
    target_standard_error: float | None = None
    max_samples: int | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.sampling, "sample", None)):
            msg = "sampling must implement sample(distribution) -> SampleBatch."
            raise TypeError(msg)
        if self.target_standard_error is None:
            if self.max_samples is not None or self.batch_size is not None:
                msg = "max_samples and batch_size require target_standard_error."
                raise ValueError(msg)
            return
        _ = validate_positive_finite(self.target_standard_error, name="target_standard_error")
        if not isinstance(self.sampling, Draw):
            msg = "Adaptive MonteCarlo requires sampling.Draw."
            raise TypeError(msg)
        if self.max_samples is None:
            msg = "max_samples is required for adaptive MonteCarlo."
            raise ValueError(msg)
        _ = validate_positive_int(self.max_samples, name="max_samples")
        if self.max_samples < self.sampling.n_samples:
            msg = "max_samples must be at least sampling.n_samples."
            raise ValueError(msg)
        if self.batch_size is not None:
            _ = validate_positive_int(self.batch_size, name="batch_size")


@dataclass(frozen=True, slots=True, kw_only=True)
class Unscented:
    """Estimate directed KL using deterministic points from reference components.

    Uses twice the number of features in sigma points per reference component,
    weighted by component mass. Reports n_evaluations, without a Monte Carlo
    sample count or standard error. The approximation can be negative.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class MomentMatchedGaussian:
    """Compute KL between Gaussian approximations of both distributions.

    Each approximation matches its distribution's aggregate mean and covariance.
    This is exact for Gaussian inputs, but ignores higher moments of mixtures.
    No samples or uncertainty estimates are produced.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class Variational:
    """Estimate KL with the Hershey--Olsen variational approximation.

    Uses pairwise component KL values and mixture weights without sampling.
    The mixture approximation is not a certified bound on the true divergence.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ClosedForm:
    """Compute exact directed KL for two Gaussian objects, in nats.

    GaussianMixture inputs are rejected even when they have one component.
    Convert those explicitly with as_gaussian() before calling kl_divergence.
    """


KLEstimator: TypeAlias = MonteCarlo | Unscented | MomentMatchedGaussian | ClosedForm | Variational
