"""Explicit divergence-estimator configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from gmm_divergence._core._sampling import Draw, SampleSpec
from gmm_divergence._core._validation import validate_positive_finite, validate_positive_int

Approximation: TypeAlias = Literal["nearest", "moment_matching"]


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    """Estimate KL with an explicit sample source.

    Set ``target_standard_error`` to enable adaptive sampling. Adaptive sampling
    is intentionally restricted to ``Draw`` because it needs additional samples.
    """

    sampling: SampleSpec = field(default_factory=Draw)
    target_standard_error: float | None = None
    max_samples: int | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.target_standard_error is None:
            if self.max_samples is not None or self.batch_size is not None:
                msg = "max_samples and batch_size require target_standard_error."
                raise ValueError(msg)
            return
        validate_positive_finite(self.target_standard_error, name="target_standard_error")
        if not isinstance(self.sampling, Draw):
            msg = "Adaptive MonteCarlo requires sampling.Draw."
            raise TypeError(msg)
        if self.max_samples is None:
            msg = "max_samples is required for adaptive MonteCarlo."
            raise ValueError(msg)
        validate_positive_int(self.max_samples, name="max_samples")
        if self.max_samples < self.sampling.n_samples:
            msg = "max_samples must be at least sampling.n_samples."
            raise ValueError(msg)
        if self.batch_size is not None:
            validate_positive_int(self.batch_size, name="batch_size")


@dataclass(frozen=True, slots=True)
class Unscented:
    """Estimate KL using deterministic sigma points."""


@dataclass(frozen=True, slots=True)
class MomentMatchedGaussian:
    """Estimate KL from moment-matched Gaussian approximations."""

    approximation: Approximation = "moment_matching"

    def __post_init__(self) -> None:
        if self.approximation not in {"nearest", "moment_matching"}:
            msg = "approximation must be 'nearest' or 'moment_matching'."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Variational:
    """Estimate KL with the Hershey--Olsen variational approximation."""


@dataclass(frozen=True, slots=True)
class ClosedForm:
    """Compute exact KL for two single Gaussian distributions."""


KLEstimator: TypeAlias = MonteCarlo | Unscented | MomentMatchedGaussian | ClosedForm | Variational
