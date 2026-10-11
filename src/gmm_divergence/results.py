"""Immutable result containers for divergence and weight fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from gmm_divergence._core._arrays import readonly_copy

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray, Weights
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.divergence._options import KLEstimator
    from gmm_divergence.fitting._options import FitObjective, Optimizer

__all__ = [
    "AlignedJointKLResult",
    "CategoricalWeightFitResult",
    "DivergenceResult",
    "GaussianWeightFitResult",
    "MonteCarloStatistics",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class MonteCarloStatistics:
    """Sampling diagnostics; uncertainty is absent when the design cannot estimate it."""

    n_samples: int
    standard_error: float | None
    target_reached: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DivergenceResult:
    """A divergence in nats, its estimator, and optional directional results."""

    value: float
    divergence: str = "kl"
    estimator: KLEstimator | None = None
    n_evaluations: int | None = None
    monte_carlo: MonteCarloStatistics | None = None
    terms: tuple[DivergenceResult, ...] = ()

    @property
    def n_samples(self) -> int | None:
        """Monte Carlo observations, excluding deterministic integration points."""
        if self.monte_carlo is not None:
            return self.monte_carlo.n_samples
        counts = [term.n_samples for term in self.terms]
        return (
            sum(count for count in counts if count is not None)
            if any(count is not None for count in counts)
            else None
        )

    @override
    def __repr__(self) -> str:
        """Summarize the estimate without printing precomputed observations."""
        estimator = type(self.estimator).__name__ if self.estimator is not None else None
        return (
            f"DivergenceResult(value={self.value:.6g}, divergence={self.divergence!r}, "
            f"estimator={estimator}, n_evaluations={self.n_evaluations}, "
            f"n_samples={self.n_samples})"
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class GaussianWeightFitResult:
    """Fitted candidate weights and a directly usable mixture with provenance."""

    weights: Weights
    objective: FitObjective
    optimizer: Optimizer
    objective_value: float
    mixture: GaussianMixture
    converged: bool
    n_iterations: int
    message: str
    active_candidate_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Take independent, read-only ownership of result arrays."""
        object.__setattr__(self, "weights", readonly_copy(self.weights, dtype=np.float64))

    def candidate_weights(self) -> list[tuple[int, float]]:
        """Original candidate indices and weights in stable descending order."""
        return sorted(enumerate(map(float, self.weights)), key=lambda item: -item[1])

    def candidate_weight_dict(self) -> dict[int, float]:
        """Weights keyed by original candidate index."""
        return dict(self.candidate_weights())

    @override
    def __repr__(self) -> str:
        """Summarize objective, optimizer, and termination."""
        return (
            f"GaussianWeightFitResult(objective={type(self.objective).__name__}, "
            f"optimizer={type(self.optimizer).__name__}, "
            f"objective_value={self.objective_value:.6g}, "
            f"converged={self.converged}, n_iterations={self.n_iterations})"
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CategoricalWeightFitResult:
    """Candidate weights and fitted normalized categorical probabilities."""

    weights: FloatArray
    probabilities: FloatArray
    optimizer: Optimizer
    objective_value: float
    converged: bool
    n_iterations: int
    message: str
    objective: Literal["forward_kl"] = field(default="forward_kl", init=False)

    def __post_init__(self) -> None:
        """Take independent, read-only ownership of result arrays."""
        object.__setattr__(self, "weights", readonly_copy(self.weights, dtype=np.float64))
        object.__setattr__(
            self, "probabilities", readonly_copy(self.probabilities, dtype=np.float64)
        )

    @override
    def __repr__(self) -> str:
        """Summarize objective, optimizer, and termination."""
        return (
            "CategoricalWeightFitResult(objective='forward_kl', "
            f"optimizer={type(self.optimizer).__name__}, "
            f"objective_value={self.objective_value:.6g}, "
            f"converged={self.converged}, n_iterations={self.n_iterations})"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignedJointKLResult:
    """KL of aligned component labels and observations; an upper bound on marginal KL."""

    value: float
    weight_kl: float
    component_kls: FloatArray
    weighted_component_kl: float

    def __post_init__(self) -> None:
        """Take independent, read-only ownership of result arrays."""
        object.__setattr__(
            self, "component_kls", readonly_copy(self.component_kls, dtype=np.float64)
        )
