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
    """Monte Carlo sampling diagnostics for one directed estimate.

    Attributes
    ----------
    n_samples : int
        Number of observations used, including all adaptive batches.
    standard_error : float or None
        Estimated standard error in nats, when supported by the sampling design.
        None indicates that uncertainty cannot be estimated, not zero error.
    target_reached : bool or None
        Whether adaptive sampling reached its requested standard error. False
        means the sample budget was exhausted first; None means a fixed run.
    """

    n_samples: int
    standard_error: float | None
    target_reached: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DivergenceResult:
    """A divergence in nats with estimator and evaluation metadata.

    Attributes
    ----------
    value : float
        Directed or composite divergence value. Estimates may be negative;
        categorical KL may be infinite when comparison support is missing.
    divergence : str
        Quantity computed: kl, categorical_kl, symmetric_kl, or jensen_shannon.
    estimator : KLEstimator or None
        Configuration used for a directed Gaussian-family estimate. None for
        categorical and composite results; composites retain it on their terms.
    n_evaluations : int or None
        Number of sample or deterministic integration points evaluated. None
        for methods without such counts, or composites with an uncounted term.
        Each point counts once, even though both densities are evaluated there.
    monte_carlo : MonteCarloStatistics or None
        Sampling diagnostics for a directed Monte Carlo estimate. Other methods
        and composites have no aggregate sampling-uncertainty record.
    terms : tuple of DivergenceResult
        Ordered directional results for composite quantities: forward then
        reverse for symmetric KL, p-to-midpoint then q-to-midpoint for JS.
    """

    value: float
    divergence: str = "kl"
    estimator: KLEstimator | None = None
    n_evaluations: int | None = None
    monte_carlo: MonteCarloStatistics | None = None
    terms: tuple[DivergenceResult, ...] = ()

    @property
    def n_samples(self) -> int | None:
        """Monte Carlo observation counts, excluding deterministic points.

        Composite results sum counts from their sampled terms. Returns None
        when no term reports Monte Carlo observations; repeated observations in
        different terms are counted separately.
        """
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
    """Fitted candidate weights and a directly usable mixture with provenance.

    Attributes
    ----------
    weights : ndarray, shape (n_candidates,)
        Independent, read-only weights in original candidate order. Rejected
        candidates receive zero weight.
    objective : FitObjective
        Objective and sampling configuration used for preparation.
    optimizer : Optimizer
        Optimizer configuration, including any initial conditions.
    objective_value : float
        Final cached objective: KL or JS in nats, including target constants,
        or a squared raw-moment residual for MomentMatching.
    mixture : GaussianMixture
        Combined fitted distribution. Its mapping identifies original candidate
        indices and component positions within those candidates.
    converged : bool
        Whether the optimizer reported successful termination.
    n_iterations : int
        Number of optimizer iterations.
    message : str
        Optimizer termination explanation.
    active_candidate_indices : tuple of int
        Retained positions in their optimization order. Slice weights by these
        indices when constructing a warm start for the prepared problem.
    """

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
        """List original candidate indices and weights in descending weight order.

        Returns
        -------
        list of tuple
            Index and weight pairs, including rejected candidates with zero
            weight. Ties preserve the original candidate order.
        """
        return sorted(enumerate(map(float, self.weights)), key=lambda item: -item[1])

    def candidate_weight_dict(self) -> dict[int, float]:
        """Return weights keyed by original candidate index.

        Returns
        -------
        dict
            Candidate indices mapped to float weights, including zero weights.
            Insertion order follows candidate_weights(), descending by weight.
        """
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
    """Categorical forward-KL fit with independently owned, read-only arrays.

    Attributes
    ----------
    weights : ndarray, shape (n_candidates,)
        Simplex weights in original candidate-row order.
    probabilities : ndarray, shape (n_categories,)
        Weighted combination of candidate rows in original category order.
    optimizer : Optimizer
        Optimizer configuration, including any initial conditions.
    objective_value : float
        Exact forward KL from the target to fitted probabilities, in nats.
    converged : bool
        Whether the optimizer reported successful termination.
    n_iterations : int
        Number of optimizer iterations.
    message : str
        Optimizer termination explanation.
    objective : str
        Always forward_kl; categorical fitting supports this objective only.
    """

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
    """Joint KL of aligned labels and observations, an upper bound on marginal KL.

    Attributes
    ----------
    value : float
        Total joint KL in nats, equal to weight_kl + weighted_component_kl.
    weight_kl : float
        Categorical KL between aligned mixture weights. Infinite when a
        positive reference weight has zero comparison weight.
    component_kls : ndarray, shape (n_components,)
        Independent, read-only directed Gaussian KL values for aligned pairs.
    weighted_component_kl : float
        Sum of component_kls weighted by the reference mixture's weights.
    """

    value: float
    weight_kl: float
    component_kls: FloatArray
    weighted_component_kl: float

    def __post_init__(self) -> None:
        """Take independent, read-only ownership of result arrays."""
        object.__setattr__(
            self, "component_kls", readonly_copy(self.component_kls, dtype=np.float64)
        )
