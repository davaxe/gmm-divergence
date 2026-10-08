"""Prepare, solve, and report Gaussian-mixture weight fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_points, as_sample_batches, as_weights
from gmm_divergence.distributions._combine import (
    CombinedGaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.fitting._objectives import build_gaussian_mixture_objective
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitMethod,
    FitObjective,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
)
from gmm_divergence.fitting._simplex import ObjectiveFn, SimplexOptimizationResult, optimize_simplex
from gmm_divergence.results import GaussianMixtureFitResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._selector import CandidateSelection, CandidateSelector


@dataclass(frozen=True, slots=True, repr=False)
class PreparedGaussianMixtureFit:
    """Reusable data and objective for fitting mixture weights.

    Preparation performs candidate selection, sampling, and all log-density or
    moment calculations required by the objective. Call :meth:`solve` any
    number of times without repeating that work, then pass a solution to
    :meth:`report` to construct the full :class:`~gmm_divergence.GaussianMixtureFitResult`.
    """

    objective: FitObjective
    """Fitting-objective configuration used during preparation."""
    active_candidates: tuple[Gaussian | GaussianMixture, ...]
    """Candidate distributions retained for optimization."""
    active_candidate_indices: tuple[int, ...]
    """Indices of active candidates in the original candidate sequence."""
    candidate_count: int
    """Number of candidates in the original sequence."""
    simplex_objective: ObjectiveFn = field(repr=False)
    """Cached objective callable over active candidate weights."""

    @property
    def n_active_candidates(self) -> int:
        """Number of candidates represented by the prepared objective."""
        return len(self.active_candidates)

    def evaluate(self, weights: npt.ArrayLike) -> tuple[float, FloatArray]:
        """Evaluate the cached objective and gradient at active weights.

        The weights must be nonnegative and have a positive sum. They are not
        normalized, which permits finite-difference gradient checks around a
        simplex point.
        """
        weights_arr = as_weights(
            weights, expected_length=self.n_active_candidates, name="weights", normalize=False
        )
        value, gradient = self.simplex_objective(weights_arr)
        return float(value), np.asarray(gradient, dtype=np.float64)

    def solve(self, *, method: FitMethod) -> SimplexOptimizationResult:
        """Optimize the prepared objective without rebuilding cached data."""
        return solve_gaussian_mixture_fit(self, method=method)

    def report(self, solution: SimplexOptimizationResult, /) -> GaussianMixtureFitResult:
        """Construct a full fit result from an optimizer solution."""
        return build_gaussian_mixture_fit_result(self, solution)


def _validate_q_i(q_i: Sequence[Gaussian | GaussianMixture], p_dim: int) -> int:
    """Validate candidate distributions and return their count."""
    q_component = len(q_i)
    if q_component == 0:
        msg = "q_i must contain at least one distribution."
        raise ValueError(msg)
    q_dims = {q.dim for q in q_i}
    if len(q_dims) != 1:
        msg = "All q_i distributions must have the same dimensionality."
        raise ValueError(msg)
    q_dim = next(iter(q_dims))
    if q_dim != p_dim:
        msg = f"p and q_i distributions must have the same dimensionality, got {p_dim} and {q_dim}."
        raise ValueError(msg)
    return q_component


def _resolve_objective_samples(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
) -> tuple[FloatArray | None, FloatArray | None]:
    match objective:
        case ForwardKL(sampling=sampling):
            return _validated_samples(sampling.sample(p), p), None
        case ReverseKL(q_sampling=q_sampling):
            return None, _validated_sample_batches(q_sampling.sample_batches(q_i), q_i)
        case BidirectionalKL(p_sampling=p_sampling, q_sampling=q_sampling, alpha=alpha):
            p_samples = _validated_samples(p_sampling.sample(p), p) if alpha > 0.0 else None
            q_samples = (
                _validated_sample_batches(q_sampling.sample_batches(q_i), q_i)
                if alpha < 1.0
                else None
            )
            return p_samples, q_samples
        case JensenShannon(p_sampling=p_sampling, q_sampling=q_sampling):
            return _validated_samples(p_sampling.sample(p), p), _validated_sample_batches(
                q_sampling.sample_batches(q_i), q_i
            )
        case MomentMatching():
            return None, None


def _validated_samples(
    samples: npt.ArrayLike, distribution: Gaussian | GaussianMixture
) -> FloatArray:
    return as_points(samples, n_features=distribution.dim, name="samples", require_nonempty=True)


def _validated_sample_batches(
    samples: npt.ArrayLike, distributions: Sequence[Gaussian | GaussianMixture]
) -> FloatArray:
    return as_sample_batches(
        samples, n_distributions=len(distributions), n_features=distributions[0].dim, name="samples"
    )


def prepare_gaussian_mixture_fit(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
    candidate_selection: CandidateSelector | None = None,
) -> PreparedGaussianMixtureFit:
    """Prepare and cache all objective data required for fitting."""
    candidates = tuple(q_i)
    original_count = len(candidates)
    selection: CandidateSelection | None = None
    if candidate_selection is not None:
        selection = candidate_selection.select(p, candidates)
        _validate_selection(selection, original_count)
        active_candidates = tuple(candidates[index] for index in selection.selected_indices)
        active_indices = selection.selected_indices
    else:
        active_candidates = candidates
        active_indices = tuple(range(original_count))

    _ = _validate_q_i(active_candidates, p.dim)
    p_samples, q_samples = _resolve_objective_samples(p, active_candidates, objective)
    simplex_objective = build_gaussian_mixture_objective(
        objective=objective, p=p, q_i=active_candidates, p_samples=p_samples, q_samples=q_samples
    )
    return PreparedGaussianMixtureFit(
        objective=objective,
        active_candidates=active_candidates,
        active_candidate_indices=active_indices,
        candidate_count=original_count,
        simplex_objective=simplex_objective,
    )


def solve_gaussian_mixture_fit(
    prepared: PreparedGaussianMixtureFit, /, *, method: FitMethod
) -> SimplexOptimizationResult:
    """Optimize a prepared fitting objective using the shared scipy backend."""
    return optimize_simplex(
        prepared.simplex_objective, n_weights=prepared.n_active_candidates, method=method
    )


def build_gaussian_mixture_fit_result(
    prepared: PreparedGaussianMixtureFit, solution: SimplexOptimizationResult, /
) -> GaussianMixtureFitResult:
    """Map a prepared-fit solution back to the original candidates."""
    if (
        solution.active_weights.ndim != 1
        or solution.active_weights.shape[0] != prepared.n_active_candidates
        or not np.all(np.isfinite(solution.active_weights))
    ):
        msg = (
            "solution.active_weights must be a finite 1D array with length "
            f"{prepared.n_active_candidates}."
        )
        raise ValueError(msg)
    weights = np.zeros(prepared.candidate_count, dtype=np.float64)
    weights[list(prepared.active_candidate_indices)] = solution.active_weights
    weights.setflags(write=False)
    combined = combine_gaussians(
        weights=solution.active_weights, sources=prepared.active_candidates, include_mapping=True
    )
    remapped_sources = np.take(
        np.asarray(prepared.active_candidate_indices, dtype=np.intp), combined.mapping.source_index
    )
    remapped_sources.setflags(write=False)
    fitted_mixture = CombinedGaussianMixture(
        mixture=combined.mixture,
        mapping=MixtureMapping(
            source_index=remapped_sources,
            local_component_index=combined.mapping.local_component_index,
        ),
    )
    return GaussianMixtureFitResult(
        weights=weights,
        fit_objective=prepared.objective,
        fit_method=solution.method,
        objective_value=solution.objective_value,
        fitted_mixture=fitted_mixture,
        alpha=prepared.objective.alpha if isinstance(prepared.objective, BidirectionalKL) else None,
        iterations=solution.iterations,
        converged=solution.converged,
        active_candidate_indices=prepared.active_candidate_indices,
        optimizer_message=solution.optimizer_message,
    )


def _validate_selection(selection: CandidateSelection, candidate_count: int) -> None:
    selected = selection.selected_indices
    rejected = selection.rejected_indices
    if not selected:
        msg = "candidate_selector must retain at least one candidate."
        raise ValueError(msg)
    combined = selected + rejected
    if any(type(index) is not int for index in combined):
        msg = "CandidateSelection indices must be integers."
        raise ValueError(msg)
    if len(set(combined)) != candidate_count or set(combined) != set(range(candidate_count)):
        msg = "CandidateSelection indices must form a complete non-overlapping candidate partition."
        raise ValueError(msg)
