"""Fit mixture weights to minimize KL divergence."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence._core._validation import as_weights
from gmm_divergence.distributions._combine import (
    CombinedGaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.fitting._objectives import build_objective, softmax
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitParameterization,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.results import FitResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence._core._types import FloatArray, Weights
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._selector import CandidateSelection, CandidateSelector


FitObjective: TypeAlias = ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching
FitOptimizer: TypeAlias = SoftmaxLBFGSB | SimplexSLSQP


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
            return sampling.sample(p), None
        case ReverseKL(p_sampling=p_sampling, q_sampling=q_sampling):
            return p_sampling.sample(p), q_sampling.sample_batches(q_i)
        case BidirectionalKL(p_sampling=p_sampling, q_sampling=q_sampling):
            return p_sampling.sample(p), q_sampling.sample_batches(q_i)
        case JensenShannon(p_sampling=p_sampling, q_sampling=q_sampling):
            return p_sampling.sample(p), q_sampling.sample_batches(q_i)
        case MomentMatching():
            return None, None


def fit_mixture_weights(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
    optimizer: FitOptimizer,
    candidate_selection: CandidateSelector | None = None,
) -> FitResult:
    original_count = len(q_i)
    selection: CandidateSelection | None = None
    if candidate_selection is not None:
        selection = candidate_selection.select(p, q_i)
        _validate_selection(selection, original_count)
        q_i = [q_i[index] for index in selection.selected_indices]
    q_component = _validate_q_i(q_i, p.dim)
    resolved_p_samples, resolved_q_samples = _resolve_objective_samples(p, q_i, objective)
    if isinstance(optimizer, SoftmaxLBFGSB):
        parameterization: FitParameterization = "softmax"
        initial = (
            np.array(optimizer.initial_logits, dtype=np.float64)
            if optimizer.initial_logits is not None
            else np.zeros(q_component, dtype=np.float64)
        )
        _validate_initial_vector(initial, q_component, name="initial_logits")
        scipy_method = "L-BFGS-B"
        constraints = ()
        bounds = None

        def weights_from_result(values: FloatArray) -> Weights:
            return softmax(values)

    else:
        parameterization = "simplex"
        if optimizer.min_weight * q_component > 1.0:
            msg = "min_weight is infeasible for the number of active candidates."
            raise ValueError(msg)
        initial = (
            as_weights(
                optimizer.initial_weights, expected_length=q_component, name="initial_weights"
            )
            if optimizer.initial_weights is not None
            else np.full(q_component, 1.0 / q_component, dtype=np.float64)
        )
        scipy_method = "SLSQP"
        constraints = LinearConstraint(
            A=np.ones((1, q_component), dtype=np.float64),
            lb=np.array([1.0], dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
        )
        bounds = Bounds(
            lb=np.full(q_component, optimizer.min_weight, dtype=np.float64),
            ub=np.ones(q_component, dtype=np.float64),
        )

        def weights_from_result(values: FloatArray) -> Weights:
            return values.astype(np.float64)

    result = minimize(
        build_objective(
            parameterization=parameterization,
            objective=objective,
            p=p,
            q_i=q_i,
            p_samples=resolved_p_samples,
            q_samples=resolved_q_samples,
        ),
        initial,
        method=scipy_method,
        jac=True,
        constraints=constraints,
        bounds=bounds,
        tol=optimizer.tol,
        options={"maxiter": optimizer.max_iterations},
    )
    active_weights: Weights = weights_from_result(result.x)
    weights = np.zeros(original_count, dtype=np.float64)
    active_indices = (
        tuple(range(original_count)) if selection is None else selection.selected_indices
    )
    weights[list(active_indices)] = active_weights
    weights.setflags(write=False)
    combined = combine_gaussians(weights=active_weights, sources=q_i, include_mapping=True)
    remapped_sources = np.take(
        np.asarray(active_indices, dtype=np.intp), combined.mapping.source_index
    )
    remapped_sources.setflags(write=False)
    fitted_mixture = CombinedGaussianMixture(
        mixture=combined.mixture,
        mapping=MixtureMapping(
            source_index=remapped_sources,
            local_component_index=combined.mapping.local_component_index,
        ),
    )
    return FitResult(
        weights=weights,
        fit_objective=objective,
        fit_method=optimizer,
        objective_value=float(result.fun),
        fitted_mixture=fitted_mixture,
        alpha=objective.alpha if isinstance(objective, BidirectionalKL) else None,
        iterations=result.nit,
        converged=bool(result.success),
        active_candidate_indices=active_indices,
        optimizer_message=str(result.message),
    )


def _validate_initial_vector(values: FloatArray, expected_length: int, *, name: str) -> None:
    if values.ndim != 1 or values.shape[0] != expected_length or not np.all(np.isfinite(values)):
        msg = f"{name} must be a finite 1D array with length {expected_length}."
        raise ValueError(msg)


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
