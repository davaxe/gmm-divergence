"""Prepare, solve, and report mixture-weight fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence._core._validation import as_weights
from gmm_divergence.distributions._combine import (
    CombinedGaussianMixture,
    MixtureMapping,
    combine_gaussians,
)
from gmm_divergence.fitting._objectives import (
    ObjectiveFn,
    build_simplex_objective,
    softmax,
    with_softmax,
)
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitMethod,
    FitObjective,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
    SoftmaxLBFGSB,
)
from gmm_divergence.results import FitResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray, Weights
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._selector import CandidateSelection, CandidateSelector


@dataclass(frozen=True, slots=True)
class FitSolution:
    """Optimizer output for a prepared mixture-weight fit.

    ``parameters`` contains the optimizer coordinates: logits for
    :class:`SoftmaxLBFGSB` and simplex weights for :class:`SimplexSLSQP`.
    ``active_weights`` always contains the corresponding candidate weights.
    Both arrays are independent and read-only, making them safe to reuse as
    warm-start inputs.
    """

    method: FitMethod
    """Optimizer configuration used to obtain the solution."""
    parameters: FloatArray
    """Final optimizer coordinates for the active candidates."""
    active_weights: Weights
    """Final simplex weights for the active candidates."""
    objective_value: float
    """Final scalar objective value."""
    iterations: int
    """Number of optimizer iterations."""
    converged: bool
    """Whether the optimizer reported convergence."""
    optimizer_message: str
    """Optimizer termination message."""

    def __post_init__(self) -> None:
        parameters = _freeze_vector(self.parameters, name="parameters")
        active_weights = as_weights(
            self.active_weights,
            expected_length=parameters.shape[0],
            name="active_weights",
            normalize=False,
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "active_weights", active_weights)


@dataclass(frozen=True, slots=True, repr=False)
class PreparedFit:
    """Reusable data and objective for fitting mixture weights.

    Preparation performs candidate selection, sampling, and all log-density or
    moment calculations required by the objective. Call :meth:`solve` any
    number of times without repeating that work, then pass a solution to
    :meth:`report` to construct the full :class:`~gmm_divergence.FitResult`.
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

    def solve(self, *, method: FitMethod) -> FitSolution:
        """Optimize the prepared objective without rebuilding cached data."""
        return solve_prepared_fit(self, method=method)

    def report(self, solution: FitSolution, /) -> FitResult:
        """Construct a full fit result from an optimizer solution."""
        return build_fit_result(self, solution)


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


def prepare_mixture_weight_fit(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
    candidate_selection: CandidateSelector | None = None,
) -> PreparedFit:
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
    simplex_objective = build_simplex_objective(
        objective=objective, p=p, q_i=active_candidates, p_samples=p_samples, q_samples=q_samples
    )
    return PreparedFit(
        objective=objective,
        active_candidates=active_candidates,
        active_candidate_indices=active_indices,
        candidate_count=original_count,
        simplex_objective=simplex_objective,
    )


def solve_prepared_fit(prepared: PreparedFit, /, *, method: FitMethod) -> FitSolution:
    """Optimize a prepared fitting objective."""
    n_candidates = prepared.n_active_candidates
    if isinstance(method, SoftmaxLBFGSB):
        initial = (
            np.array(method.initial_logits, dtype=np.float64)
            if method.initial_logits is not None
            else np.zeros(n_candidates, dtype=np.float64)
        )
        _validate_initial_vector(initial, n_candidates, name="initial_logits")
        scipy_objective = with_softmax(prepared.simplex_objective)
        scipy_method = "L-BFGS-B"
        constraints = ()
        bounds = None

        def weights_from_parameters(values: FloatArray) -> Weights:
            return softmax(values)

    else:
        if method.min_weight * n_candidates > 1.0:
            msg = "min_weight is infeasible for the number of active candidates."
            raise ValueError(msg)
        initial = (
            as_weights(method.initial_weights, expected_length=n_candidates, name="initial_weights")
            if method.initial_weights is not None
            else np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)
        )
        scipy_objective = prepared.simplex_objective
        scipy_method = "SLSQP"
        constraints = LinearConstraint(
            A=np.ones((1, n_candidates), dtype=np.float64),
            lb=np.array([1.0], dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
        )
        bounds = Bounds(
            lb=np.full(n_candidates, method.min_weight, dtype=np.float64),
            ub=np.ones(n_candidates, dtype=np.float64),
        )

        def weights_from_parameters(values: FloatArray) -> Weights:
            return values.astype(np.float64)

    result = minimize(
        scipy_objective,
        initial,
        method=scipy_method,
        jac=True,
        constraints=constraints,
        bounds=bounds,
        tol=method.tol,
        options={"maxiter": method.max_iterations},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    return FitSolution(
        method=method,
        parameters=parameters,
        active_weights=weights_from_parameters(parameters),
        objective_value=float(result.fun),
        iterations=result.nit,
        converged=bool(result.success),
        optimizer_message=str(result.message),
    )


def build_fit_result(prepared: PreparedFit, solution: FitSolution, /) -> FitResult:
    """Map a prepared-fit solution back to the original candidates."""
    _validate_initial_vector(
        solution.active_weights, prepared.n_active_candidates, name="solution.active_weights"
    )
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
    return FitResult(
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


def _validate_initial_vector(values: FloatArray, expected_length: int, *, name: str) -> None:
    if values.ndim != 1 or values.shape[0] != expected_length or not np.all(np.isfinite(values)):
        msg = f"{name} must be a finite 1D array with length {expected_length}."
        raise ValueError(msg)


def _freeze_vector(values: npt.ArrayLike, *, name: str) -> FloatArray:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.shape[0] == 0 or not np.all(np.isfinite(vector)):
        msg = f"{name} must be a nonempty finite 1D array."
        raise ValueError(msg)
    vector.setflags(write=False)
    return vector


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
