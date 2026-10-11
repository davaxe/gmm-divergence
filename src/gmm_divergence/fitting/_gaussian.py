"""Reusable Gaussian-mixture weight fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._sampling import (
    SampleBatches,
    sample_batches_with_weights,
    sample_with_weights,
)
from gmm_divergence._core._validation import as_weights
from gmm_divergence.distributions._combine import combine_distributions
from gmm_divergence.distributions._mapping import MixtureMapping
from gmm_divergence.distributions._mixture import GaussianMixture
from gmm_divergence.fitting._objectives import build_gaussian_mixture_objective
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitObjective,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    Optimizer,
    ReverseKL,
)
from gmm_divergence.fitting._simplex import ObjectiveFn, SimplexOptimizationResult, optimize_simplex
from gmm_divergence.results import GaussianWeightFitResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence._core._sampling import BatchSampleSpec
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.fitting._selector import CandidateSelection, CandidateSelector


@dataclass(frozen=True, slots=True, repr=False)
class PreparedGaussianWeightFit:
    """Reusable data and objective for fitting mixture weights.

    Preparation performs candidate selection, sampling, and all log-density or
    moment calculations required by the objective. Call :meth:`fit` any
    number of times without repeating that work, or inspect the objective
    and gradient with :meth:`evaluate`.

    Notes
    -----
    Obtain this object with fitting.prepare_gaussian_weight_fit. The target,
    objective configuration, selected candidates, and sample batches are fixed
    for its lifetime. Changing them requires a new preparation. Optimizer
    choices and initial conditions can change between fit calls.
    """

    objective: FitObjective
    """Fitting-objective configuration used during preparation."""
    active_candidates: tuple[Gaussian | GaussianMixture, ...]
    """Candidate distributions retained for optimization."""
    active_candidate_indices: tuple[int, ...]
    """Indices of active candidates in the original candidate sequence."""
    n_candidates: int
    """Number of candidates in the original sequence."""
    _objective_fn: ObjectiveFn = field(repr=False)
    """Cached objective callable over active candidate weights."""

    def fit(self, *, optimizer: Optimizer) -> GaussianWeightFitResult:
        """Optimize this cached problem and return a complete immutable result.

        For a warm start, pass the previous result's weights at
        `active_candidate_indices` as `SimplexSLSQP.initial_weights`.

        Parameters
        ----------
        optimizer : Optimizer
            Explicit optimizer configuration. Initial vectors refer to retained
            candidates in active_candidate_indices order, not all candidates.

        Returns
        -------
        GaussianWeightFitResult
            Complete result with weights for the original candidate sequence,
            a combined mixture with provenance, and termination diagnostics.
            Rejected candidates receive zero weight.
        """
        solution = optimize_simplex(
            self._objective_fn, n_weights=self.n_active_candidates, optimizer=optimizer
        )
        return _build_fit_result(self, solution)

    @property
    def n_active_candidates(self) -> int:
        """Number of candidates represented by the prepared objective."""
        return len(self.active_candidates)

    def evaluate(self, weights: npt.ArrayLike) -> tuple[float, FloatArray]:
        """Evaluate the cached objective and gradient at active weights.

        The weights must be nonnegative and have a positive sum. They are not
        normalized, which permits finite-difference gradient checks around a
        simplex point.

        Parameters
        ----------
        weights : array-like, shape (n_active_candidates,)
            Finite, nonnegative weights with positive total mass in retained
            candidate order. Use unit total mass for the fitting objective.

        Returns
        -------
        value : float
            Cached objective value. KL and Jensen-Shannon objectives use nats;
            MomentMatching returns a squared raw-moment residual.
        gradient : ndarray, shape (n_active_candidates,)
            Derivatives with respect to the supplied weights.
        """
        weights_arr = as_weights(
            weights, expected_length=self.n_active_candidates, name="weights", normalize=False
        )
        value, gradient = self._objective_fn(weights_arr)
        return float(value), np.asarray(gradient, dtype=np.float64)


def _validate_q_i(candidates: Sequence[Gaussian | GaussianMixture], p_dim: int) -> None:
    """Validate the candidate distributions."""
    q_component = len(candidates)
    if q_component == 0:
        msg = "candidates must contain at least one distribution."
        raise ValueError(msg)
    q_dims = {q.n_features for q in candidates}
    if len(q_dims) != 1:
        msg = "All candidate distributions must have the same dimensionality."
        raise ValueError(msg)
    q_dim = next(iter(q_dims))
    if q_dim != p_dim:
        msg = (
            f"target and candidate distributions must have the same dimensionality, "
            f"got {p_dim} and {q_dim}."
        )
        raise ValueError(msg)


def _resolve_objective_samples(
    target: Gaussian | GaussianMixture,
    candidates: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
    original_candidates: Sequence[Gaussian | GaussianMixture],
    active_indices: tuple[int, ...],
) -> tuple[FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None]:
    def candidate_batches(sampling: BatchSampleSpec) -> tuple[FloatArray, FloatArray | None]:
        if isinstance(sampling, SampleBatches):
            batches, weights = sample_batches_with_weights(original_candidates, sampling)
            indices = list(active_indices)
            return batches[indices], None if weights is None else weights[indices]
        return sample_batches_with_weights(candidates, sampling)

    p_samples = q_samples = p_weights = q_weights = None
    match objective:
        case ForwardKL(target_sampling=sampling):
            p_samples, p_weights = sample_with_weights(target, sampling)
        case ReverseKL(candidate_sampling=candidate_sampling):
            q_samples, q_weights = candidate_batches(candidate_sampling)
        case BidirectionalKL(
            target_sampling=target_sampling,
            candidate_sampling=candidate_sampling,
            forward_weight=forward_weight,
        ):
            if forward_weight > 0.0:
                p_samples, p_weights = sample_with_weights(target, target_sampling)
            if forward_weight < 1.0:
                q_samples, q_weights = candidate_batches(candidate_sampling)
        case JensenShannon(target_sampling=target_sampling, candidate_sampling=candidate_sampling):
            p_samples, p_weights = sample_with_weights(target, target_sampling)
            q_samples, q_weights = candidate_batches(candidate_sampling)
        case MomentMatching():
            pass
    return p_samples, q_samples, p_weights, q_weights


def prepare_gaussian_weight_fit(
    *,
    target: Gaussian | GaussianMixture,
    candidates: Sequence[Gaussian | GaussianMixture],
    objective: FitObjective,
    candidate_selection: CandidateSelector | None = None,
) -> PreparedGaussianWeightFit:
    """Prepare candidate selection, sampling, and cached objective data.

    Parameters
    ----------
    target : Gaussian or GaussianMixture
        Fixed reference distribution.
    candidates : sequence of Gaussian or GaussianMixture
        Nonempty candidate sequence sharing the target's feature dimension.
    objective : FitObjective
        Explicit objective and sampling configuration.
    candidate_selection : CandidateSelector or None, default=None
        Optional selection applied once before generating samples. Precomputed
        candidate batches remain indexed by the original candidate sequence.

    Returns
    -------
    PreparedGaussianWeightFit
        Reusable objective with retained candidate indices and cached densities
        or moments. Changing inputs requires preparing another problem.
    """
    if candidate_selection is not None and not callable(
        getattr(candidate_selection, "select", None)
    ):
        msg = "candidate_selector must implement select(target, candidates)."
        raise TypeError(msg)
    candidates = tuple(candidates)
    _validate_q_i(candidates, target.n_features)
    original_count = len(candidates)
    selection: CandidateSelection | None = None
    if candidate_selection is not None:
        selection = candidate_selection.select(target, candidates)
        _validate_selection(selection, original_count)
        active_candidates = tuple(candidates[index] for index in selection.selected_indices)
        active_indices = selection.selected_indices
    else:
        active_candidates = candidates
        active_indices = tuple(range(original_count))

    _validate_q_i(active_candidates, target.n_features)
    p_samples, q_samples, p_weights, q_weights = _resolve_objective_samples(
        target, active_candidates, objective, candidates, active_indices
    )
    objective_fn = build_gaussian_mixture_objective(
        objective=objective,
        target=target,
        candidates=active_candidates,
        p_samples=p_samples,
        q_samples=q_samples,
        p_integration_weights=p_weights,
        q_integration_weights=q_weights,
    )
    return PreparedGaussianWeightFit(
        objective=objective,
        active_candidates=active_candidates,
        active_candidate_indices=active_indices,
        n_candidates=original_count,
        _objective_fn=objective_fn,
    )


def _build_fit_result(
    prepared: PreparedGaussianWeightFit, solution: SimplexOptimizationResult, /
) -> GaussianWeightFitResult:
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
    weights = np.zeros(prepared.n_candidates, dtype=np.float64)
    weights[list(prepared.active_candidate_indices)] = solution.active_weights
    weights.setflags(write=False)
    combined = combine_distributions(
        weights=solution.active_weights, sources=prepared.active_candidates
    )
    if combined.mapping is None:
        msg = "combined mixture must carry provenance."
        raise AssertionError(msg)
    remapped_sources = np.take(
        np.asarray(prepared.active_candidate_indices, dtype=np.intp), combined.mapping.source_index
    )
    remapped_sources.setflags(write=False)
    mixture = GaussianMixture(
        weights=combined.weights,
        means=combined.means,
        covariances=combined.covariances,
        mapping=MixtureMapping(
            source_index=remapped_sources,
            local_component_index=combined.mapping.local_component_index,
        ),
    )
    return GaussianWeightFitResult(
        weights=weights,
        objective=prepared.objective,
        optimizer=solution.optimizer,
        objective_value=solution.objective_value,
        mixture=mixture,
        n_iterations=solution.n_iterations,
        converged=solution.converged,
        active_candidate_indices=prepared.active_candidate_indices,
        message=solution.message,
    )


def _validate_selection(selection: CandidateSelection, n_candidates: int) -> None:
    selected = selection.selected_indices
    rejected = selection.rejected_indices
    if not selected:
        msg = "candidate_selector must retain at least one candidate."
        raise ValueError(msg)
    combined = selected + rejected
    if any(type(index) is not int for index in combined):
        msg = "CandidateSelection indices must be integers."
        raise TypeError(msg)
    if len(set(combined)) != n_candidates or set(combined) != set(range(n_candidates)):
        msg = "CandidateSelection indices must form a complete non-overlapping candidate partition."
        raise ValueError(msg)
