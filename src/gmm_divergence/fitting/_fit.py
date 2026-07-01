"""Fit mixture weights to minimize KL divergence."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence._core._sampling import (
    Draw,
    SampleSpec,
    Stratified,
    resolve_sample_batches,
    resolve_samples,
)
from gmm_divergence._core._validation import as_weights
from gmm_divergence.distributions._combine import combine_gaussians
from gmm_divergence.divergence import MonteCarlo, kl_divergence
from gmm_divergence.fitting._objectives import build_objective, softmax
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitObjective,
    FitParameterization,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.results import KLFitResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence._core._types import FloatArray, Weights
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._selector import CandidateSelection, CandidateSelector


FitObjectiveConfig: TypeAlias = (
    ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching
)
FitOptimizerConfig: TypeAlias = SoftmaxLBFGSB | SimplexSLSQP


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


def _objective_name(objective: FitObjectiveConfig) -> FitObjective:
    match objective:
        case ForwardKL():
            return "forward"
        case ReverseKL():
            return "reverse"
        case BidirectionalKL():
            return "bidirectional"
        case JensenShannon():
            return "jensen_shannon"
        case MomentMatching():
            return "moment_matching"


def _objective_alpha(objective: FitObjectiveConfig) -> float | None:
    if isinstance(objective, BidirectionalKL):
        return objective.alpha
    return None


def _objective_rng(objective: FitObjectiveConfig) -> np.random.Generator | int | None:
    if isinstance(objective, ForwardKL):
        return _sample_spec_rng(objective.sampling)
    if isinstance(objective, (ReverseKL, BidirectionalKL, JensenShannon)):
        return _sample_spec_rng(objective.p_sampling)
    return None


def _sample_spec_rng(spec: SampleSpec) -> np.random.Generator | int | None:
    if isinstance(spec, (Draw, Stratified)):
        return spec.rng
    return None


def _resolve_objective_samples(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjectiveConfig,
) -> tuple[FloatArray, FloatArray | None]:
    match objective:
        case ForwardKL(sampling=sampling):
            return resolve_samples(p, sampling), None
        case ReverseKL(p_sampling=p_sampling, q_sampling=q_sampling):
            return resolve_samples(p, p_sampling), resolve_sample_batches(q_i, q_sampling)
        case BidirectionalKL(p_sampling=p_sampling, q_sampling=q_sampling):
            return resolve_samples(p, p_sampling), resolve_sample_batches(q_i, q_sampling)
        case JensenShannon(p_sampling=p_sampling, q_sampling=q_sampling):
            return resolve_samples(p, p_sampling), resolve_sample_batches(q_i, q_sampling)
        case MomentMatching():
            return resolve_samples(p, Draw(10_000)), None


def fit_mixture_weights(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjectiveConfig,
    optimizer: FitOptimizerConfig,
    parameterization: FitParameterization,
    x0: npt.ArrayLike | None = None,
    candidate_selection: CandidateSelector[Gaussian | GaussianMixture] | None = None,
) -> KLFitResult:
    selection: CandidateSelection[Gaussian | GaussianMixture] | None = None
    if candidate_selection is not None:
        selection = candidate_selection.select(p, q_i)
        q_i = [q_i[int(i)] for i in selection.selected_indices]
    q_component = _validate_q_i(q_i, p.dim)
    resolved_p_samples, resolved_q_samples = _resolve_objective_samples(p, q_i, objective)
    resolved_num_p_samples = int(resolved_p_samples.shape[0])

    if parameterization == "softmax":
        x0 = (
            np.array(x0, dtype=np.float64)
            if x0 is not None
            else np.zeros(q_component, dtype=np.float64)
        )
        scipy_method = "L-BFGS-B"
        constraints = ()
        bounds = None

        def weights_from_result(values: FloatArray) -> Weights:
            return softmax(values)

    else:
        x0 = (
            as_weights(x0, expected_length=q_component, name="Initial weights")
            if x0 is not None
            else np.full(q_component, 1.0 / q_component, dtype=np.float64)
        )
        scipy_method = "SLSQP"
        constraints = LinearConstraint(
            A=np.ones((1, q_component), dtype=np.float64),
            lb=np.array([1.0], dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
        )
        bounds = Bounds(
            lb=np.zeros(q_component, dtype=np.float64), ub=np.ones(q_component, dtype=np.float64)
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
        x0,
        method=scipy_method,
        jac=True,
        constraints=constraints,
        bounds=bounds,
        tol=optimizer.tol,
        options={"maxiter": optimizer.max_iterations},
    )
    rng = _objective_rng(objective)
    diagnostic_sampling = Draw(resolved_num_p_samples, rng=rng)
    weights: Weights = weights_from_result(result.x)
    fitted_mixture = combine_gaussians(weights=weights, sources=q_i, include_mapping=True)
    return KLFitResult(
        weights=weights,
        fit_objective=_objective_name(objective),
        objective_value=float(result.fun),
        forward_kl=kl_divergence(
            p, fitted_mixture.mixture, method=MonteCarlo(sampling=diagnostic_sampling)
        ),
        reverse_kl=kl_divergence(
            fitted_mixture.mixture, p, method=MonteCarlo(sampling=diagnostic_sampling)
        ),
        scipy_result=result,
        fitted_mixture=fitted_mixture,
        alpha=_objective_alpha(objective),
        iterations=result.nit,
        converged=bool(result.success),
        used_candidate_indices=list(selection.selected_indices) if selection is not None else None,
    )
