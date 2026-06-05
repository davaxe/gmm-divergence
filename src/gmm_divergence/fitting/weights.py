from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence.distribution.combine import combine_gaussians
from gmm_divergence.divergence import MonteCarlo, kl_divergence
from gmm_divergence.fitting._objective import build_objective, softmax
from gmm_divergence.fitting.config import (
    BidirectionalKL,
    FitObjective,
    FitParameterization,
    ForwardKL,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.results import KLFitResult
from gmm_divergence.utils import resolve_samples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.optimize import OptimizeResult

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import FloatArray, Weights


FitObjectiveConfig: TypeAlias = ForwardKL | ReverseKL | BidirectionalKL | MomentMatching
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


def _resolve_q_samples(
    q_i: Sequence[Gaussian | GaussianMixture],
    q_sampling: npt.ArrayLike | int | None,
    rng: np.random.Generator | int | None,
) -> FloatArray:
    """Return provided q samples or draw samples from each component."""
    if isinstance(q_sampling, int) or q_sampling is None:
        n_samples = q_sampling or 10_000
        rng = np.random.default_rng(rng)
        return np.asarray([q.sample(n_samples, rng=rng) for q in q_i], dtype=np.float64)

    q_sampling = np.asarray(q_sampling, dtype=np.float64)
    if q_sampling.ndim != 3 or q_sampling.shape[0] != len(q_i):
        msg = (
            "Expected q_sampling with shape "
            f"({len(q_i)}, n_samples, n_features), got {q_sampling.shape}."
        )
        raise ValueError(msg)
    return q_sampling


def _objective_name(objective: FitObjectiveConfig) -> FitObjective:
    match objective:
        case ForwardKL():
            return "forward"
        case ReverseKL():
            return "reverse"
        case BidirectionalKL():
            return "bidirectional"
        case MomentMatching():
            return "moment_matching"


def _objective_alpha(objective: FitObjectiveConfig) -> float | None:
    if isinstance(objective, BidirectionalKL):
        return objective.alpha
    return None


def _objective_rng(objective: FitObjectiveConfig) -> np.random.Generator | int | None:
    if isinstance(objective, (ForwardKL, ReverseKL, BidirectionalKL)):
        return objective.rng
    return None


def _resolve_objective_samples(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjectiveConfig,
) -> tuple[FloatArray, FloatArray | None]:
    match objective:
        case ForwardKL(sampling=sampling, rng=rng):
            return resolve_samples(p, sampling, rng), None
        case ReverseKL(p_sampling=p_sampling, q_sampling=q_sampling, rng=rng):
            return resolve_samples(p, p_sampling, rng), _resolve_q_samples(q_i, q_sampling, rng)
        case BidirectionalKL(p_sampling=p_sampling, q_sampling=q_sampling, rng=rng):
            return resolve_samples(p, p_sampling, rng), _resolve_q_samples(q_i, q_sampling, rng)
        case MomentMatching():
            return resolve_samples(p, 10_000, None), None


def _construct_kl_fit_result_from_weights(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    weights: Weights,
    objective: FitObjectiveConfig,
    objective_value: float,
    p_samples: FloatArray,
    reverse_diagnostic_sampling: int,
    scipy_result: OptimizeResult | None = None,
    iterations: int | None = None,
    converged: bool | None = None,
) -> KLFitResult:
    """Construct a KLFitResult from fitted mixture weights."""
    rng = _objective_rng(objective)
    fitted_mixture = combine_gaussians(weights=weights, sources=q_i, include_mapping=True)
    return KLFitResult(
        weights=fitted_mixture.mixture.weights.astype(np.float64),
        fit_objective=_objective_name(objective),
        objective_value=objective_value,
        forward_kl=kl_divergence(
            p, fitted_mixture.mixture, method=MonteCarlo(sampling=p_samples, rng=rng)
        ),
        reverse_kl=kl_divergence(
            fitted_mixture.mixture,
            p,
            method=MonteCarlo(sampling=reverse_diagnostic_sampling, rng=rng),
        ),
        scipy_result=scipy_result,
        fitted_mixture=fitted_mixture,
        alpha=_objective_alpha(objective),
        iterations=iterations,
        converged=converged,
    )


def fit_mixture_weights_softmax(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    *,
    objective: FitObjectiveConfig,
    optimizer: SoftmaxLBFGSB,
) -> KLFitResult:
    r"""Fit mixture weights using softmax logits and L-BFGS-B."""
    return _fit_mixture_weights(
        p=p, q_i=q_i, objective=objective, optimizer=optimizer, parameterization="softmax"
    )


def fit_mixture_weights_simplex(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    *,
    objective: FitObjectiveConfig,
    optimizer: SimplexSLSQP,
) -> KLFitResult:
    r"""Fit mixture weights directly on the simplex using SLSQP."""
    return _fit_mixture_weights(
        p=p, q_i=q_i, objective=objective, optimizer=optimizer, parameterization="simplex"
    )


def _fit_mixture_weights(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    objective: FitObjectiveConfig,
    optimizer: FitOptimizerConfig,
    parameterization: FitParameterization,
) -> KLFitResult:
    q_component = _validate_q_i(q_i, p.dim)
    resolved_p_samples, resolved_q_samples = _resolve_objective_samples(p, q_i, objective)
    resolved_num_p_samples = int(resolved_p_samples.shape[0])

    if parameterization == "softmax":
        x0 = np.zeros(q_component, dtype=np.float64)
        scipy_method = "L-BFGS-B"
        constraints = ()
        bounds = None

        def weights_from_result(values: FloatArray) -> Weights:
            return softmax(values)

    else:
        x0 = np.full(q_component, 1 / q_component, dtype=np.float64)
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
    return _construct_kl_fit_result_from_weights(
        p=p,
        q_i=q_i,
        weights=weights_from_result(result.x),
        objective=objective,
        objective_value=float(result.fun),
        p_samples=resolved_p_samples,
        reverse_diagnostic_sampling=resolved_num_p_samples,
        scipy_result=result,
        iterations=result.nit,
        converged=result.success,
    )
