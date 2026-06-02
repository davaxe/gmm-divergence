from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence.distribution.combine import combine_gaussians
from gmm_divergence.divergence import kl_divergence
from gmm_divergence.results import KLFitResult
from gmm_divergence.utils import logsumexp, resolve_samples

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from scipy.optimize import OptimizeResult

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import FloatArray


def _softmax(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable softmax."""
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float64)


def _validate_components(
    components: Sequence[Gaussian | GaussianMixture],
) -> int:
    """Validate components and return their count."""
    q_component = len(components)
    if q_component == 0:
        msg = "components must contain at least one component."
        raise ValueError(msg)
    return q_component


def _component_logpdf_matrix(
    components: Sequence[Gaussian | GaussianMixture],
    samples: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate each component log-density at each sample."""
    log_q = np.empty((samples.shape[0], len(components)), dtype=np.float64)

    for i, qi in enumerate(components):
        log_q[:, i] = qi.logpdf(samples)

    return log_q


def _softmax_objective_and_jacobian_fn(
    components: Sequence[Gaussian | GaussianMixture],
    samples: npt.NDArray[np.float64],
) -> Callable[[npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64]]]:
    """Return objective function and Jacobian for optimizing mixture logits."""
    log_q: FloatArray = _component_logpdf_matrix(components, samples)

    def objective_and_jacobian(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        weights = _softmax(theta)
        log_w = np.log(weights)
        log_terms = log_w[None, :] + log_q
        log_qw = logsumexp(log_terms, axis=1)
        objective = -float(np.mean(log_qw))
        r = np.exp(log_terms - log_qw[:, None])
        gradient = weights - np.mean(r, axis=0).astype(np.float64)

        return objective, gradient.astype(np.float64)

    return objective_and_jacobian


def _simplex_objective_and_jacobian_fn(
    components: Sequence[Gaussian | GaussianMixture],
    samples: npt.NDArray[np.float64],
) -> Callable[[npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64]]]:
    """Return objective function and Jacobian for optimizing simplex weights."""
    log_q: FloatArray = _component_logpdf_matrix(components, samples)

    def objective_and_jacobian(
        w: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        w_safe = np.maximum(w, 1e-300)
        log_w = np.log(w_safe)
        log_terms = log_w[None, :] + log_q
        log_qw = logsumexp(log_terms, axis=1)
        objective = -float(np.mean(log_qw))
        r = np.exp(log_terms - log_qw[:, None])
        gradient = -np.mean(r / w_safe[None, :], axis=0).astype(np.float64)

        return objective, gradient.astype(np.float64)

    return objective_and_jacobian


def _construct_kl_fit_result_from_weights(
    *,
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    weights: npt.NDArray[np.float64],
    objective: float,
    num_samples: int,
    samples: npt.NDArray[np.float64],
    rng: np.random.Generator | int | None,
    scipy_result: OptimizeResult | None = None,
    iterations: int | None = None,
    converged: bool | None = None,
) -> KLFitResult:
    """Construct a KLFitResult from fitted mixture weights."""
    fitted_mixture = combine_gaussians(
        weights=weights,
        sources=components,
        include_mapping=True,
    )
    return KLFitResult(
        weights=fitted_mixture.mixture.weights.astype(np.float64),
        objective=objective,
        estimated_kl=kl_divergence(
            target,
            fitted_mixture.mixture,
            rng=rng,
            num_samples=num_samples,
            samples=samples,
        ),
        scipy_result=scipy_result,
        fitted_mixture=fitted_mixture,
        iterations=iterations,
        converged=converged,
    )


def fit_mixture_weights_softmax(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    num_samples: int = 10_000,
    rng: np.random.Generator | int | None = None,
    samples: npt.ArrayLike | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    """Fit mixture weights by minimizing forward KL with unconstrained logits."""
    samples_arr = resolve_samples(target, num_samples, samples, rng)
    resolved_num_samples, _dim = samples_arr.shape
    q_component = _validate_components(components)
    theta0 = np.zeros(q_component, dtype=np.float64)
    result = minimize(
        _softmax_objective_and_jacobian_fn(components, samples_arr),
        theta0,
        method="L-BFGS-B",
        jac=True,
        tol=tol,
        options={
            "maxiter": max_iterations,
        },
    )
    return _construct_kl_fit_result_from_weights(
        target=target,
        components=components,
        weights=_softmax(result.x),
        objective=float(result.fun),
        num_samples=resolved_num_samples,
        samples=samples_arr,
        rng=rng,
        scipy_result=result,
        iterations=result.nit,
        converged=result.success,
    )


def fit_mixture_weights_simplex(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    num_samples: int = 10_000,
    rng: np.random.Generator | int | None = None,
    samples: npt.ArrayLike | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    """Fit mixture weights by minimizing forward KL with constrained weights."""
    samples_arr = resolve_samples(target, num_samples, samples, rng)
    resolved_num_samples, _dim = samples_arr.shape

    q_component = _validate_components(components)
    w0 = np.full(q_component, 1 / q_component, dtype=np.float64)
    constraint = LinearConstraint(
        A=np.ones((1, q_component), dtype=np.float64),
        lb=np.array([1.0], dtype=np.float64),
        ub=np.array([1.0], dtype=np.float64),
    )
    bounds = Bounds(
        lb=np.zeros(q_component, dtype=np.float64),
        ub=np.ones(q_component, dtype=np.float64),
    )
    result = minimize(
        _simplex_objective_and_jacobian_fn(components, samples_arr),
        w0,
        method="SLSQP",
        jac=True,
        constraints=constraint,
        bounds=bounds,
        tol=tol,
        options={
            "maxiter": max_iterations,
        },
    )
    return _construct_kl_fit_result_from_weights(
        target=target,
        components=components,
        weights=result.x.astype(np.float64),
        objective=float(result.fun),
        num_samples=resolved_num_samples,
        samples=samples_arr,
        rng=rng,
        scipy_result=result,
        iterations=result.nit,
        converged=result.success,
    )


def fit_mixture_weights_em(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    num_samples: int = 10_000,
    rng: np.random.Generator | int | None = None,
    samples: npt.ArrayLike | None = None,
    max_iterations: int = 1000,
    tol: float = 1e-8,
) -> KLFitResult:
    """Fit mixture weights by minimizing forward KL with EM algorithm."""
    samples_arr = resolve_samples(target, num_samples, samples, rng)
    num_samples, _dim = samples_arr.shape
    q_component = _validate_components(components)
    log_q = _component_logpdf_matrix(components, samples_arr)
    log_w = np.full(q_component, -np.log(q_component), dtype=np.float64)
    objective = float("inf")
    it: int = 0
    weights = np.exp(log_w).astype(np.float64)
    for i in range(1, max_iterations + 1):
        log_terms = log_w[None, :] + log_q
        log_qw = logsumexp(log_terms, axis=1)
        objective = -float(np.mean(log_qw))
        log_r = log_terms - log_qw[:, None]
        new_log_w = logsumexp(log_r, axis=0) - np.log(num_samples)
        new_log_w -= logsumexp(new_log_w)
        new_weights = np.exp(new_log_w)
        delta = np.linalg.norm(new_weights - weights, ord=1)

        log_w = new_log_w
        weights = new_weights
        it = i
        if delta < tol:
            break

    weights = np.exp(log_w).astype(np.float64)
    weights[~np.isfinite(weights)] = 0.0

    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        msg = "EM weight fitting produced invalid weights."
        raise RuntimeError(msg)

    weights /= weight_sum
    return _construct_kl_fit_result_from_weights(
        target=target,
        components=components,
        weights=weights,
        objective=objective,
        num_samples=num_samples,
        samples=samples_arr,
        rng=rng,
        scipy_result=None,
        iterations=it,
        converged=it < max_iterations - 1,
    )
