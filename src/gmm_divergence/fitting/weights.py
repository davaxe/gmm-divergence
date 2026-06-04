from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence.distribution.combine import combine_gaussians
from gmm_divergence.divergence import kl_divergence
from gmm_divergence.fitting._objective import FitObjective, build_objective, softmax
from gmm_divergence.results import KLFitResult
from gmm_divergence.utils import resolve_samples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.optimize import OptimizeResult

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import FloatArray, Weights


def _validate_q_i(q_i: Sequence[Gaussian | GaussianMixture]) -> int:
    """Validate candidate distributions and return their count."""
    q_component = len(q_i)
    if q_component == 0:
        msg = "q_i must contain at least one distribution."
        raise ValueError(msg)
    if len({q.dim for q in q_i}) != 1:
        msg = "All q_i distributions must have the same dimensionality."
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


def _construct_kl_fit_result_from_weights(
    *,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    weights: Weights,
    fit_objective: FitObjective,
    objective_value: float,
    p_samples: FloatArray,
    reverse_diagnostic_sampling: int,
    alpha: float,
    rng: np.random.Generator | int | None,
    scipy_result: OptimizeResult | None = None,
    iterations: int | None = None,
    converged: bool | None = None,
) -> KLFitResult:
    """Construct a KLFitResult from fitted mixture weights."""
    fitted_mixture = combine_gaussians(weights=weights, sources=q_i, include_mapping=True)
    return KLFitResult(
        weights=fitted_mixture.mixture.weights.astype(np.float64),
        fit_objective=fit_objective,
        objective_value=objective_value,
        forward_kl=kl_divergence(p, fitted_mixture.mixture, rng=rng, sampling=p_samples),
        reverse_kl=kl_divergence(
            fitted_mixture.mixture, p, rng=rng, sampling=reverse_diagnostic_sampling
        ),
        scipy_result=scipy_result,
        fitted_mixture=fitted_mixture,
        alpha=alpha if fit_objective == "bidirectional" else None,
        iterations=iterations,
        converged=converged,
    )


def fit_mixture_weights_softmax(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    rng: np.random.Generator | int | None = None,
    p_sampling: npt.ArrayLike | int = 10_000,
    q_sampling: npt.ArrayLike | int = 10_000,
    objective: FitObjective = "forward",
    alpha: float = 0.5,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    r"""Fit mixture weights by minimizing forward KL using softmax logits.

    Fits nonnegative mixture weights for `q_i` by minimizing a Monte Carlo
    estimate of

    $$
    D_{\mathrm{KL}}(p \| q_w),
    $$

    where `p` is the reference distribution and `q_w` is the weighted mixture of
    the provided `q_i`. The weights are parameterized by unconstrained
    logits and optimized with L-BFGS-B.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution to approximate.
    q_i : sequence of Gaussian or GaussianMixture
        Candidate distributions whose weights are fitted.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed used when sampling is required.
    p_sampling : int or array-like, default=10_000
        Number of samples drawn from `p`, or precomputed samples from `p`.
    q_sampling : int or array-like, default=10_000
        Number of samples drawn from each `q_i`, or precomputed samples with
        shape `(len(q_i), n_samples, n_features)`.
    objective : {"forward", "reverse", "bidirectional"}, default="forward"
        KL objective used to fit the mixture weights.
    alpha : float, default=0.5
        Relative weighting used by the bidirectional objective.
    tol : float, default=1e-8
        Optimization tolerance.
    max_iterations : int, default=1000
        Maximum number of optimizer iterations.

    Returns
    -------
    KLFitResult
        Result object containing the fitted weights, fitted mixture, fit
        objective, optimizer objective value, forward/reverse KL diagnostics,
        and optimizer metadata.
    """
    resolved_p_samples = resolve_samples(p, p_sampling, rng)
    resolved_num_p_samples, _dim = resolved_p_samples.shape
    q_component = _validate_q_i(q_i)
    resolved_q_samples = None
    if objective in {"reverse", "bidirectional"}:
        resolved_q_samples = _resolve_q_samples(q_i, q_sampling, rng)
    theta0 = np.zeros(q_component, dtype=np.float64)
    result = minimize(
        build_objective(
            parameterization="softmax",
            objective=objective,
            p=p,
            q_i=q_i,
            p_samples=resolved_p_samples,
            q_samples=resolved_q_samples,
            alpha=alpha,
        ),
        theta0,
        method="L-BFGS-B",
        jac=True,
        tol=tol,
        options={"maxiter": max_iterations},
    )
    return _construct_kl_fit_result_from_weights(
        p=p,
        q_i=q_i,
        weights=softmax(result.x),
        fit_objective=objective,
        objective_value=float(result.fun),
        p_samples=resolved_p_samples,
        reverse_diagnostic_sampling=resolved_num_p_samples,
        alpha=alpha,
        rng=rng,
        scipy_result=result,
        iterations=result.nit,
        converged=result.success,
    )


def fit_mixture_weights_simplex(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    *,
    rng: np.random.Generator | int | None = None,
    p_sampling: npt.ArrayLike | int = 10_000,
    q_sampling: npt.ArrayLike | int = 10_000,
    objective: FitObjective = "forward",
    alpha: float = 0.5,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    r"""Fit mixture weights by minimizing forward KL on the probability simplex.

    Fits mixture weights for `q_i` by minimizing a Monte Carlo estimate of

    $$
    D_{\mathrm{KL}}(p \| q_w),
    $$

    where `p` is the reference distribution and `q_w` is the weighted mixture of
    the provided `q_i`. The weights are optimized directly under simplex
    constraints: nonnegative weights summing to one.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution to approximate.
    q_i : sequence of Gaussian or GaussianMixture
        Candidate distributions whose weights are fitted.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed used when sampling is required.
    p_sampling : int or array-like, default=10_000
        Number of samples drawn from `p`, or precomputed samples from `p`.
    q_sampling : int or array-like, default=10_000
        Number of samples drawn from each `q_i`, or precomputed samples with
        shape `(len(q_i), n_samples, n_features)`.
    objective : {"forward", "reverse", "bidirectional"}, default="forward"
        KL objective used to fit the mixture weights.
    alpha : float, default=0.5
        Relative weighting used by the bidirectional objective.
    tol : float, default=1e-8
        Optimization tolerance.
    max_iterations : int, default=1000
        Maximum number of optimizer iterations.

    Returns
    -------
    KLFitResult
        Result object containing the fitted weights, fitted mixture, fit
        objective, optimizer objective value, forward/reverse KL diagnostics,
        and optimizer metadata.
    """
    resolved_p_samples = resolve_samples(p, p_sampling, rng)
    resolved_num_p_samples, _dim = resolved_p_samples.shape
    q_component = _validate_q_i(q_i)
    resolved_q_samples = None
    if objective in {"reverse", "bidirectional"}:
        resolved_q_samples = _resolve_q_samples(q_i, q_sampling, rng)
    w0 = np.full(q_component, 1 / q_component, dtype=np.float64)
    constraint = LinearConstraint(
        A=np.ones((1, q_component), dtype=np.float64),
        lb=np.array([1.0], dtype=np.float64),
        ub=np.array([1.0], dtype=np.float64),
    )
    bounds = Bounds(
        lb=np.zeros(q_component, dtype=np.float64), ub=np.ones(q_component, dtype=np.float64)
    )
    result = minimize(
        build_objective(
            parameterization="simplex",
            objective=objective,
            p=p,
            q_i=q_i,
            p_samples=resolved_p_samples,
            q_samples=resolved_q_samples,
            alpha=alpha,
        ),
        w0,
        method="SLSQP",
        jac=True,
        constraints=constraint,
        bounds=bounds,
        tol=tol,
        options={"maxiter": max_iterations},
    )
    return _construct_kl_fit_result_from_weights(
        p=p,
        q_i=q_i,
        weights=result.x.astype(np.float64),
        fit_objective=objective,
        objective_value=float(result.fun),
        p_samples=resolved_p_samples,
        reverse_diagnostic_sampling=resolved_num_p_samples,
        alpha=alpha,
        rng=rng,
        scipy_result=result,
        iterations=result.nit,
        converged=result.success,
    )
