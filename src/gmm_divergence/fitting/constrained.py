from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import LinearConstraint, minimize

from gmm_divergence.distribution.gmm import GaussianMixture
from gmm_divergence.divergence import kl_divergence
from gmm_divergence.results import KLFitResult
from gmm_divergence.utils import logsumexp, resolve_samples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.typing import FloatArray


def fit_forward_kl_weights_constrained(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    num_samples: int = 10_000,
    rng: np.random.Generator | int | None = None,
    samples: npt.ArrayLike | None = None,
) -> KLFitResult:
    """Fit mixture weights by minimizing forward KL with constrained optimization."""
    samples = resolve_samples(target, num_samples, samples, rng)
    q_component = len(components)
    log_q: FloatArray = np.zeros((num_samples, q_component), dtype=np.float64)

    for i, qi in enumerate(components):
        log_q[:, i] = qi.logpdf(samples)

    def objective(weights: npt.NDArray[np.float64]) -> float:
        log_w: npt.NDArray[np.float64] = np.log(weights + 1e-300)
        log_qw = logsumexp(log_w[None, :] + log_q, axis=1)
        return -np.mean(log_qw - target.logpdf(samples)).astype(np.float64)

    def gradient(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        log_w: npt.NDArray[np.float64] = np.log(weights + 1e-300)
        log_terms = log_w[None, :] + log_q
        log_qw = logsumexp(log_terms, axis=1)
        r = np.exp(log_terms - log_qw[:, None])
        grad = -np.mean(r / (weights[None, :] + 1e-300), axis=0)
        return grad.astype(np.float64)

    constraints = LinearConstraint(A=np.ones((1, q_component), dtype=np.float64), lb=1.0, ub=1.0)
    bounds = [(0, 1) for _ in range(q_component)]
    w0 = np.full(q_component, 1 / q_component, dtype=np.float64)
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )
    fitted_mixture = GaussianMixture.from_distributions(weights=result.x, distributions=components)
    return KLFitResult(
        weights=result.x.astype(np.float64),
        objective=result.fun,
        estimated_kl=kl_divergence(
            target, fitted_mixture, rng=rng, num_samples=num_samples, samples=samples
        ),
        scipy_result=result,
        fitted_mixture=fitted_mixture,
    )
