"""Categorical-mixture weight fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_weights
from gmm_divergence.fitting._simplex import ObjectiveFn, optimize_simplex
from gmm_divergence.results import CategoricalMixtureFitResult

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.fitting._options import FitMethod


def fit_categorical_mixture_weights(
    p_w: npt.ArrayLike, q_i_w: npt.ArrayLike, /, *, method: FitMethod
) -> CategoricalMixtureFitResult:
    """Fit a convex combination of categorical distributions using forward KL."""
    p_w, q_i_w = _validate_categorical_fit_inputs(p_w, q_i_w)

    objective = _categorical_kl_objective(p_w, q_i_w)

    solution = optimize_simplex(objective, n_weights=len(q_i_w), method=method)

    fitted_probabilities = solution.active_weights @ q_i_w

    return CategoricalMixtureFitResult(
        weights=solution.active_weights,
        fitted_probabilities=fitted_probabilities,
        objective_value=solution.objective_value,
        fit_method=solution.method,
        iterations=solution.iterations,
        converged=solution.converged,
        optimizer_message=solution.optimizer_message,
    )


def _validate_categorical_fit_inputs(
    p_w: npt.ArrayLike, q_i_w: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    p_w = as_weights(p_w, name="p_w", writable=False, normalize=False)
    if np.abs(p_w.sum() - 1.0) > 1e-12:
        msg = "p_w must sum to one."
        raise ValueError(msg)

    q_i_w = np.asarray(q_i_w, dtype=np.float64)

    if q_i_w.ndim != 2:
        msg = f"q_i_w must be a 2D array, got {q_i_w.ndim}D."
        raise ValueError(msg)

    if q_i_w.shape[1] != p_w.shape[0]:
        msg = (
            "q_i_w must have the same number of columns as p_w, got"
            f" {q_i_w.shape[1]} vs {p_w.shape[0]}."
        )
        raise ValueError(msg)

    if q_i_w.shape[0] == 0:
        msg_0 = "q_i_w must contain at least one candidate."
        raise ValueError(msg_0)

    if not np.all(np.isfinite(q_i_w)):
        msg = "q_i_w must contain only finite values."
        raise ValueError(msg)

    if np.any(q_i_w < 0.0):
        msg = "q_i_w must be nonnegative."
        raise ValueError(msg)

    if not np.allclose(q_i_w.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        msg = "Each row of q_i_w must sum to one."
        raise ValueError(msg)

    return p_w, q_i_w


def _categorical_kl_objective(p_w: FloatArray, q_i_w: FloatArray) -> ObjectiveFn:
    """Construct an exact forward-KL objective and gradient."""
    support = p_w > 0.0

    p = p_w[support]
    q = q_i_w[:, support]

    if np.any(q.sum(axis=0) == 0.0):  # ruff: ignore[float-equality-comparison]
        msg = "Some target categories have no support in any candidate."
        raise ValueError(msg)

    constant = float(np.dot(p, np.log(p)))

    def objective(weights: FloatArray) -> tuple[float, FloatArray]:
        fitted = weights @ q
        if np.any(fitted <= 0.0):
            return float("inf"), np.full_like(weights, np.nan)

        value = constant - np.dot(p, np.log(fitted))
        gradient = -(q @ (p / fitted))
        return float(value), gradient

    return objective
