"""Categorical-mixture weight fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_probabilities, as_probability_rows
from gmm_divergence.fitting._simplex import ObjectiveFn, optimize_simplex
from gmm_divergence.results import CategoricalWeightFitResult

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.fitting._options import Optimizer


def fit_categorical_weights(
    target: npt.ArrayLike, candidates: npt.ArrayLike, /, *, optimizer: Optimizer
) -> CategoricalWeightFitResult:
    """Fit a convex combination of categorical distributions using exact forward KL.

    Parameters
    ----------
    target : array-like, shape (n_categories,)
        Finite, nonnegative probabilities summing to one.
    candidates : array-like, shape (n_candidates, n_categories)
        Nonempty candidate probability rows with the same categories as target.
        Each positive target category must have support in at least one row.
    optimizer : Optimizer
        Explicit simplex optimizer configuration.

    Returns
    -------
    CategoricalWeightFitResult
        Read-only candidate weights and fitted category probabilities, the exact
        forward KL value in nats, and optimizer termination diagnostics.
    """
    target, candidates = _validate_categorical_fit_inputs(target, candidates)

    objective = _categorical_kl_objective(target, candidates)

    solution = optimize_simplex(objective, n_weights=len(candidates), optimizer=optimizer)

    probabilities = solution.active_weights @ candidates

    return CategoricalWeightFitResult(
        weights=solution.active_weights,
        probabilities=probabilities,
        objective_value=solution.objective_value,
        optimizer=solution.optimizer,
        n_iterations=solution.n_iterations,
        converged=solution.converged,
        message=solution.message,
    )


def _validate_categorical_fit_inputs(
    target: npt.ArrayLike, candidates: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    target = as_probabilities(target, name="p_w")
    candidates = as_probability_rows(candidates, name="q_i_w")
    if candidates.shape[1] != target.shape[0]:
        msg = (
            "q_i_w must have the same number of columns as p_w, got"
            f" {candidates.shape[1]} vs {target.shape[0]}."
        )
        raise ValueError(msg)

    return target, candidates


def _categorical_kl_objective(target: FloatArray, candidates: FloatArray) -> ObjectiveFn:
    """Construct an exact forward-KL objective and gradient."""
    support = target > 0.0

    target = target[support]
    q = candidates[:, support]

    if np.any(q.sum(axis=0) == 0.0):  # ruff: ignore[float-equality-comparison]
        msg = "Some target categories have no support in any candidate."
        raise ValueError(msg)

    constant = float(np.dot(target, np.log(target)))

    def objective(weights: FloatArray) -> tuple[float, FloatArray]:
        fitted = weights @ q
        if np.any(fitted <= 0.0):
            return float("inf"), np.full_like(weights, np.nan)

        value = constant - np.dot(target, np.log(fitted))
        gradient = -(q @ (target / fitted))
        return float(value), gradient

    return objective
