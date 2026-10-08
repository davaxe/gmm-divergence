from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence.fitting._simplex import SimplexOptimizationResult, optimize_simplex

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.fitting._options import FitMethod


def _objective(weights: FloatArray) -> tuple[float, FloatArray]:
    residual = weights - np.array([0.2, 0.3, 0.5])
    return float(np.dot(residual, residual)), 2 * residual


@pytest.mark.parametrize("method", [gd.fitting.SoftmaxLBFGSB(), gd.fitting.SimplexSLSQP()])
def test_solver_accepts_objectives_without_prepared_fits(method: FitMethod) -> None:
    solution = optimize_simplex(_objective, n_weights=3, method=method)
    assert gd.fitting.SimplexOptimizationResult is SimplexOptimizationResult
    assert solution.converged
    assert solution.method is method
    np.testing.assert_allclose(solution.active_weights, [0.2, 0.3, 0.5], atol=1e-4)
    assert solution.objective_value == pytest.approx(0, abs=1e-8)
    assert solution.iterations > 0
    assert solution.optimizer_message
    assert not solution.parameters.flags.writeable
    assert not solution.active_weights.flags.writeable
    warm_method = (
        gd.fitting.SoftmaxLBFGSB(initial_logits=solution.parameters)
        if isinstance(method, gd.fitting.SoftmaxLBFGSB)
        else gd.fitting.SimplexSLSQP(initial_weights=solution.active_weights)
    )
    warm_solution = optimize_simplex(_objective, n_weights=3, method=warm_method)
    np.testing.assert_allclose(warm_solution.active_weights, solution.active_weights, atol=1e-4)
    assert warm_solution.converged


def test_slsqp_enforces_minimum_weights() -> None:
    def objective(weights: FloatArray) -> tuple[float, FloatArray]:
        residual = weights - np.array([0, 1])
        return float(np.dot(residual, residual)), 2 * residual

    solution = optimize_simplex(
        objective, n_weights=2, method=gd.fitting.SimplexSLSQP(min_weight=0.1)
    )
    np.testing.assert_allclose(solution.active_weights, [0.1, 0.9], atol=1e-8)
    assert solution.converged


def test_iteration_limit_returns_termination_metadata() -> None:
    solution = optimize_simplex(
        _objective, n_weights=3, method=gd.fitting.SimplexSLSQP(max_iterations=1)
    )
    assert not solution.converged
    assert solution.iterations == 1
    assert "Iteration limit" in solution.optimizer_message
    assert np.isfinite(solution.objective_value)


@pytest.mark.parametrize("n_weights", [0, -1])
def test_solver_rejects_nonpositive_weight_count(n_weights: int) -> None:
    with pytest.raises(ValueError, match="n_weights must be a positive integer"):
        _ = optimize_simplex(_objective, n_weights=n_weights, method=gd.fitting.SimplexSLSQP())
