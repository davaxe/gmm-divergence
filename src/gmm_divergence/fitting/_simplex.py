"""Shared scipy optimization for simplex-weight objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence._core._types import FloatArray
from gmm_divergence._core._validation import as_weights, validate_positive_int
from gmm_divergence.fitting._options import validate_optimizer

if TYPE_CHECKING:
    from gmm_divergence._core._types import Weights
    from gmm_divergence.fitting._options import Optimizer


ObjectiveFn = Callable[[FloatArray], tuple[float, FloatArray]]


@dataclass(frozen=True, slots=True)
class SimplexOptimizationResult:
    """Result of optimizing an objective over the probability simplex.

    Stores the optimized mixture weights and solver termination metadata.
    The weights are validated and stored as a read-only float64 array.
    """

    optimizer: Optimizer
    """Optimizer configuration used to obtain the solution."""
    active_weights: Weights
    """Optimized simplex weights for the active candidates."""
    objective_value: float
    """Final scalar objective value."""
    n_iterations: int
    """Number of optimizer iterations."""
    converged: bool
    """Whether the optimizer reported successful termination."""
    message: str
    """Optimizer termination message."""

    def __post_init__(self) -> None:
        weights = as_weights(self.active_weights, name="active_weights", normalize=False)
        weight_sum = float(np.sum(weights))
        if not np.isclose(weight_sum, 1.0, rtol=1e-9, atol=1e-12):
            msg = f"active_weights must sum to one, got {weight_sum}."
            raise ValueError(msg)

        if not np.isfinite(self.objective_value):
            msg = f"objective_value must be finite, got {self.objective_value}."
            raise ValueError(msg)

        if type(self.n_iterations) is not int or self.n_iterations < 0:
            msg = f"n_iterations must be a nonnegative integer, got {self.n_iterations}."
            raise ValueError(msg)

        object.__setattr__(self, "active_weights", weights)


def optimize_simplex(
    objective: ObjectiveFn, /, *, n_weights: int, optimizer: Optimizer
) -> SimplexOptimizationResult:
    """Optimize an objective and gradient over simplex weights using scipy.

    The callable receives a weight vector and returns its scalar objective
    value and gradient. No distributions or prepared-fit data are required.

    Parameters
    ----------
    objective : callable
        Receives a vector of n_weights simplex weights and returns the scalar
        objective and its gradient with respect to those weights.
    n_weights : int
        Positive number of retained candidates represented by the objective.
    optimizer : Optimizer
        SimplexSLSQP configuration with optional initial weights
        matching n_weights.

    Returns
    -------
    SimplexOptimizationResult
        Internal result with read-only optimized weights,
        objective value, and optimizer termination metadata. Iteration-limit
        termination can return converged=False rather than raising an exception.
    """
    _ = validate_positive_int(n_weights, name="n_weights")
    _ = validate_optimizer(optimizer)
    if optimizer.min_weight * n_weights > 1.0:
        msg = "min_weight is infeasible for the number of weights."
        raise ValueError(msg)
    initial = (
        as_weights(optimizer.initial_weights, expected_length=n_weights, name="initial_weights")
        if optimizer.initial_weights is not None
        else np.full(n_weights, 1.0 / n_weights, dtype=np.float64)
    )
    constraints = LinearConstraint(
        A=np.ones((1, n_weights), dtype=np.float64),
        lb=np.array([1.0], dtype=np.float64),
        ub=np.array([1.0], dtype=np.float64),
    )
    bounds = Bounds(
        lb=np.full(n_weights, optimizer.min_weight, dtype=np.float64),
        ub=np.ones(n_weights, dtype=np.float64),
    )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        constraints=constraints,
        bounds=bounds,
        tol=optimizer.tol,
        options={"maxiter": optimizer.max_iterations},
    )
    return SimplexOptimizationResult(
        optimizer=optimizer,
        active_weights=np.asarray(result.x, dtype=np.float64),
        objective_value=float(result.fun),
        n_iterations=int(result.nit),
        converged=bool(result.success),
        message=str(result.message),
    )
