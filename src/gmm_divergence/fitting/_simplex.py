"""Shared scipy optimization for simplex-weight objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._types import FloatArray
from gmm_divergence._core._validation import as_weights, validate_positive_int
from gmm_divergence.fitting._options import SoftmaxLBFGSB, validate_optimizer

if TYPE_CHECKING:
    from gmm_divergence._core._types import Weights
    from gmm_divergence.fitting._options import Optimizer


ObjectiveFn = Callable[[FloatArray], tuple[float, FloatArray]]


def softmax(theta: FloatArray) -> Weights:
    """Map a finite, nonempty logit vector theta to normalized float64 weights.

    Subtracts the maximum logit for stability. Input validation is the caller's
    responsibility; returns a newly allocated vector with the same shape.
    """
    theta = np.asarray(theta, dtype=np.float64)
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float64)


def with_softmax(simplex_objective: ObjectiveFn) -> ObjectiveFn:
    """Wrap simplex_objective as a callable over unconstrained softmax logits.

    The input callable returns a scalar value and weight gradient. The returned
    callable maps logits to weights and applies the chain rule to produce the
    scalar value and gradient with respect to the logits.
    """

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        weights = softmax(theta)
        value, grad_w = simplex_objective(weights)
        grad_w = np.asarray(grad_w, dtype=np.float64)
        grad_theta = weights * (grad_w - np.dot(weights, grad_w))
        return float(value), grad_theta.astype(np.float64)

    return objective


@dataclass(frozen=True, slots=True)
class SimplexOptimizationResult:
    """Optimizer output for a simplex-weight objective.

    `parameters` contains the optimizer coordinates: logits for `SoftmaxLBFGSB`
    and simplex weights for `SimplexSLSQP`. `active_weights` always contains the
    corresponding candidate weights. Both arrays are independent and read-only,
    making them safe to reuse as warm-start inputs.
    """

    optimizer: Optimizer
    """Optimizer configuration used to obtain the solution."""
    parameters: FloatArray
    """Final optimizer coordinates for the active candidates."""
    active_weights: Weights
    """Final simplex weights for the active candidates."""
    objective_value: float
    """Final scalar objective value."""
    n_iterations: int
    """Number of optimizer iterations."""
    converged: bool
    """Whether the optimizer reported convergence."""
    message: str
    """Optimizer termination message."""

    def __post_init__(self) -> None:
        parameters = readonly_copy(self.parameters, dtype=np.float64)
        if parameters.ndim != 1 or parameters.size == 0 or not np.isfinite(parameters).all():
            msg = "parameters must be a nonempty finite 1D array."
            raise ValueError(msg)
        active_weights = as_weights(
            self.active_weights,
            expected_length=parameters.shape[0],
            name="active_weights",
            normalize=False,
            writable=True,
        )
        weight_sum = float(np.sum(active_weights))
        if not np.isclose(weight_sum, 1.0, rtol=1e-9, atol=1e-12):
            msg = f"active_weights must sum to one, got {weight_sum}."
            raise ValueError(msg)
        normalized_weights: Weights = np.asarray(active_weights / weight_sum, dtype=np.float64)
        normalized_weights.setflags(write=False)
        if not np.isfinite(self.objective_value):
            msg = f"objective_value must be finite, got {self.objective_value}."
            raise ValueError(msg)
        if type(self.n_iterations) is not int or self.n_iterations < 0:
            msg = f"n_iterations must be a nonnegative integer, got {self.n_iterations}."
            raise ValueError(msg)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "active_weights", normalized_weights)


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
        SoftmaxLBFGSB or SimplexSLSQP configuration with optional initial values
        matching n_weights.

    Returns
    -------
    SimplexOptimizationResult
        Internal result with final coordinates, read-only normalized weights,
        objective value, and optimizer termination metadata. Iteration-limit
        termination can return converged=False rather than raising an exception.
    """
    _ = validate_positive_int(n_weights, name="n_weights")
    _ = validate_optimizer(optimizer)
    if isinstance(optimizer, SoftmaxLBFGSB):
        initial = (
            np.array(optimizer.initial_logits, dtype=np.float64)
            if optimizer.initial_logits is not None
            else np.zeros(n_weights, dtype=np.float64)
        )
        _validate_initial_vector(initial, n_weights, name="initial_logits")
        scipy_objective = with_softmax(objective)
        scipy_method = "L-BFGS-B"
        constraints = ()
        bounds = None

        def weights_from_parameters(values: FloatArray) -> Weights:
            return softmax(values)

    else:
        if optimizer.min_weight * n_weights > 1.0:
            msg = "min_weight is infeasible for the number of weights."
            raise ValueError(msg)
        initial = (
            as_weights(optimizer.initial_weights, expected_length=n_weights, name="initial_weights")
            if optimizer.initial_weights is not None
            else np.full(n_weights, 1.0 / n_weights, dtype=np.float64)
        )
        scipy_objective = objective
        scipy_method = "SLSQP"
        constraints = LinearConstraint(
            A=np.ones((1, n_weights), dtype=np.float64),
            lb=np.array([1.0], dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
        )
        bounds = Bounds(
            lb=np.full(n_weights, optimizer.min_weight, dtype=np.float64),
            ub=np.ones(n_weights, dtype=np.float64),
        )

        def weights_from_parameters(values: FloatArray) -> Weights:
            return values.astype(np.float64)

    result = minimize(
        scipy_objective,
        initial,
        method=scipy_method,
        jac=True,
        constraints=constraints,
        bounds=bounds,
        tol=optimizer.tol,
        options={"maxiter": optimizer.max_iterations},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    return SimplexOptimizationResult(
        optimizer=optimizer,
        parameters=parameters,
        active_weights=weights_from_parameters(parameters),
        objective_value=float(result.fun),
        n_iterations=int(result.nit),
        converged=bool(result.success),
        message=str(result.message),
    )


def _validate_initial_vector(values: FloatArray, expected_length: int, *, name: str) -> None:
    if values.ndim != 1 or values.shape[0] != expected_length or not np.all(np.isfinite(values)):
        msg = f"{name} must be a finite 1D array with length {expected_length}."
        raise ValueError(msg)
