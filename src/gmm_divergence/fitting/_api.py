from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

import gmm_divergence.fitting._weights as wfit
from gmm_divergence._core._dispatch import MethodSpec, Registry
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    ForwardKL,
    MomentMatching,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    WeightFitMethod,
    WeightFitObjective,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.results import KLFitResult

OptionsT = TypeVar("OptionsT")

_OPTIMIZER_REGISTRY = Registry(
    label="fit optimizer",
    specs=(
        MethodSpec(name="softmax-lbfgsb", option_type=SoftmaxLBFGSB, default=SoftmaxLBFGSB()),
        MethodSpec(name="simplex-slsqp", option_type=SimplexSLSQP, default=SimplexSLSQP()),
    ),
)

_OBJECTIVE_REGISTRY = Registry(
    label="fit objective",
    specs=(
        MethodSpec(name="forward", option_type=ForwardKL, default=ForwardKL()),
        MethodSpec(name="reverse", option_type=ReverseKL, default=ReverseKL()),
        MethodSpec(name="bidirectional", option_type=BidirectionalKL, default=BidirectionalKL()),
        MethodSpec(name="moment_matching", option_type=MomentMatching, default=MomentMatching()),
    ),
)


def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: WeightFitMethod = "softmax-lbfgsb",
    objective: WeightFitObjective = "forward",
) -> KLFitResult:
    r"""Fit weights for a mixture of fixed candidate distributions.

    Fits nonnegative weights `w_i` such that the weighted mixture

    $$
    q_w(x) = \sum_i w_i q_i(x)
    $$

    approximates the reference distribution `p` according to the selected
    objective.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution.
    q_i : sequence of Gaussian or GaussianMixture
        Candidate distributions whose weights are fitted.
    method : str or optimizer configuration, optional
        Optimizer used for the weights. Passing a string runs that optimizer
        with defaults. Use `SoftmaxLBFGSB(...)` or `SimplexSLSQP(...)` for
        optimizer-specific options.
    objective : str or WeightFitObjective configuration, optional
        Objective used for fitting. Passing a string runs that objective with
        defaults. Use `ForwardKL(...)`, `ReverseKL(...)`, `BidirectionalKL(...)`,
        or `MomentMatching(...)` for objective-specific options.

    Returns
    -------
    KLFitResult
        Result containing the fitted weights, fitted mixture, fit objective,
        objective value, forward/reverse KL diagnostics, and optimizer metadata.

    """
    method_spec, optimizer = _OPTIMIZER_REGISTRY.resolve(method)
    _objective_spec, objective_config = _OBJECTIVE_REGISTRY.resolve(objective)

    match method_spec.name:
        case "softmax-lbfgsb":
            optimizer = _cast_options(optimizer, SoftmaxLBFGSB)
            objective_config = _cast_fit_objective(objective_config)
            return wfit.fit_mixture_weights_softmax(
                p, q_i, objective=objective_config, optimizer=optimizer
            )
        case "simplex-slsqp":
            optimizer = _cast_options(optimizer, SimplexSLSQP)
            objective_config = _cast_fit_objective(objective_config)
            return wfit.fit_mixture_weights_simplex(
                p, q_i, objective=objective_config, optimizer=optimizer
            )
        case _:
            msg = "Unhandled fit optimizer registry entry."
            raise AssertionError(msg)


def prune_mixture(
    mixture: GaussianMixture, *, min_weight: float = 1e-4, renormalize: bool = True
) -> GaussianMixture:
    """Prune components of a Gaussian mixture with small weights.

    This is a common post-processing step after fitting to remove components
    that contribute negligibly to the mixture, improving efficiency and
    interpretability.

    Parameters
    ----------
    mixture : GaussianMixture
        The mixture to prune.
    min_weight : float, optional
        Minimum weight threshold for keeping components. Components with weights
        below this threshold will be removed. Default is 1e-4.
    renormalize : bool, optional
        Whether to renormalize the remaining weights to sum to 1 after pruning.

    Returns
    -------
    GaussianMixture
        The pruned mixture.

    Raises
    ------
    ValueError
        If all components are pruned, or if renormalization fails due to zero
        total weight after pruning.
    """
    weights = mixture.weights
    keep_mask = weights >= min_weight
    if not np.any(keep_mask):
        msg = "All components were pruned, increase min_weight threshold."
        raise ValueError(msg)

    return mixture.select_components(np.nonzero(keep_mask)[0], renormalize=renormalize)


def _cast_options(options: object, option_type: type[OptionsT]) -> OptionsT:
    if not isinstance(options, option_type):
        msg = "Dispatcher returned an option object with the wrong type."
        raise TypeError(msg)
    return options


def _cast_fit_objective(options: object) -> wfit.FitObjectiveConfig:
    if not isinstance(options, (ForwardKL, ReverseKL, BidirectionalKL, MomentMatching)):
        msg = "Dispatcher returned an objective object with the wrong type."
        raise TypeError(msg)
    return options
