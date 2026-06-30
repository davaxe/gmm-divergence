"""Public API for fitting mixture weights."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt

import gmm_divergence.fitting._fit as wfit
from gmm_divergence._core._dispatch import MethodSpec, Registry
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    ForwardKL,
    JensenShannon,
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
    from gmm_divergence.fitting._selector import CandidateSelector
    from gmm_divergence.results import KLFitResult

OptionsT = TypeVar("OptionsT")

_OPTIMIZER_REGISTRY = Registry(
    label="fit optimizer",
    specs=(
        MethodSpec(name="softmax_lbfgsb", option_type=SoftmaxLBFGSB, default=SoftmaxLBFGSB()),
        MethodSpec(name="simplex_slsqp", option_type=SimplexSLSQP, default=SimplexSLSQP()),
    ),
)

_OBJECTIVE_REGISTRY = Registry(
    label="fit objective",
    specs=(
        MethodSpec(name="forward", option_type=ForwardKL, default=ForwardKL()),
        MethodSpec(name="reverse", option_type=ReverseKL, default=ReverseKL()),
        MethodSpec(name="bidirectional", option_type=BidirectionalKL, default=BidirectionalKL()),
        MethodSpec(name="jensen_shannon", option_type=JensenShannon, default=JensenShannon()),
        MethodSpec(name="moment_matching", option_type=MomentMatching, default=MomentMatching()),
    ),
)


def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: WeightFitMethod = "softmax_lbfgsb",
    objective: WeightFitObjective = "forward",
    x0: npt.ArrayLike | None = None,
    candidate_selector: CandidateSelector[Gaussian | GaussianMixture] | None = None,
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
        `JensenShannon(...)`, or `MomentMatching(...)` for objective-specific
        options.
    x0 : array-like, optional
        Initial weights for the optimized variables. If `None`, the optimizer's
        default initialization is used.

    Returns
    -------
    KLFitResult
        Result containing the fitted weights, fitted mixture, fit objective,
        objective value, forward/reverse KL diagnostics, and optimizer metadata.

    """
    method_spec, optimizer = _OPTIMIZER_REGISTRY.resolve(method)
    _objective_spec, objective_config = _OBJECTIVE_REGISTRY.resolve(objective)

    match method_spec.name:
        case "softmax_lbfgsb":
            optimizer = _cast_options(optimizer, SoftmaxLBFGSB)
            objective_config = _cast_fit_objective(objective_config)
            return wfit.fit_mixture_weights(
                p=p,
                q_i=q_i,
                objective=objective_config,
                optimizer=optimizer,
                parameterization="softmax",
                x0=x0,
                candidate_selection=candidate_selector,
            )
        case "simplex_slsqp":
            optimizer = _cast_options(optimizer, SimplexSLSQP)
            objective_config = _cast_fit_objective(objective_config)
            return wfit.fit_mixture_weights(
                p=p,
                q_i=q_i,
                objective=objective_config,
                optimizer=optimizer,
                parameterization="simplex",
                x0=x0,
                candidate_selection=candidate_selector,
            )
        case _:
            msg = "Unhandled fit optimizer registry entry."
            raise AssertionError(msg)


def prune_mixture(mixture: GaussianMixture, *, min_weight: float = 1e-4) -> GaussianMixture:
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

    Returns
    -------
    GaussianMixture
        The pruned mixture with weights normalized to sum to one.

    Raises
    ------
    ValueError
        If all components are pruned.
    """
    if not np.isfinite(min_weight) or min_weight < 0.0:
        msg = f"min_weight must be a nonnegative finite value, got {min_weight}."
        raise ValueError(msg)

    weights = mixture.weights
    keep_mask = weights >= min_weight
    if not np.any(keep_mask):
        msg = "All components were pruned, increase min_weight threshold."
        raise ValueError(msg)

    return mixture.select_components(np.nonzero(keep_mask)[0])


def _cast_options(options: object, option_type: type[OptionsT]) -> OptionsT:
    if not isinstance(options, option_type):
        msg = "Dispatcher returned an option object with the wrong type."
        raise TypeError(msg)
    return options


def _cast_fit_objective(options: object) -> wfit.FitObjectiveConfig:
    if not isinstance(
        options, (ForwardKL, ReverseKL, BidirectionalKL, JensenShannon, MomentMatching)
    ):
        msg = "Dispatcher returned an objective object with the wrong type."
        raise TypeError(msg)
    return options
