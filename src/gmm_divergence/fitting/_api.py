"""Public API for fitting mixture weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import gmm_divergence.fitting._fit as wfit
from gmm_divergence._core._validation import validate_nonnegative_finite

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._options import FitMethod, FitObjective
    from gmm_divergence.fitting._selector import CandidateSelector
    from gmm_divergence.results import FitResult


def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod,
    objective: FitObjective,
    candidate_selector: CandidateSelector | None = None,
) -> FitResult:
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
    method : FitMethod
        Explicit optimizer configuration.
    objective : FitObjective
        Explicit fitting-objective configuration.

    Returns
    -------
    FitResult
        Result containing the fitted weights, combined mixture, final objective
        value, objective and optimizer configurations, and termination metadata.

    """
    return wfit.fit_mixture_weights(
        p=p, q_i=q_i, objective=objective, optimizer=method, candidate_selection=candidate_selector
    )


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
    validate_nonnegative_finite(min_weight, name="min_weight")

    weights = mixture.weights
    keep_mask = weights >= min_weight
    if not np.any(keep_mask):
        msg = "All components were pruned, increase min_weight threshold."
        raise ValueError(msg)

    return mixture.select_components(np.nonzero(keep_mask)[0])
