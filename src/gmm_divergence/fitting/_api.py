"""Public API for Gaussian- and categorical-mixture weight fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import gmm_divergence.fitting._categorical as categorical
import gmm_divergence.fitting._gaussian as gaussian
from gmm_divergence._core._validation import validate_nonnegative_finite

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._gaussian import PreparedGaussianMixtureFit
    from gmm_divergence.fitting._options import FitMethod, FitObjective
    from gmm_divergence.fitting._selector import CandidateSelector
    from gmm_divergence.results import CategoricalMixtureFitResult, GaussianMixtureFitResult


def prepare_gaussian_mixture_fit(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    objective: FitObjective,
    candidate_selector: CandidateSelector | None = None,
) -> PreparedGaussianMixtureFit:
    """Prepare a reusable Gaussian-mixture weight-fitting objective.

    Candidate selection, sampling, and density or moment calculations happen
    once during preparation. The returned object can evaluate gradients and run
    repeated optimizations without recomputing those inputs.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution.
    q_i : sequence of Gaussian or GaussianMixture
        Candidate distributions whose weights will be fitted.
    objective : FitObjective
        Explicit fitting-objective configuration.
    candidate_selector : CandidateSelector or None, default=None
        Optional candidate-selection strategy applied before preparation.

    Returns
    -------
    PreparedGaussianMixtureFit
        Cached objective data with separate `solve` and `report` stages.
    """
    return gaussian.prepare_gaussian_mixture_fit(
        p=p, q_i=q_i, objective=objective, candidate_selection=candidate_selector
    )


def fit_gaussian_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod,
    objective: FitObjective,
    candidate_selector: CandidateSelector | None = None,
) -> GaussianMixtureFitResult:
    r"""Fit weights for a mixture of fixed Gaussian-family candidate distributions.

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
    GaussianMixtureFitResult
        Result containing the fitted weights, combined mixture, final objective
        value, objective and optimizer configurations, and termination metadata.

    """
    prepared = prepare_gaussian_mixture_fit(
        p, q_i, objective=objective, candidate_selector=candidate_selector
    )
    solution = prepared.solve(method=method)
    return prepared.report(solution)


def fit_categorical_mixture_weights(
    p_w: npt.ArrayLike, q_i_w: npt.ArrayLike, /, *, method: FitMethod
) -> CategoricalMixtureFitResult:
    r"""Fit a weighted mixture of categorical distributions using forward KL.

    Find nonnegative candidate weights that sum to one and minimize

    $$
    D_{\mathrm{KL}}\left(
        p \;\middle\|\; \sum_{i=1}^{M} \alpha_i q_i
    \right),
    $$

    where `p` is the target distribution and `q_i` are the candidates.

    Parameters
    ----------
    p_w : array-like, shape (n_categories,)
        Target categorical probability distribution.
    q_i_w : array-like, shape (n_candidates, n_categories)
        Candidate probability distributions, one per row.
    method : FitMethod
        Simplex optimization method.

    Returns
    -------
    CategoricalMixtureFitResult
        Optimized candidate weights, fitted categorical probabilities,
        KL divergence, and optimizer convergence information.

    Notes
    -----
    Probabilities must be finite and strictly nonnegative. Target and candidate
    rows must sum to one within `rtol=1e-7`, `atol=1e-8`. Inputs are not
    normalized or clipped.
    """
    return categorical.fit_categorical_mixture_weights(p_w, q_i_w, method=method)


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
    _ = validate_nonnegative_finite(min_weight, name="min_weight")

    weights = mixture.weights
    keep_mask = weights >= min_weight
    if not np.any(keep_mask):
        msg = "All components were pruned, increase min_weight threshold."
        raise ValueError(msg)

    return mixture.select_components(np.nonzero(keep_mask)[0])
