"""Public API for Gaussian- and categorical-mixture weight fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gmm_divergence.fitting._categorical as categorical
import gmm_divergence.fitting._gaussian as gaussian
from gmm_divergence.distributions._combine import prune_mixture
from gmm_divergence.fitting._options import validate_optimizer

__all__ = [
    "fit_categorical_weights",
    "fit_gaussian_weights",
    "prepare_gaussian_weight_fit",
    "prune_mixture",
]

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.fitting._gaussian import PreparedGaussianWeightFit
    from gmm_divergence.fitting._options import FitObjective, Optimizer
    from gmm_divergence.fitting._selector import CandidateSelector
    from gmm_divergence.results import CategoricalWeightFitResult, GaussianWeightFitResult


def prepare_gaussian_weight_fit(
    target: Gaussian | GaussianMixture,
    candidates: Sequence[Gaussian | GaussianMixture],
    *,
    objective: FitObjective,
    candidate_selector: CandidateSelector | None = None,
) -> PreparedGaussianWeightFit:
    """Prepare a reusable Gaussian-mixture weight-fitting objective.

    Candidate selection, sampling, and density or moment calculations happen
    once during preparation. The returned object can evaluate gradients and run
    repeated optimizations without recomputing those inputs.

    Parameters
    ----------
    target : Gaussian or GaussianMixture
        Reference distribution.
    candidates : sequence of Gaussian or GaussianMixture
        Nonempty candidate sequence with the same feature dimension as target.
    objective : FitObjective
        Explicit fitting-objective configuration.
    candidate_selector : CandidateSelector or None, default=None
        Optional strategy applied once before sampling and caching densities.
        Precomputed candidate batches must cover all original candidates.

    Returns
    -------
    PreparedGaussianWeightFit
        Cached objective data with `fit` and `evaluate` methods.
    """
    return gaussian.prepare_gaussian_weight_fit(
        target=target,
        candidates=candidates,
        objective=objective,
        candidate_selection=candidate_selector,
    )


def fit_gaussian_weights(
    target: Gaussian | GaussianMixture,
    candidates: Sequence[Gaussian | GaussianMixture],
    *,
    optimizer: Optimizer,
    objective: FitObjective,
    candidate_selector: CandidateSelector | None = None,
) -> GaussianWeightFitResult:
    r"""Fit weights for a mixture of fixed Gaussian-family candidate distributions.

    Fits nonnegative weights `w_i` such that the weighted mixture

    $$
    q_w(x) = \sum_i w_i q_i(x)
    $$

    approximates the reference distribution `p` according to the selected
    objective.

    Parameters
    ----------
    target : Gaussian or GaussianMixture
        Reference distribution.
    candidates : sequence of Gaussian or GaussianMixture
        Nonempty candidate sequence with the same feature dimension as target.
    optimizer : Optimizer
        Explicit optimizer configuration.
    objective : FitObjective
        Explicit fitting-objective configuration.
    candidate_selector : CandidateSelector or None, default=None
        Optional candidate selection before preparation. Initial optimizer
        vectors refer to retained candidates in selected order. Precomputed
        sampling batches refer to all original candidates.

    Returns
    -------
    GaussianWeightFitResult
        Result containing the fitted weights, combined mixture, final objective
        value, objective and optimizer configurations, and termination metadata.
        Weights remain aligned to the original candidates, with zeros for those
        rejected by selection. KL and Jensen-Shannon values are in nats;
        MomentMatching reports a squared raw-moment residual.

    """
    _ = validate_optimizer(optimizer)
    prepared = prepare_gaussian_weight_fit(
        target, candidates, objective=objective, candidate_selector=candidate_selector
    )
    return prepared.fit(optimizer=optimizer)


def fit_categorical_weights(
    target: npt.ArrayLike, candidates: npt.ArrayLike, *, optimizer: Optimizer
) -> CategoricalWeightFitResult:
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
    target : array-like, shape (n_categories,)
        Target categorical probability distribution.
    candidates : array-like, shape (n_candidates, n_categories)
        Candidate probability distributions, one per row.
    optimizer : Optimizer
        Explicit optimizer configuration. Initial vectors have one entry per
        candidate row.

    Returns
    -------
    CategoricalWeightFitResult
        Optimized candidate weights, fitted categorical probabilities,
        KL divergence, and optimizer convergence information.

    Notes
    -----
    Probabilities must be finite and strictly nonnegative. Target and candidate
    rows must sum to one within `rtol=1e-7`, `atol=1e-8`. Inputs are not
    normalized or clipped.

    Every category with positive target mass must have positive mass in at
    least one candidate. The objective is evaluated exactly over categories;
    it uses no sampling or approximation. Only forward KL fitting is supported.
    """
    _ = validate_optimizer(optimizer)
    return categorical.fit_categorical_weights(target, candidates, optimizer=optimizer)
