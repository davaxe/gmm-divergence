from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gmm_divergence._core._types import Weights
    from gmm_divergence.distributions._combine import CombinedGaussianMixture
    from gmm_divergence.fitting._options import FitMethod, FitObjective


@dataclass(frozen=True, slots=True)
class MonteCarloStatistics:
    sample_mean: float
    sample_variance: float
    standard_error: float
    effective_sample_size: int


@dataclass(frozen=True, slots=True)
class DivergenceResult:
    """Result of a divergence estimation."""

    value: float
    """The estimated divergence."""
    method: str | None = None
    """The method used for estimation, if applicable."""
    num_samples: int | None = None
    """The number of samples used for estimation, if applicable."""
    monte_carlo_stats: MonteCarloStatistics | None = None
    """Monte Carlo statistics related to the estimation, if applicable.

    Contains the sample mean, sample variance, standard error, and effective
    sample size of the pointwise divergence estimates when using Monte Carlo
    estimation.
    """


@dataclass(frozen=True, slots=True, repr=False)
class FitResult:
    """Result of fitting a Gaussian mixture.

    Primary result when using
    [`fit_mixture_weights`][gmm_divergence.fitting.fit_mixture_weights]
    and related functions.
    """

    weights: Weights
    """The fitted mixture weights as a 1D array."""
    objective_value: float
    """The final scalar objective value minimized by the optimizer."""
    fitted_mixture: CombinedGaussianMixture
    """The full combined Gaussian mixture corresponding to the fitted weights."""
    fit_objective: FitObjective
    """The objective used to fit the mixture weights."""
    fit_method: FitMethod
    """The optimization method used to fit the mixture weights."""
    alpha: float | None = None
    """Forward-objective weight used by bidirectional fitting, if applicable."""
    iterations: int | None = None
    """The number of iterations taken by the optimization routine, if applicable."""
    converged: bool | None = None
    """Whether the optimization routine reported convergence, if applicable."""
    active_candidate_indices: tuple[int, ...] = ()
    """Original indices retained by candidate selection, in optimizer order."""
    optimizer_message: str = ""
    """Immutable optimizer termination message."""

    def candidate_weights(self) -> list[tuple[int, float]]:
        """Return fitted weights paired with original candidate indices.

        If candidate selection was used, indices refer to the original `q_i`
        sequence passed to `fit_mixture_weights`. Otherwise they are simply
        `0, 1, ..., n_candidates - 1`.
        """
        return sorted(
            [(int(index), float(weight)) for index, weight in enumerate(self.weights)],
            key=operator.itemgetter(1),
            reverse=True,
        )

    def candidate_weight_dict(self) -> dict[int, float]:
        """Return fitted weights keyed by original candidate index."""
        return dict(self.candidate_weights())
