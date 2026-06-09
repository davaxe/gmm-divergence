from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

    from gmm_divergence._core._types import Weights
    from gmm_divergence.distributions._combine import CombinedGaussianMixture
    from gmm_divergence.fitting._options import FitObjective


@dataclass(frozen=True, slots=True)
class DivergenceResult:
    """Result of a divergence estimation."""

    value: float
    """The estimated divergence."""
    method: str | None = None
    """The method used for estimation, if applicable."""
    num_samples: int | None = None
    """The number of samples used for estimation, if applicable."""


@dataclass(frozen=True, slots=True, repr=False)
class KLFitResult:
    """Result of fitting a Gaussian mixture to minimize KL divergence.

    Primary result when using
    [`fit_mixture_weights`][gmm_divergence.fitting.fit_mixture_weights]
    and related functions.
    """

    weights: Weights
    """The fitted mixture weights as a 1D array."""
    fit_objective: FitObjective
    """The KL objective used to fit the mixture weights."""
    objective_value: float
    """The final scalar objective value minimized by the optimizer."""
    forward_kl: DivergenceResult
    """Estimated forward KL divergence, ``KL(p || q_w)``."""
    reverse_kl: DivergenceResult
    """Estimated reverse KL divergence, ``KL(q_w || p)``."""
    scipy_result: OptimizeResult | None
    """The full result object returned by the optimization routine, if used."""
    fitted_mixture: CombinedGaussianMixture
    """The full combined Gaussian mixture corresponding to the fitted weights."""
    alpha: float | None = None
    """Forward-objective weight used by bidirectional fitting, if applicable."""
    iterations: int | None = None
    """The number of iterations taken by the optimization routine, if applicable."""
    converged: bool | None = None
    """Whether the optimization routine reported convergence, if applicable."""
    used_candidate_indices: list[int] | None = None
