from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from scipy.optimize import OptimizeResult

    from gmm_divergence.distribution.combine import CombinedGaussianMixture


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
    [`fit_mixture_weights`][gmm_divergence.fit.fit_mixture_weights]
    and related functions.
    """

    weights: npt.NDArray[np.float64]
    """The fitted mixture weights as a 1D array."""
    objective: float
    """The final value of the optimization objective."""
    estimated_kl: DivergenceResult
    """The estimated KL divergence of the fitted mixture to the target."""
    scipy_result: OptimizeResult | None
    """The full result object returned by the optimization routine, if used."""
    fitted_mixture: CombinedGaussianMixture
    """The full combined Gaussian mixture corresponding to the fitted weights."""
    iterations: int | None = None
    """The number of iterations taken by the optimization routine, if applicable."""
    converged: bool | None = None
    """Whether the optimization routine reported convergence, if applicable."""
