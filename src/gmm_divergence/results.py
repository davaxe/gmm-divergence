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
    value: float
    method: str | None = None
    num_samples: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class KLFitResult:
    weights: npt.NDArray[np.float64]
    objective: float
    estimated_kl: DivergenceResult
    scipy_result: OptimizeResult | None
    fitted_mixture: CombinedGaussianMixture
    iterations: int | None = None
    converged: bool | None = None
