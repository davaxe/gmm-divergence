from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence.results import DivergenceResult
from gmm_divergence.utils import resolve_samples

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.typing import PrecisionT


def kl_monte_carlo(
    p: Distribution[PrecisionT],
    q: Distribution[PrecisionT],
    /,
    *,
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> DivergenceResult:
    """Estimate KL divergence with samples from the first distribution."""
    samples = resolve_samples(p, num_samples, samples, rng)
    return DivergenceResult(
        value=float(np.mean(p.logpdf(samples) - q.logpdf(samples))),
        method="monte_carlo",
        num_samples=num_samples,
    )
