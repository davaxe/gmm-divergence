from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import PrecisionT


def kl_monte_carlo(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    *,
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> DivergenceResult:
    """Estimate KL divergence with samples from the first mixture."""
    samples = _resolve_samples(p, num_samples, samples, rng)
    return DivergenceResult(
        value=float(np.mean(p.logpdf(samples) - q.logpdf(samples))),
        method="monte_carlo",
        num_samples=num_samples,
    )


def _resolve_samples(
    gmm: GaussianMixture[PrecisionT],
    num_samples: int,
    samples: npt.ArrayLike | None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[PrecisionT]:
    if samples is not None:
        samples = np.asarray(samples, dtype=gmm.dtype)
        if samples.ndim != 2 or samples.shape[1] != gmm.means.shape[1]:
            msg = f"Expected samples of shape (n_samples, n_features), got {samples.shape}"
            raise ValueError(msg)
        return samples
    return gmm.sample(n_samples=num_samples, rng=rng)
