from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp as _scipy_logsumexp

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.typing import PrecisionT


def logsumexp(a: npt.NDArray[PrecisionT], axis: int = -1) -> npt.NDArray[PrecisionT]:
    """Compute log-sum-exp while preserving the input dtype."""
    return np.asarray(_scipy_logsumexp(a, axis=axis).astype(a.dtype, copy=False))


def resolve_samples(
    distribution: Distribution[PrecisionT],
    num_samples: int,
    samples: npt.ArrayLike | None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[PrecisionT]:
    """Return provided samples or draw samples from the distribution."""
    dim = distribution.dim
    if samples is not None:
        samples = np.asarray(samples, dtype=distribution.dtype)
        if samples.ndim != 2 or samples.shape[1] != dim:
            msg = f"Expected samples of shape (n_samples, n_features), got {samples.shape}"
            raise ValueError(msg)
        return samples
    return distribution.sample(n_samples=num_samples, rng=rng)
