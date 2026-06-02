from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.typing import FloatArray


def logsumexp(a: FloatArray, axis: int = -1) -> FloatArray:
    """Compute log-sum-exp."""
    return np.asarray(_scipy_logsumexp(a, axis=axis), dtype=np.float64)


def resolve_samples(
    distribution: Distribution,
    num_samples: int,
    samples: npt.ArrayLike | None,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Return provided samples or draw samples from the distribution."""
    dim = distribution.dim
    if samples is not None:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != dim:
            msg = f"Expected samples of shape (n_samples, n_features), got {samples.shape}"
            raise ValueError(msg)
        return samples
    return distribution.sample(n_samples=num_samples, rng=rng)
