from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.distribution import Gaussian

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import GaussianComponentArrays
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.typing import PrecisionT


def as_gaussian_components(
    distribution: Gaussian[PrecisionT] | GaussianMixture[PrecisionT],
    /,
) -> GaussianComponentArrays[PrecisionT]:
    """Return normalized component weights, means, and covariances."""
    if isinstance(distribution, Gaussian):
        weights = np.ones(1, dtype=distribution.dtype)
        return weights, distribution.mean[None, :], distribution.covariance[None, :, :]

    weights = (distribution.weights / np.sum(distribution.weights)).astype(distribution.dtype)
    return weights, distribution.means, distribution.covariances
