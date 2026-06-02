from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.distribution import Gaussian

if TYPE_CHECKING:
    from gmm_divergence.distribution.base import GaussianComponentArrays
    from gmm_divergence.distribution.gmm import GaussianMixture


def as_gaussian_components(distribution: Gaussian | GaussianMixture, /) -> GaussianComponentArrays:
    """Return normalized component weights, means, and covariances."""
    if isinstance(distribution, Gaussian):
        weights = np.ones(1, dtype=np.float64)
        return weights, distribution.mean[None, :], distribution.covariance[None, :, :]

    weights = (distribution.weights / np.sum(distribution.weights)).astype(np.float64)
    return weights, distribution.means, distribution.covariances
