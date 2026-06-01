from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from gmm_divergence.gmm.model import GaussianMixture, PrecisionT


def sample_gmm(
    gmm: GaussianMixture[PrecisionT],
    /,
    n_samples: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[PrecisionT]:
    """Draw samples from a Gaussian mixture."""
    rng = np.random.default_rng(rng)
    weights = gmm.weights / np.sum(gmm.weights)

    component_ids = rng.choice(
        gmm.n_components,
        size=n_samples,
        p=weights,
    )

    if gmm.covariance_type == "diag":
        stds = np.sqrt(gmm.covariances[component_ids])
        eps = rng.standard_normal(size=gmm.means[component_ids].shape)
        samples = gmm.means[component_ids] + eps * stds
    elif gmm.covariance_type == "full":
        chol = gmm.chol()
        eps = rng.standard_normal(size=gmm.means[component_ids].shape)
        samples = gmm.means[component_ids] + np.einsum(
            "nij,nj->ni",
            chol[component_ids],
            eps,
        )
    else:
        msg = f"Unsupported covariance type: {gmm.covariance_type}"
        raise ValueError(msg)

    return samples.astype(gmm.means.dtype, copy=False)
