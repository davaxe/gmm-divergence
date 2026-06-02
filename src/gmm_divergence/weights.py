from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gmm_divergence.fitting.constrained import fit_forward_kl_weights_constrained

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.results import KLFitResult

FitMethod = Literal["forward_kl_constrained"]


def fit_weights(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "forward_kl_constrained",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> KLFitResult:
    """Fit mixture weights to minimize KL divergence."""
    match method:
        case "forward_kl_constrained":
            return fit_forward_kl_weights_constrained(
                target=target,
                components=components,
                num_samples=num_samples,
                samples=samples,
                rng=rng,
            )
