from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from gmm_divergence.fitting import weights as wfit

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.results import KLFitResult

FitMethod = Literal["softmax-lbfgsb", "simplex-slsqp", "em"]


class _CommonArgs(TypedDict):
    target: Gaussian | GaussianMixture
    components: Sequence[Gaussian | GaussianMixture]
    num_samples: int
    samples: npt.ArrayLike | None
    rng: np.random.Generator | int | None
    tol: float
    max_iterations: int


def fit_mixture_weights(
    target: Gaussian | GaussianMixture,
    components: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    """Fit mixture weights to minimize KL divergence."""
    common_kwargs: _CommonArgs = {
        "target": target,
        "components": components,
        "num_samples": num_samples,
        "samples": samples,
        "rng": rng,
        "tol": tol,
        "max_iterations": max_iterations,
    }
    match method:
        case "softmax-lbfgsb":
            return wfit.fit_mixture_weights_softmax(**common_kwargs)
        case "simplex-slsqp":
            return wfit.fit_mixture_weights_simplex(**common_kwargs)
        case "em":
            return wfit.fit_mixture_weights_em(**common_kwargs)
