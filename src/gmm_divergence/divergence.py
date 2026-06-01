from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from gmm_divergence.methods.gaussian_approx import kl_gaussian_approximation
from gmm_divergence.methods.monte_carlo import kl_monte_carlo
from gmm_divergence.methods.unscented import kl_unscented

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.gmm.model import GaussianMixture, PrecisionT
    from gmm_divergence.results import DivergenceResult

KLMethod = Literal["monte_carlo", "unscented", "gaussian_approximation"]


@overload
def kl_divergence(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    *,
    method: Literal["gaussian_approximation"] = ...,
    approximation: Literal["nearest", "moment_matching"] = "moment_matching",
) -> DivergenceResult: ...


@overload
def kl_divergence(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    *,
    method: Literal["unscented"] = ...,
) -> DivergenceResult: ...


@overload
def kl_divergence(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    *,
    method: Literal["monte_carlo"] = "monte_carlo",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> DivergenceResult: ...


def kl_divergence(
    p: GaussianMixture[PrecisionT],
    q: GaussianMixture[PrecisionT],
    /,
    *,
    method: KLMethod = "monte_carlo",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    approximation: Literal["nearest", "moment_matching"] = "moment_matching",
) -> DivergenceResult:
    """Compute KL divergence between two Gaussian mixtures.

    Parameters
    ----------
    p, q : GaussianMixture
        The two Gaussian mixtures to compare. The KL divergence is computed as
        KL(p || q).
    method : KLMethod
        The method to use for estimating the KL divergence. "monte_carlo" uses
        Monte Carlo sampling, while "unscented" uses unscented sigma points.
    num_samples : int
        The number of samples to use for the Monte Carlo method.
    samples : array-like, optional
        Pre-generated samples from the first mixture `p` to use for the Monte
        Carlo method.
    rng : np.random.Generator, int, or None
        Random number generator or seed for the Monte Carlo method.

    Returns
    -------
    DivergenceResult
        An object containing the estimated KL divergence and metadata about the
        estimation method. The value for the KL divergence estimate is stored in
        the `value` attribute of the returned object.
    """
    match method:
        case "monte_carlo":
            return kl_monte_carlo(
                p,
                q,
                num_samples=num_samples,
                samples=samples,
                rng=rng,
            )
        case "unscented":
            return kl_unscented(p, q)
        case "gaussian_approximation":
            return kl_gaussian_approximation(
                p,
                q,
                approximation=approximation,
            )
