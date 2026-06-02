from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from gmm_divergence.distribution import Gaussian, GaussianMixture
from gmm_divergence.estimators.exact import kl_exact
from gmm_divergence.estimators.gaussian_approx import kl_gaussian_approximation
from gmm_divergence.estimators.monte_carlo import kl_monte_carlo
from gmm_divergence.estimators.unscented import kl_unscented

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.results import DivergenceResult

logger = logging.getLogger(__name__)

KLMethod = Literal["monte_carlo", "unscented", "gaussian_approximation", "exact"]

GaussianFamily = Gaussian | GaussianMixture


@overload
def kl_divergence(
    p: Gaussian,
    q: Gaussian,
    /,
    *,
    method: Literal["exact"] = "exact",
) -> DivergenceResult: ...


@overload
def kl_divergence(
    p: Gaussian | GaussianMixture,
    q: Gaussian | GaussianMixture,
    /,
    *,
    method: Literal["gaussian_approximation"] = "gaussian_approximation",
    approximation: Literal["nearest", "moment_matching"] = ...,
) -> DivergenceResult: ...


@overload
def kl_divergence(
    p: Gaussian | GaussianMixture,
    q: Distribution,
    /,
    *,
    method: Literal["unscented"] = ...,
) -> DivergenceResult: ...


@overload
def kl_divergence(
    p: Distribution,
    q: Distribution,
    /,
    *,
    method: Literal["monte_carlo"] = "monte_carlo",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> DivergenceResult: ...


def kl_divergence(
    p: Distribution,
    q: Distribution,
    /,
    *,
    method: KLMethod = "monte_carlo",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    approximation: Literal["nearest", "moment_matching"] = "moment_matching",
) -> DivergenceResult:
    """Compute KL divergence between Gaussian distributions.

    Parameters
    ----------
    p, q : Distribution
        The two distributions to compare. The KL divergence is estimated as
        KL(p || q), so `p` is the "true" distribution and `q` is the
        "approximation". The distributions must have the same dimensionality.
    method : KLMethod
        The method to use for estimating the KL divergence.
    num_samples : int
        The number of samples to use for the Monte Carlo method.
    samples : array-like, optional
        Pre-generated samples from the first distribution `p` to use for the Monte
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
    if p.dim != q.dim:
        msg = f"Distribution dimensions must match, got {p.dim} and {q.dim}."
        raise ValueError(msg)
    if isinstance(p, Gaussian) and isinstance(q, Gaussian) and method != "exact":
        msg = (
            "Both distributions are Gaussian. Consider using method='exact'"
            "for a closed-form solution."
        )
        logger.warning(msg)

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
            if not isinstance(p, (Gaussian, GaussianMixture)):
                msg = "Unscented KL estimation requires Gaussian or GaussianMixture input for p."
                raise TypeError(msg)
            return kl_unscented(p, q)
        case "gaussian_approximation":
            if not isinstance(p, (Gaussian, GaussianMixture)) or not isinstance(
                q, (Gaussian, GaussianMixture)
            ):
                msg = "Gaussian approximation requires Gaussian or GaussianMixture inputs."
                raise TypeError(msg)
            return kl_gaussian_approximation(
                p,
                q,
                approximation=approximation,
            )
        case "exact":
            if not isinstance(p, Gaussian) or not isinstance(q, Gaussian):
                msg = "Exact KL computation requires Gaussian inputs."
                raise TypeError(msg)
            return kl_exact(p, q)
