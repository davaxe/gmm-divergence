from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from gmm_divergence.distribution import Gaussian, GaussianMixture
from gmm_divergence.estimators.closed_form import kl_closed_form
from gmm_divergence.estimators.gaussian_approx import kl_gaussian_approximation
from gmm_divergence.estimators.monte_carlo import kl_monte_carlo
from gmm_divergence.estimators.unscented import kl_unscented

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.distribution.base import Distribution
    from gmm_divergence.results import DivergenceResult

logger = logging.getLogger(__name__)

EstimationMethod = Literal["monte_carlo", "unscented", "gaussian_approximation", "closed_form"]

GaussianFamily = Gaussian | GaussianMixture


@overload
def kl_divergence(
    p: Gaussian,
    q: Gaussian,
    /,
    *,
    method: Literal["closed_form"] = "closed_form",
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
    method: EstimationMethod = "monte_carlo",
    num_samples: int = 10_000,
    samples: npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    approximation: Literal["nearest", "moment_matching"] = "moment_matching",
    force_closed_form: bool = False,
) -> DivergenceResult:
    r"""Compute the Kullback--Leibler divergence between two distributions.

    Computes

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{x \sim p}
    \left[
        \log p(x) - \log q(x)
    \right].
    $$

    Here, `p` is treated as the reference distribution and `q` as the
    approximating distribution.

    !!! warning "Not all methods support all distribution types"
        Some estimation methods require specific distribution types. The monte
        Carlo method is the most general, while other methods may require
        Gaussian or Gaussian mixture distributions.

    Parameters
    ----------
    p, q : Distribution
        The two distributions to compare. They must have the same dimensionality.
    method : EstimationMethod, default="monte_carlo"
        Method used to compute or estimate the KL divergence.

        Supported methods are:

        - `"closed_form"`:
            Use an analytical formula when available. This is the preferred method
            for distribution pairs with a known KL expression, such as two
            Gaussian distributions.

        - `"monte_carlo"`:
            Estimate the KL divergence using samples from `p`:

            $$
            D_{\mathrm{KL}}(p \| q)
            \approx
            \frac{1}{N}
            \sum_{i=1}^{N}
            \left[
                \log p(x_i) - \log q(x_i)
            \right],
            \qquad x_i \sim p.
            $$

            This method is general but stochastic. Accuracy depends on
            `num_samples` and the random seed.

        - `"unscented"`:
            Use the Unscented Transform to approximate the KL divergence.

        - `"gaussian_approximation"`:
            Approximate both distributions as single Gaussians (e.g., by
            matching moments) and compute the KL divergence between the
            approximations.

    num_samples : int, default=10_000
        Number of samples used when `method="monte_carlo"`.
        Ignored by closed-form methods.
    samples : array-like, optional
        Optional precomputed samples from `p`. If provided, these samples are
        used instead of drawing new samples.
    rng : numpy.random.Generator, int, optional
        Random number generator or seed used when sampling is required.
    approximation : {"nearest", "moment_matching"}
        Strategy used when "gaussian_approximation" method is selected:

        - `"nearest"`:
          Approximate the KL as the nearest (smallest) available closed-form KL
          between any pair of components from `p` and `q`.

        - `"moment_matching"`:
          Approximate unsupported distributions by matching moments, mean and
          covariance, before computing the KL divergence.

    force_closed_form : bool, default=False
        If `True`, require a closed-form expression. Raises an error if no
        analytical expression is available.

    Returns
    -------
    DivergenceResult
        Result object containing the estimated KL divergence and metadata about
        the computation, such as the method used and whether the result is exact
        or approximate.

    Notes
    -----
    The KL divergence is asymmetric:

    $$
    D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p).
    $$

    Therefore, swapping `p` and `q` generally gives a different result.

    Examples
    --------
    Compute the KL divergence using the default Monte Carlo estimator:

    ```python
    result = kl_divergence(p, q)
    print(result.value)
    ```

    Require a closed-form expression (only available if both `p` and `q` are Gaussians):

    ```python
    result = kl_divergence(p, q, method="closed_form")
    ```

    Use precomputed samples:

    ```python
    samples = p.sample(50_000, rng=0)
    result = kl_divergence(p, q, samples=samples)
    ```
    """
    _validate_same_dimension(p, q)
    exact_result = _maybe_compute_closed_form(p, q, force_closed_form=force_closed_form)
    if exact_result is not None:
        return exact_result

    return _kl_divergence_by_method(
        p,
        q,
        method=method,
        num_samples=num_samples,
        samples=samples,
        rng=rng,
        approximation=approximation,
    )


def _validate_same_dimension(p: Distribution, q: Distribution) -> None:
    if p.dim != q.dim:
        msg = f"Distribution dimensions must match, got {p.dim} and {q.dim}."
        raise ValueError(msg)


def _maybe_compute_closed_form(
    p: Distribution,
    q: Distribution,
    *,
    force_closed_form: bool,
) -> DivergenceResult | None:
    p_single, q_single = _single_gaussian(p), _single_gaussian(q)

    if p_single is None or q_single is None:
        return None

    if force_closed_form:
        return kl_closed_form(p_single, q_single)

    logger.warning(
        "Both distributions are Gaussian. Consider using method='exact' for a closed-form solution."
    )
    return None


def _kl_divergence_by_method(
    p: Distribution,
    q: Distribution,
    *,
    method: EstimationMethod,
    num_samples: int,
    samples: npt.ArrayLike | None,
    rng: np.random.Generator | int | None,
    approximation: Literal["nearest", "moment_matching"],
) -> DivergenceResult:
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
            return kl_unscented(
                _require_gaussian_or_mixture(p, method, name="p"),
                _require_gaussian_or_mixture(q, method, name="q"),
            )

        case "gaussian_approximation":
            return kl_gaussian_approximation(
                _require_gaussian_or_mixture(p, method, name="p"),
                _require_gaussian_or_mixture(q, method, name="q"),
                approximation=approximation,
            )

        case "closed_form":
            return kl_closed_form(
                _require_gaussian(p, method, name="p"),
                _require_gaussian(q, method, name="q"),
            )


def _require_gaussian(
    d: Distribution,
    method: EstimationMethod,
    *,
    name: str,
) -> Gaussian:
    if not isinstance(d, Gaussian):
        msg = f"Method '{method}' requires '{name}' to be a Gaussian, got {type(d).__name__}."
        raise TypeError(msg)
    return d


def _require_gaussian_or_mixture(
    d: Distribution,
    method: EstimationMethod,
    *,
    name: str,
) -> Gaussian | GaussianMixture:
    if not isinstance(d, (Gaussian, GaussianMixture)):
        msg = (
            f"Method '{method}' requires '{name}' to be a Gaussian or GaussianMixture, "
            f"got {type(d).__name__}."
        )
        raise TypeError(msg)
    return d


def _single_gaussian(d: Distribution) -> Gaussian | None:
    if isinstance(d, Gaussian):
        return d
    if isinstance(d, GaussianMixture):
        return d.as_gaussian(only_if_single=True)
    return None
