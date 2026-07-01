from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from gmm_divergence._core._dispatch import MethodSpec, Registry
from gmm_divergence._core._numeric import pairwise_gaussian_kl
from gmm_divergence.distributions._combine import combine_gaussians
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture
from gmm_divergence.divergence._options import (
    ClosedForm,
    GaussianApproximation,
    KLMethod,
    MonteCarlo,
    Unscented,
    Variational,
)
from gmm_divergence.divergence.methods._closed_form import kl_closed_form
from gmm_divergence.divergence.methods._gaussian_approx import kl_gaussian_approximation
from gmm_divergence.divergence.methods._monte_carlo import kl_monte_carlo
from gmm_divergence.divergence.methods._unscented import kl_unscented
from gmm_divergence.divergence.methods._variational import kl_variational
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._base import Distribution

OptionsT = TypeVar("OptionsT")

KL_REGISTRY = Registry(
    label="KL",
    specs=(
        MethodSpec(name="monte_carlo", option_type=MonteCarlo, default=MonteCarlo()),
        MethodSpec(name="unscented", option_type=Unscented, default=Unscented()),
        MethodSpec(
            name="gaussian_approximation",
            option_type=GaussianApproximation,
            default=GaussianApproximation(),
        ),
        MethodSpec(name="closed_form", option_type=ClosedForm, default=ClosedForm()),
        MethodSpec(name="variational", option_type=Variational, default=Variational()),
    ),
)


def kl_divergence(
    p: Distribution,
    q: Distribution,
    /,
    *,
    method: KLMethod = "monte_carlo",
    prefer_closed_form: bool = False,
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

    Parameters
    ----------
    p, q : Distribution
        The two distributions to compare. They must have the same dimensionality.
    method : str or KL method configuration, default="monte_carlo"
        Method used to compute or estimate the KL divergence. Passing a string
        runs that method with its defaults. Use a method configuration object,
        such as `MonteCarlo(sampling=DrawSamples(50_000, rng=0))`, for
        method-specific options.
    prefer_closed_form : bool, default=False
        If `True`, the function will attempt use closed form if both inputs are
        Gaussian, even if the user specified a different method.

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

    Require a closed-form expression:

    ```python
    result = kl_divergence(p, q, method="closed_form")
    ```

    Configure Monte Carlo sampling:

    ```python
    result = kl_divergence(
        p,
        q,
        method=MonteCarlo(sampling=DrawSamples(50_000, rng=0)),
    )

    Use precomputed samples:

    ```python
    samples = p.sample(50_000, rng=0)
    result = kl_divergence(p, q, method=MonteCarlo(sampling=UseSamples(samples)))
    ```


    """
    _validate_same_dimension(p, q)
    spec, options = KL_REGISTRY.resolve(method)
    if prefer_closed_form and isinstance(p, Gaussian) and isinstance(q, Gaussian):
        return kl_closed_form(p, q)

    match spec.name:
        case "monte_carlo":
            options = _cast_options(options, MonteCarlo)
            return kl_monte_carlo(
                p,
                q,
                sampling=options.sampling,
                target_standard_error=options.target_standard_error,
                max_samples=options.max_samples,
                batch_size=options.batch_size,
            )
        case "unscented":
            p = _require_unscented_input(p, spec.name)
            return kl_unscented(p, q)
        case "gaussian_approximation":
            options = _cast_options(options, GaussianApproximation)
            p, q = _require_gaussian_family_pair(p, q, spec.name)
            return kl_gaussian_approximation(p, q, approximation=options.approximation)
        case "closed_form":
            p, q = _require_gaussian_pair(p, q, spec.name)
            return kl_closed_form(p, q)
        case "variational":
            p, q = _require_gaussian_family_pair(p, q, spec.name)
            return kl_variational(p, q)
        case _:
            msg = "Unhandled KL method registry entry."
            raise AssertionError(msg)


def component_kl_matrix(
    p: Gaussian | GaussianMixture, q: Gaussian | GaussianMixture, /
) -> FloatArray:
    r"""Return pairwise Gaussian-component KL divergences.

    The returned matrix has shape `(p_components, q_components)`, where entry
    `(i, j)` is

    $$
    D_{\mathrm{KL}}\!\left(p_i \| q_j\right)
    $$

    for component `i` of `p` and component `j` of `q`. A single `Gaussian` is
    treated as a one-component Gaussian mixture.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Gaussian-family distributions whose components are compared.

    Returns
    -------
    FloatArray
        Pairwise component KL matrix.
    """
    _validate_same_dimension(p, q)
    _, p_means, p_covariances = p.component_arrays()
    _, q_means, q_covariances = q.component_arrays()
    return pairwise_gaussian_kl(p_means, p_covariances, q_means, q_covariances)


def symmetric_kl_divergence(
    p: Distribution,
    q: Distribution,
    /,
    *,
    method: KLMethod = "monte_carlo",
    prefer_closed_form: bool = False,
) -> DivergenceResult:
    r"""Compute the symmetric KL divergence between two distributions.

    Computes

    $$
    D_{\mathrm{SKL}}(p, q)
    =
    \frac{1}{2}
    \left[
        D_{\mathrm{KL}}(p \| q) + D_{\mathrm{KL}}(q \| p)
    \right].
    $$

    The same KL estimation method is used in both directions.

    Parameters
    ----------
    p, q : Distribution
        The two distributions to compare. They must have the same dimensionality.
    method : str or KL method configuration, default="monte_carlo"
        Method used for each directed KL estimate.
    prefer_closed_form : bool, default=False
        If `True`, each directed estimate will attempt to use closed form when
        both inputs are Gaussian.

    Returns
    -------
    DivergenceResult
        Result containing the symmetric KL value. For sampled methods,
        `num_samples` is the total sample count across both directed estimates
        when both counts are available.
    """
    forward = kl_divergence(p, q, method=method, prefer_closed_form=prefer_closed_form)
    reverse = kl_divergence(q, p, method=method, prefer_closed_form=prefer_closed_form)
    return DivergenceResult(
        value=0.5 * (forward.value + reverse.value),
        method="symmetric_kl",
        num_samples=_sum_num_samples(forward, reverse),
    )


def jensen_shannon_divergence(
    p: Gaussian | GaussianMixture,
    q: Gaussian | GaussianMixture,
    /,
    *,
    method: KLMethod = "monte_carlo",
    prefer_closed_form: bool = False,
) -> DivergenceResult:
    r"""Compute the Jensen-Shannon divergence between two Gaussian-family distributions.

    Computes

    $$
    D_{\mathrm{JS}}(p, q)
    =
    \frac{1}{2}D_{\mathrm{KL}}(p \| m)
    +
    \frac{1}{2}D_{\mathrm{KL}}(q \| m),
    \qquad
    m = \frac{1}{2}p + \frac{1}{2}q.
    $$

    For Gaussian and Gaussian-mixture inputs, the midpoint distribution `m` is
    represented as a Gaussian mixture using existing component-combination
    logic.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        The two Gaussian-family distributions to compare. They must have the
        same dimensionality.
    method : str or KL method configuration, default="monte_carlo"
        Method used for the two KL estimates against the midpoint mixture.
    prefer_closed_form : bool, default=False
        Passed through to `kl_divergence`. Note that the midpoint is generally
        a Gaussian mixture, so closed form is only available for methods and
        inputs that support it.

    Returns
    -------
    DivergenceResult
        Result containing the Jensen-Shannon divergence value. For sampled
        methods, `num_samples` is the total sample count across both directed
        estimates when both counts are available.
    """
    _validate_same_dimension(p, q)
    midpoint = combine_gaussians([p, q], weights=[0.5, 0.5])
    p_to_midpoint = kl_divergence(p, midpoint, method=method, prefer_closed_form=prefer_closed_form)
    q_to_midpoint = kl_divergence(q, midpoint, method=method, prefer_closed_form=prefer_closed_form)
    return DivergenceResult(
        value=0.5 * (p_to_midpoint.value + q_to_midpoint.value),
        method="jensen_shannon",
        num_samples=_sum_num_samples(p_to_midpoint, q_to_midpoint),
    )


def _sum_num_samples(*results: DivergenceResult) -> int | None:
    total = 0
    for result in results:
        if result.num_samples is None:
            return None
        total += result.num_samples
    return total


def _cast_options(options: object, option_type: type[OptionsT]) -> OptionsT:
    if not isinstance(options, option_type):
        msg = "Dispatcher returned an option object with the wrong type."
        raise TypeError(msg)
    return options


def _validate_same_dimension(p: Distribution, q: Distribution) -> None:
    if p.dim != q.dim:
        msg = f"Distribution dimensions must match, got {p.dim} and {q.dim}."
        raise ValueError(msg)


def _require_gaussian_pair(
    p: Distribution, q: Distribution, method: str
) -> tuple[Gaussian, Gaussian]:
    if not isinstance(p, Gaussian) or not isinstance(q, Gaussian):
        msg = (
            f"KL method '{method}' requires p and q to be Gaussian; "
            f"got {type(p).__name__} and {type(q).__name__}."
        )
        raise TypeError(msg)
    return p, q


def _require_gaussian_family_pair(
    p: Distribution, q: Distribution, method: str
) -> tuple[Gaussian | GaussianMixture, Gaussian | GaussianMixture]:
    if not isinstance(p, (Gaussian, GaussianMixture)) or not isinstance(
        q, (Gaussian, GaussianMixture)
    ):
        msg = (
            f"KL method '{method}' requires p and q to be Gaussian or GaussianMixture; "
            f"got {type(p).__name__} and {type(q).__name__}."
        )
        raise TypeError(msg)
    return p, q


def _require_unscented_input(p: Distribution, method: str) -> Gaussian | GaussianMixture:
    if not isinstance(p, (Gaussian, GaussianMixture)):
        msg = (
            f"KL method '{method}' requires p to be Gaussian or GaussianMixture; "
            f"got {type(p).__name__}."
        )
        raise TypeError(msg)
    return p
