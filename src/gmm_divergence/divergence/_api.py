from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from gmm_divergence._core._dispatch import MethodSpec, Registry
from gmm_divergence.distributions import Gaussian, GaussianMixture
from gmm_divergence.divergence._options import (
    ClosedForm,
    GaussianApproximation,
    KLMethod,
    MonteCarlo,
    Unscented,
)
from gmm_divergence.divergence.methods._closed_form import kl_closed_form
from gmm_divergence.divergence.methods._gaussian_approx import kl_gaussian_approximation
from gmm_divergence.divergence.methods._monte_carlo import kl_monte_carlo
from gmm_divergence.divergence.methods._unscented import kl_unscented

if TYPE_CHECKING:
    from gmm_divergence.distributions._base import Distribution
    from gmm_divergence.results import DivergenceResult

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
    ),
)


def kl_divergence(
    p: Distribution, q: Distribution, /, *, method: KLMethod = "monte_carlo"
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
        such as `MonteCarlo(sampling=50_000, rng=0)`, for method-specific
        options.

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
    samples = p.sample(50_000, rng=0)
    result = kl_divergence(p, q, method=MonteCarlo(sampling=samples))
    ```


    """
    _validate_same_dimension(p, q)
    spec, options = KL_REGISTRY.resolve(method)
    match spec.name:
        case "monte_carlo":
            options = _cast_options(options, MonteCarlo)
            return kl_monte_carlo(p, q, sampling=options.sampling, rng=options.rng)
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
        case _:
            msg = "Unhandled KL method registry entry."
            raise AssertionError(msg)


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
