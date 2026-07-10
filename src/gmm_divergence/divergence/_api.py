"""Explicit public API for Gaussian-family divergences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gmm_divergence._core._numeric import pairwise_gaussian_kl
from gmm_divergence.distributions._combine import combine_gaussians
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.divergence._options import (
    ClosedForm,
    KLEstimator,
    MomentMatchedGaussian,
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
    from gmm_divergence.distributions._typing import GaussianLike


def kl_divergence(
    p: GaussianLike, q: GaussianLike, /, *, estimator: KLEstimator
) -> DivergenceResult:
    r"""Compute the directed Kullback--Leibler divergence ``KL(p || q)``.

    Computes

    $$
    D_{\mathrm{KL}}(p \| q)
    = \mathbb{E}_{X \sim p}[\log p(X) - \log q(X)].
    $$

    The estimator is always explicit: it is never replaced automatically when
    the inputs happen to be Gaussian. Select :class:`ClosedForm` for an exact
    Gaussian--Gaussian result, or a sampled or approximate estimator for
    mixtures.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Reference and comparison distributions. They must have equal
        dimensionality; the divergence is evaluated from ``p`` to ``q``.
    estimator : KLEstimator
        Explicit estimator configuration. ``ClosedForm`` requires two
        :class:`Gaussian` inputs. ``MonteCarlo`` samples from ``p``; therefore
        ``sampling.Samples`` must contain samples drawn from ``p``.

    Returns
    -------
    DivergenceResult
        Estimated value, estimator name, and—when applicable—sample count and
        Monte Carlo uncertainty statistics.

    """
    _validate_same_dimension(p, q)
    match estimator:
        case MonteCarlo():
            return kl_monte_carlo(p, q, estimator=estimator)
        case Unscented():
            return kl_unscented(p, q)
        case MomentMatchedGaussian(approximation=approximation):
            return kl_gaussian_approximation(p, q, approximation=approximation)
        case ClosedForm():
            p_gaussian, q_gaussian = _require_gaussian_pair(p, q)
            return kl_closed_form(p_gaussian, q_gaussian)
        case Variational():
            return kl_variational(p, q)


def component_kl_matrix(p: GaussianLike, q: GaussianLike, /) -> FloatArray:
    r"""Return pairwise closed-form KL divergences between components.

    The result has shape ``(p_components, q_components)`` and element
    ``(i, j)`` equals ``KL(p_i || q_j)``. A :class:`Gaussian` is treated as a
    one-component mixture.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Distributions whose Gaussian components are compared. They must have
        equal dimensionality.

    Returns
    -------
    FloatArray
        A ``float64`` diagnostic matrix. It is not the KL divergence between
        the full mixtures.

    """
    _validate_same_dimension(p, q)
    _, p_means, p_covariances = p.component_arrays()
    _, q_means, q_covariances = q.component_arrays()
    return pairwise_gaussian_kl(p_means, p_covariances, q_means, q_covariances)


def symmetric_kl_divergence(
    p: GaussianLike, q: GaussianLike, /, *, forward: KLEstimator, reverse: KLEstimator
) -> DivergenceResult:
    r"""Compute the symmetrized Kullback--Leibler divergence.

    Computes

    $$
    D_{\mathrm{SKL}}(p, q)
    = \frac{1}{2}\left[D_{\mathrm{KL}}(p \| q) + D_{\mathrm{KL}}(q \| p)\right].
    $$

    The directional estimators are intentionally separate. In particular,
    precomputed samples for ``forward`` must originate from ``p``, while those
    for ``reverse`` must originate from ``q``.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Distributions to compare. They must have equal dimensionality.
    forward : KLEstimator
        Estimator for ``KL(p || q)``.
    reverse : KLEstimator
        Estimator for ``KL(q || p)``.

    Returns
    -------
    DivergenceResult
        Average of the directed estimates. ``num_samples`` is the total only
        when both estimates report a sample count.

    """
    forward_result = kl_divergence(p, q, estimator=forward)
    reverse_result = kl_divergence(q, p, estimator=reverse)
    return DivergenceResult(
        value=0.5 * (forward_result.value + reverse_result.value),
        method="symmetric_kl",
        num_samples=_sum_num_samples(forward_result, reverse_result),
    )


def jensen_shannon_divergence(
    p: GaussianLike, q: GaussianLike, /, *, p_to_midpoint: KLEstimator, q_to_midpoint: KLEstimator
) -> DivergenceResult:
    r"""Compute the Jensen--Shannon divergence between two distributions.

    Computes

    $$
    D_{\mathrm{JS}}(p, q)
    = \frac{1}{2}D_{\mathrm{KL}}(p \| m)
    + \frac{1}{2}D_{\mathrm{KL}}(q \| m),
    \qquad m = \frac{1}{2}p + \frac{1}{2}q.
    $$

    The midpoint is represented as a Gaussian mixture. Each directed term has
    an independent estimator so that precomputed samples remain attached to
    the distribution from which they were drawn.

    Parameters
    ----------
    p, q : Gaussian or GaussianMixture
        Distributions to compare. They must have equal dimensionality.
    p_to_midpoint : KLEstimator
        Estimator for ``KL(p || m)``. Any supplied samples must be drawn from
        ``p``.
    q_to_midpoint : KLEstimator
        Estimator for ``KL(q || m)``. Any supplied samples must be drawn from
        ``q``.

    Returns
    -------
    DivergenceResult
        Jensen--Shannon estimate. ``num_samples`` is the total only when both
        directed estimators report a sample count.

    """
    _validate_same_dimension(p, q)
    midpoint = combine_gaussians([p, q], weights=[0.5, 0.5])
    p_result = kl_divergence(p, midpoint, estimator=p_to_midpoint)
    q_result = kl_divergence(q, midpoint, estimator=q_to_midpoint)
    return DivergenceResult(
        value=0.5 * (p_result.value + q_result.value),
        method="jensen_shannon",
        num_samples=_sum_num_samples(p_result, q_result),
    )


def _sum_num_samples(*results: DivergenceResult) -> int | None:
    if any(result.num_samples is None for result in results):
        return None
    return sum(result.num_samples for result in results if result.num_samples is not None)


def _validate_same_dimension(p: GaussianLike, q: GaussianLike) -> None:
    if p.dim != q.dim:
        msg = f"Gaussian-family distribution dimensions must match, got {p.dim} and {q.dim}."
        raise ValueError(msg)


def _require_gaussian_pair(p: GaussianLike, q: GaussianLike) -> tuple[Gaussian, Gaussian]:
    if not isinstance(p, Gaussian) or not isinstance(q, Gaussian):
        msg = f"ClosedForm requires Gaussian inputs, got {type(p).__name__} and {type(q).__name__}."
        raise TypeError(msg)
    return p, q
