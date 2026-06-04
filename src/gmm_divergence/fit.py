from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from typing_extensions import TypedDict

from gmm_divergence.fitting import weights as wfit
from gmm_divergence.fitting._objective import FitObjective, FitParameterization

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from gmm_divergence.distribution.gaussian import Gaussian
    from gmm_divergence.distribution.gmm import GaussianMixture
    from gmm_divergence.results import KLFitResult

__all__ = ["FitObjective", "FitParameterization"]

FitMethod = Literal["softmax-lbfgsb", "simplex-slsqp"]


class _CommonArgs(TypedDict):
    p: Gaussian | GaussianMixture
    q_i: Sequence[Gaussian | GaussianMixture]
    p_sampling: npt.ArrayLike | int
    q_sampling: npt.ArrayLike | int
    objective: FitObjective
    alpha: float
    rng: np.random.Generator | int | None
    tol: float
    max_iterations: int


@overload
def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    objective: Literal["moment_matching"] = "moment_matching",
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult: ...


@overload
def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    p_sampling: npt.ArrayLike | int = 10_000,
    objective: Literal["forward"] = "forward",
    rng: np.random.Generator | int | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult: ...


@overload
def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    p_sampling: npt.ArrayLike | int = 10_000,
    q_sampling: npt.ArrayLike | int = 10_000,
    objective: Literal["reverse"] = "reverse",
    rng: np.random.Generator | int | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult: ...


@overload
def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    p_sampling: npt.ArrayLike | int = 10_000,
    q_sampling: npt.ArrayLike | int = 10_000,
    objective: Literal["bidirectional"] = "bidirectional",
    alpha: float = 0.5,
    rng: np.random.Generator | int | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult: ...


def fit_mixture_weights(
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    /,
    *,
    method: FitMethod = "softmax-lbfgsb",
    p_sampling: npt.ArrayLike | int = 10_000,
    q_sampling: npt.ArrayLike | int = 10_000,
    objective: FitObjective = "forward",
    alpha: float = 0.5,
    rng: np.random.Generator | int | None = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> KLFitResult:
    r"""Fit weights for a mixture of fixed candidate distributions.

    Fits nonnegative weights `w_i` such that the weighted mixture

    $$
    q_w(x) = \sum_i w_i q_i(x)
    $$

    approximates the reference distribution `p` according to the selected KL
    objective.

    Parameters
    ----------
    p : Gaussian or GaussianMixture
        Reference distribution.
    q_i : sequence of Gaussian or GaussianMixture
        Candidate distributions whose weights are fitted.
    method : {"softmax-lbfgsb", "simplex-slsqp"}
        Optimization method and parameterization used for the weights.
    p_sampling, q_sampling : int or array-like
        Number of samples to draw, or precomputed samples. `q_sampling` is used
        by reverse and bidirectional objectives.
    objective : {"forward", "reverse", "bidirectional"}, default="forward"
        KL objective used for fitting.
    alpha : float, default=0.5
        Forward-objective weight used by the bidirectional objective.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed used when sampling is required.
    tol : float, default=1e-8
        Optimization tolerance.
    max_iterations : int
        Maximum number of optimizer iterations.

    Returns
    -------
    KLFitResult
        Result containing the fitted weights, fitted mixture, fit objective,
        objective value, forward/reverse KL diagnostics, and optimizer metadata.
    """
    common_kwargs: _CommonArgs = {
        "p": p,
        "q_i": q_i,
        "p_sampling": p_sampling,
        "q_sampling": q_sampling,
        "objective": objective,
        "alpha": alpha,
        "rng": rng,
        "tol": tol,
        "max_iterations": max_iterations,
    }
    match method:
        case "softmax-lbfgsb":
            return wfit.fit_mixture_weights_softmax(**common_kwargs)
        case "simplex-slsqp":
            return wfit.fit_mixture_weights_simplex(**common_kwargs)
