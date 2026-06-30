"""Options for fitting candidate-mixture weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class SoftmaxLBFGSB:
    r"""L-BFGS-B optimizer over unconstrained softmax logits.

    The optimizer represents mixture weights through logits
    $\mathbf{z} \in \mathbb{R}^N$ and maps them to simplex weights with

    $$
    w_i = \frac{\exp(z_i)}{\sum_{j=1}^{N}\exp(z_j)}.
    $$

    This avoids explicit simplex constraints in the optimizer while still
    ensuring that all fitted weights are non-negative and sum to one.
    """

    tol: float = 1e-8
    """Optimizer convergence tolerance."""
    max_iterations: int = 1000
    """Maximum number of optimizer iterations."""

    def __post_init__(self) -> None:
        _validate_positive_float(self.tol, name="tol")
        _validate_positive_int(self.max_iterations, name="max_iterations")


@dataclass(frozen=True, slots=True)
class SimplexSLSQP:
    r"""SLSQP optimizer over simplex-constrained weights.

    The optimizer works directly with the mixture weights
    $\mathbf{w} \in \Delta_N$, where

    $$
    \Delta_N =
    \left\{
        \mathbf{w} \in \mathbb{R}^N :
        w_i \ge 0,\ \sum_{i=1}^{N} w_i = 1
    \right\}.
    $$

    This keeps the optimization variables interpretable but requires the
    optimizer to enforce the simplex equality and bound constraints.
    """

    tol: float = 1e-8
    """Optimizer convergence tolerance."""
    max_iterations: int = 1000
    """Maximum number of optimizer iterations."""

    def __post_init__(self) -> None:
        _validate_positive_float(self.tol, name="tol")
        _validate_positive_int(self.max_iterations, name="max_iterations")


@dataclass(frozen=True, slots=True)
class ForwardKL:
    r"""Forward KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{KL}}\!\left(p \,\|\, q_{\mathbf{w}}\right)
    =
    \mathbb{E}_{X \sim p}
    \left[
        \log p(X) - \log q_{\mathbf{w}}(X)
    \right],
    $$

    where $p$ is the reference distribution and $q_{\mathbf{w}}$ is the
    weighted combination of the candidate mixtures. Since $\log p(X)$ does
    not depend on $\mathbf{w}$, the implemented objective minimizes the
    negative expected log-density of $q_{\mathbf{w}}$ under samples from $p$.
    """

    sampling: npt.ArrayLike | int = 10_000
    """Samples from p, or the number of samples to draw from p."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _validate_sampling(self.sampling, name="sampling")


@dataclass(frozen=True, slots=True)
class ReverseKL:
    r"""Reverse KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, p\right)
    =
    \mathbb{E}_{X \sim q_{\mathbf{w}}}
    \left[
        \log q_{\mathbf{w}}(X) - \log p(X)
    \right].
    $$

    The implementation evaluates a fixed-sample estimator using samples from
    the reference distribution $p$ for diagnostics and samples from each
    candidate mixture $q_i$ for the reverse objective.
    """

    p_sampling: npt.ArrayLike | int = 10_000
    """Samples from p, or the number of samples to draw from p."""
    q_sampling: npt.ArrayLike | int = 10_000
    """Samples from each q_i, or the number to draw from each q_i."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _validate_sampling(self.p_sampling, name="p_sampling")
        _validate_sampling(self.q_sampling, name="q_sampling")


@dataclass(frozen=True, slots=True)
class BidirectionalKL:
    r"""Bidirectional KL fitting objective configuration.

    Fits the candidate-mixture weights by minimizing a weighted combination of
    forward and reverse KL objectives:

    $$
    \alpha\,D_{\mathrm{KL}}\!\left(p \,\|\, q_{\mathbf{w}}\right)
    +
    (1-\alpha)\,D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, p\right),
    \qquad \alpha \in [0, 1].
    $$

    Values of $\alpha$ closer to one emphasize coverage of $p$ by
    $q_{\mathbf{w}}$, while values closer to zero emphasize the reverse-KL
    objective.
    """

    p_sampling: npt.ArrayLike | int = 10_000
    """Samples from p, or the number of samples to draw from p."""
    q_sampling: npt.ArrayLike | int = 10_000
    """Samples from each q_i, or the number to draw from each q_i."""
    alpha: float = 0.5
    """Weight assigned to the forward KL term."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _validate_sampling(self.p_sampling, name="p_sampling")
        _validate_sampling(self.q_sampling, name="q_sampling")
        if not isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            msg = f"alpha must be in [0, 1], got {self.alpha}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class JensenShannon:
    r"""Jensen-Shannon fitting objective configuration.

    Fits the candidate-mixture weights by minimizing

    $$
    D_{\mathrm{JS}}\!\left(p, q_{\mathbf{w}}\right)
    =
    \frac{1}{2}D_{\mathrm{KL}}\!\left(p \,\|\, m_{\mathbf{w}}\right)
    +
    \frac{1}{2}D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \,\|\, m_{\mathbf{w}}\right),
    $$

    where

    $$
    m_{\mathbf{w}} = \frac{1}{2}p + \frac{1}{2}q_{\mathbf{w}}.
    $$

    This objective is symmetric and bounded, while still using fixed samples
    from `p` and each candidate mixture `q_i` during optimization.
    """

    p_sampling: npt.ArrayLike | int = 10_000
    """Samples from p, or the number of samples to draw from p."""
    q_sampling: npt.ArrayLike | int = 10_000
    """Samples from each q_i, or the number to draw from each q_i."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _validate_sampling(self.p_sampling, name="p_sampling")
        _validate_sampling(self.q_sampling, name="q_sampling")


@dataclass(frozen=True, slots=True)
class MomentMatching:
    r"""Moment-matching fitting objective configuration.

    Fits weights by matching moments of $q_{\mathbf{w}}$ to moments of the
    reference distribution $p$ instead of optimizing a sampled KL estimate.
    The first-moment objective compares mixture means,

    $$
    \left\|\mu_p - \mu_{q_{\mathbf{w}}}\right\|_2^2,
    $$

    and can optionally include second-moment information as well.
    """

    fit_second_moments: bool = False
    """Whether to include covariance information in the objective."""


FitMethodName: TypeAlias = Literal["softmax_lbfgsb", "simplex_slsqp"]
FitObjective: TypeAlias = Literal[
    "forward", "reverse", "bidirectional", "jensen_shannon", "moment_matching"
]
FitParameterization: TypeAlias = Literal["simplex", "softmax"]
WeightFitMethod: TypeAlias = FitMethodName | SoftmaxLBFGSB | SimplexSLSQP
WeightFitObjective: TypeAlias = (
    FitObjective | ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching
)


def _validate_positive_float(value: float, /, *, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be a positive finite value, got {value}."
        raise ValueError(msg)


def _validate_positive_int(value: int, /, *, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer, got {value}."
        raise ValueError(msg)


def _validate_sampling(value: npt.ArrayLike | int, /, *, name: str) -> None:
    if isinstance(value, int):
        _validate_positive_int(value, name=name)
