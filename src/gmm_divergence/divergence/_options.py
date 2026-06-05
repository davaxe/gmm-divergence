from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


Approximation: TypeAlias = Literal["nearest", "moment_matching"]


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    r"""Monte Carlo KL estimator configuration.

    Estimates the KL divergence with samples from the reference distribution
    $p$:

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{X \sim p}
    \left[
        \log p(X) - \log q(X)
    \right].
    $$

    This is the most general estimator: it can be used whenever `p` can be
    sampled and both `p` and `q` can evaluate log densities.
    """

    sampling: npt.ArrayLike | int = 10_000
    """Samples from p, or the number of samples to draw from p."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""


@dataclass(frozen=True, slots=True)
class Unscented:
    r"""Unscented-transform KL estimator configuration.

    Estimates the KL divergence by replacing random samples from $p$ with
    deterministic sigma points generated from the Gaussian components of $p$.
    For each component, the sigma points are chosen to capture its mean and
    covariance structure, then the estimator averages

    $$
    \log p(x) - \log q(x)
    $$

    over those points. This can be useful when deterministic, low-sample-count
    estimates are preferred over Monte Carlo sampling.
    """


@dataclass(frozen=True, slots=True)
class GaussianApproximation:
    r"""Gaussian-approximation KL estimator configuration.

    Approximates one or both inputs with Gaussian summaries and then computes a
    Gaussian KL surrogate for

    $$
    D_{\mathrm{KL}}(p \| q).
    $$

    This is a fast heuristic rather than an exact Gaussian-mixture KL
    computation. It is mainly useful as a rough baseline or when a cheap
    approximation is preferable to a sampled estimate.

    The `"moment_matching"` strategy replaces each Gaussian mixture by a single
    Gaussian with the same mean and covariance, then computes the closed-form
    Gaussian KL divergence between those summaries. This preserves global first
    and second moments but discards multimodality.

    The `"nearest"` strategy computes pairwise closed-form KL divergences
    between Gaussian components and returns the smallest component-level value.
    This captures the closest local match between the mixtures but ignores
    mixture weights and the full mixture shape.
    """

    approximation: Approximation = "moment_matching"
    """Gaussian approximation strategy to use."""


@dataclass(frozen=True, slots=True)
class ClosedForm:
    r"""Closed-form Gaussian KL configuration.

    Computes the exact KL divergence between two Gaussian distributions:

    $$
    D_{\mathrm{KL}}(p \| q) =
    \frac{1}{2}
    \left[
        \mathrm{tr}(\Sigma_q^{-1}\Sigma_p)
        + (\mu_q-\mu_p)^\top\Sigma_q^{-1}(\mu_q-\mu_p)
        - d
        + \log\frac{\det\Sigma_q}{\det\Sigma_p}
    \right].
    $$

    This method is only valid when both inputs are single Gaussian
    distributions.
    """


EstimationMethod: TypeAlias = Literal[
    "monte_carlo", "unscented", "gaussian_approximation", "closed_form"
]
KLMethod: TypeAlias = EstimationMethod | MonteCarlo | Unscented | GaussianApproximation | ClosedForm
