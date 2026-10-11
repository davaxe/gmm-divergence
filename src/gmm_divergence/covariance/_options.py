from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

from gmm_divergence._core._validation import validate_positive_finite, validate_positive_int
from gmm_divergence._core._validation import validate_unit_interval as _validate_unit_interval
from gmm_divergence.covariance._epsilon import (
    EpsilonSpec,
    RelativeToTrace,
    ResidualVariance,
    TargetConditionNumber,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagonalLoading:
    r"""Diagonal loading regularization configuration.

    Applies the regularizer

    $$
    \Sigma_{\mathrm{reg}} = \Sigma + \varepsilon I,
    $$

    where $\Sigma$ is the input covariance, $I$ is the identity matrix, and
    $\varepsilon \ge 0$ is the diagonal loading amount.

    Parameters
    ----------
    epsilon : float or EpsilonHeuristic, default=1e-6
        Finite, nonnegative loading amount, or a heuristic evaluated per matrix.
        Zero is allowed but does not repair a singular covariance.
    """

    epsilon: EpsilonSpec = 1e-6
    """Diagonal loading amount or epsilon heuristic."""

    def __post_init__(self) -> None:
        _validate_epsilon_spec(self.epsilon, name="epsilon")


@dataclass(frozen=True, slots=True, kw_only=True)
class IsotropicShrinkage:
    r"""Linear shrinkage toward an isotropic covariance target.

    Applies the shrinkage

    $$
    \Sigma_{\mathrm{reg}}
    =
    (1-\alpha)\,\Sigma + \alpha\,\tau I,
    \qquad
    \tau = \frac{\mathrm{tr}(\Sigma)}{d},
    $$

    where $d$ is the dimensionality and $\tau I$ is an isotropic covariance
    with the same average marginal variance as $\Sigma$.

    Parameters
    ----------
    strength : float, default=1e-2
        Finite interpolation weight in [0, 1]. Zero leaves the covariance
        unchanged; one replaces it with its isotropic trace-matched target.
    """

    strength: float = 1e-2
    """Interpolation weight between the covariance and isotropic target."""

    def __post_init__(self) -> None:
        _ = _validate_unit_interval(self.strength, name="strength")


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagonalShrinkage:
    r"""Shrinkage toward the covariance diagonal.

    Applies the shrinkage

    $$
    \Sigma_{\mathrm{reg}}
    =
    (1-\alpha)\,\Sigma + \alpha\,\mathrm{diag}(\Sigma),
    $$

    where $\mathrm{diag}(\Sigma)$ is the diagonal matrix containing the
    marginal variances of $\Sigma$.

    Parameters
    ----------
    strength : float, default=1e-2
        Finite interpolation weight in [0, 1]. Zero leaves the covariance
        unchanged; one removes all off-diagonal covariance entries.
    """

    strength: float = 1e-2
    """Interpolation weight between the covariance and diagonal target."""

    def __post_init__(self) -> None:
        _ = _validate_unit_interval(self.strength, name="strength")


@dataclass(frozen=True, slots=True, kw_only=True)
class EigenvalueClipping:
    r"""Eigenvalue-clipping regularization configuration.

    Given an eigendecomposition

    $$
    \Sigma = Q \, \mathrm{diag}(\lambda_1, \ldots, \lambda_d) \, Q^T,
    $$

    this method returns

    $$
    \Sigma_{\mathrm{reg}}
    =
    Q \, \mathrm{diag}(\max(\lambda_i, \lambda_{\min})) \, Q^T,
    $$

    where $\lambda_{\min} > 0$ is the eigenvalue floor.

    Parameters
    ----------
    min_eigenvalue : float, default=1e-6
        Finite, strictly positive eigenvalue floor.
    """

    min_eigenvalue: float = 1e-6
    """Smallest allowed eigenvalue after clipping."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.min_eigenvalue, name="min_eigenvalue")


@dataclass(frozen=True, slots=True, kw_only=True)
class LowRank:
    r"""Low-rank covariance approximation with diagonal loading.

    For an eigendecomposition

    $$
    \Sigma = U \, \mathrm{diag}(\lambda_1, \ldots, \lambda_d) \, U^T,
    $$

    the approximation keeps the leading `rank` eigenpairs and returns

    $$
    \Sigma_{\mathrm{reg}}
    =
    U_r \, \Lambda_r \, U_r^T + \varepsilon I,
    $$

    where $U_r$ contains the leading eigenvectors, $\Lambda_r$ the
    corresponding eigenvalues, and $\varepsilon \ge 0$ is the diagonal loading
    applied after truncation.

    Parameters
    ----------
    rank : int, default=1
        Positive number of leading eigenpairs, at most the covariance dimension.
    epsilon : float or EpsilonHeuristic, default=1e-6
        Finite, nonnegative diagonal loading after truncation, or a heuristic
        evaluated on the input covariance. ResidualVariance inherits this rank
        when its rank is None; an explicit residual rank must match.

    Notes
    -----
    A truncated covariance needs sufficient positive loading to be positive
    definite. The public regularization operation rejects singular results.
    """

    rank: int = 1
    """Target rank for the approximation."""
    epsilon: EpsilonSpec = 1e-6
    """Diagonal loading amount or epsilon heuristic."""

    def __post_init__(self) -> None:
        _ = validate_positive_int(self.rank, name="rank")
        _validate_epsilon_spec(self.epsilon, name="epsilon")


CovarianceRegularizer: TypeAlias = (
    DiagonalLoading | IsotropicShrinkage | DiagonalShrinkage | EigenvalueClipping | LowRank
)


def _validate_epsilon_spec(value: EpsilonSpec, /, *, name: str) -> None:
    if isinstance(value, bool):
        msg = f"{name} must be a nonnegative finite value, got {value}."
        raise TypeError(msg)
    if isinstance(value, Real):
        value_float = float(value)
        if not isfinite(value_float) or value_float < 0.0:
            msg = f"{name} must be a nonnegative finite value, got {value}."
            raise ValueError(msg)
        return
    if isinstance(value, (RelativeToTrace, TargetConditionNumber, ResidualVariance)):
        return
    msg = f"{name} must be a nonnegative scalar or epsilon heuristic, got {type(value).__name__}."
    raise TypeError(msg)
