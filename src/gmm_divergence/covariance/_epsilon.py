from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import numpy as np

from gmm_divergence._core._numeric import symmetrize
from gmm_divergence._core._validation import validate_positive_finite, validate_positive_int
from gmm_divergence.covariance._shape import validate_covariance_input

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray


@dataclass(frozen=True, slots=True)
class RelativeToTrace:
    r"""Scale epsilon with the covariance trace.

    Sets

    $$
    \varepsilon = c\,\frac{\mathrm{tr}(\Sigma)}{d},
    $$

    where $d$ is the covariance dimension.
    """

    c: float = 1e-6
    r"""Multiplier $c$ in $\varepsilon = c\,\mathrm{tr}(\Sigma)/d$."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.c, name="c")


@dataclass(frozen=True, slots=True)
class TargetConditionNumber:
    r"""Choose epsilon to enforce a target condition number.

    For

    $$
    \Sigma_{\mathrm{reg}} = \Sigma + \varepsilon I,
    $$

    picks the smallest $\varepsilon \ge 0$ such that
    $\kappa(\Sigma_{\mathrm{reg}}) \le \text{kappa}$.
    """

    kappa: float = 1e8
    r"""Target upper bound $\kappa(\Sigma + \varepsilon I)$."""

    def __post_init__(self) -> None:
        if not isfinite(self.kappa) or self.kappa <= 1.0:
            msg = f"kappa must be a finite value greater than 1, got {self.kappa}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ResidualVariance:
    r"""Scale epsilon from discarded low-rank variance.

    With target rank $r$, sets

    $$
    \varepsilon
    =
    c\,\frac{1}{d-r}\sum_{i=1}^{d-r}\lambda_i^{\mathrm{disc}},
    $$

    where $\lambda_i^{\mathrm{disc}}$ are the discarded eigenvalues.
    """

    c: float = 1.0
    r"""Multiplier $c$ on the mean discarded eigenvalue."""
    r: int | None = None
    r"""Target rank $r$ used to define the discarded spectrum."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.c, name="c")
        if self.r is not None:
            _ = validate_positive_int(self.r, name="r")


EpsilonHeuristic: TypeAlias = RelativeToTrace | TargetConditionNumber | ResidualVariance
EpsilonSpec: TypeAlias = float | EpsilonHeuristic


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic, batched: Literal[False] = False
) -> float: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic, batched: Literal[True]
) -> FloatArray: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic, batched: None = None
) -> float | FloatArray: ...


def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic, batched: bool | None = None
) -> float | FloatArray:
    """Estimate a diagonal-loading epsilon from covariance scale or spectrum.

    Parameters
    ----------
    covariance : array-like
        Covariance matrix with shape `(d, d)` or batch of matrices with shape
        `(n, d, d)`.
    heuristic : EpsilonHeuristic
        Explicit heuristic configuration used to estimate epsilon.
    batched : bool or None, default=None
        Whether to interpret the input as batched. If `None`, the shape is
        inferred from the input rank.

    Returns
    -------
    float or array
        Estimated epsilon value(s). If the input is a single covariance, a float
        is returned. If the input is a batch of covariances, an array of shape
        `(n,)` is returned with one epsilon per covariance.
    """
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    _ = validate_covariance_input(covariance_arr, batched=batched)
    covariance_arr = symmetrize(covariance_arr)
    return epsilon_from_covariance(covariance_arr, _as_epsilon_heuristic(heuristic))


def epsilon_from_covariance(
    covariance: FloatArray, heuristic: EpsilonHeuristic
) -> float | FloatArray:
    """Evaluate an epsilon heuristic on validated, symmetric covariance input."""
    match heuristic:
        case RelativeToTrace(c=c):
            return _relative_trace(covariance, c=c)
        case TargetConditionNumber(kappa=kappa):
            return _target_condition_number(covariance, kappa=kappa)
        case ResidualVariance(c=c, r=rank):
            return _residual_variance(covariance, c=c, rank=rank)


def _relative_trace(covariance: FloatArray, *, c: float) -> float | FloatArray:
    _ = validate_positive_finite(c, name="c")
    scale = np.maximum(np.trace(covariance, axis1=-2, axis2=-1) / covariance.shape[-1], 0.0)
    return _epsilon_result(c * scale)


def _target_condition_number(covariance: FloatArray, *, kappa: float) -> float | FloatArray:
    if not np.isfinite(kappa) or kappa <= 1.0:
        msg = f"kappa must be a finite value greater than 1, got {kappa}."
        raise ValueError(msg)
    eigvals = np.linalg.eigvalsh(covariance)
    eps = np.maximum((eigvals[..., -1] - kappa * eigvals[..., 0]) / (kappa - 1.0), 0.0)
    return _epsilon_result(eps)


def _residual_variance(covariance: FloatArray, *, c: float, rank: int | None) -> float | FloatArray:
    _ = validate_positive_finite(c, name="c")
    if rank is None:
        msg = "ResidualVariance.r must be provided when using the residual_variance heuristic."
        raise ValueError(msg)
    rank = validate_positive_int(rank, name="rank")
    eigvals = np.linalg.eigvalsh(covariance)
    n_discarded = max(eigvals.shape[-1] - rank, 0)
    if n_discarded == 0:
        return _epsilon_result(np.zeros(covariance.shape[:-2], dtype=np.float64))
    discarded = np.maximum(eigvals[..., :n_discarded], 0.0)
    return _epsilon_result(c * np.mean(discarded, axis=-1))


def _epsilon_result(value: npt.ArrayLike) -> float | FloatArray:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array


def _as_epsilon_heuristic(value: object) -> EpsilonHeuristic:
    if not isinstance(value, (RelativeToTrace, TargetConditionNumber, ResidualVariance)):
        msg = f"Unknown epsilon heuristic: {type(value).__name__}."
        raise TypeError(msg)
    return value
