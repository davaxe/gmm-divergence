from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from gmm_divergence._core._numeric import symmetrize
from gmm_divergence._core._validation import validate_positive_finite, validate_positive_int
from gmm_divergence.covariance._shape import validate_covariance_input

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray


@dataclass(frozen=True, slots=True, kw_only=True)
class RelativeToTrace:
    r"""Scale epsilon with the covariance trace.

    Sets

    $$
    \varepsilon = scale\,\frac{\mathrm{tr}(\Sigma)}{d},
    $$

    where $d$ is the covariance dimension.
    """

    scale: float = 1e-6
    r"""Multiplier $scale$ in $\varepsilon = scale\,\mathrm{tr}(\Sigma)/d$."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.scale, name="scale")


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetConditionNumber:
    r"""Choose epsilon to enforce a target condition number.

    For

    $$
    \Sigma_{\mathrm{reg}} = \Sigma + \varepsilon I,
    $$

    picks the smallest $\varepsilon \ge 0$ such that
    $\kappa(\Sigma_{\mathrm{reg}}) \le \text{max_condition_number}$.
    """

    max_condition_number: float = 1e8
    r"""Target upper bound $\kappa(\Sigma + \varepsilon I)$."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.max_condition_number, name="max_condition_number")
        if not isfinite(self.max_condition_number) or self.max_condition_number <= 1.0:
            msg = (
                f"max_condition_number must be a finite value greater than 1, got "
                f"{self.max_condition_number}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidualVariance:
    r"""Scale epsilon from discarded low-rank variance.

    With target rank $rank$, sets

    $$
    \varepsilon
    =
    scale\,\frac{1}{d-rank}\sum_{i=1}^{d-rank}\lambda_i^{\mathrm{disc}},
    $$

    where $\lambda_i^{\mathrm{disc}}$ are the discarded eigenvalues.
    """

    scale: float = 1.0
    r"""Multiplier $scale$ on the mean discarded eigenvalue."""
    rank: int | None = None
    r"""Target rank $rank$ used to define the discarded spectrum."""

    def __post_init__(self) -> None:
        _ = validate_positive_finite(self.scale, name="scale")
        if self.rank is not None:
            _ = validate_positive_int(self.rank, name="rank")


EpsilonHeuristic: TypeAlias = RelativeToTrace | TargetConditionNumber | ResidualVariance
EpsilonSpec: TypeAlias = float | EpsilonHeuristic


def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic
) -> float | FloatArray:
    """Estimate a diagonal-loading epsilon from covariance scale or spectrum.

    Parameters
    ----------
    covariance : array-like
        Covariance matrix with shape `(d, d)` or batch of matrices with shape
        `(n, d, d)`.
    heuristic : EpsilonHeuristic
        Explicit heuristic configuration used to estimate epsilon.

    Returns
    -------
    float or array
        Estimated epsilon value(s). If the input is a single covariance, a float
        is returned. If the input is a batch of covariances, an array of shape
        `(n,)` is returned with one epsilon per covariance.
    """
    covariance_arr: FloatArray = np.asarray(covariance, dtype=np.float64)
    _ = validate_covariance_input(covariance_arr)
    covariance_arr = symmetrize(covariance_arr)
    return epsilon_from_covariance(covariance_arr, _as_epsilon_heuristic(heuristic))


def epsilon_from_covariance(
    covariance: FloatArray, heuristic: EpsilonHeuristic
) -> float | FloatArray:
    """Evaluate an epsilon heuristic on validated, symmetric covariance input."""
    match heuristic:
        case RelativeToTrace(scale=scale):
            return _relative_trace(covariance, scale=scale)
        case TargetConditionNumber(max_condition_number=max_condition_number):
            return _target_condition_number(covariance, max_condition_number=max_condition_number)
        case ResidualVariance(scale=scale, rank=rank):
            return _residual_variance(covariance, scale=scale, rank=rank)


def _relative_trace(covariance: FloatArray, *, scale: float) -> float | FloatArray:
    _ = validate_positive_finite(scale, name="scale")
    average_variance = np.maximum(
        np.trace(covariance, axis1=-2, axis2=-1) / covariance.shape[-1], 0.0
    )
    return _epsilon_result(scale * average_variance)


def _target_condition_number(
    covariance: FloatArray, *, max_condition_number: float
) -> float | FloatArray:
    if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
        msg = (
            f"max_condition_number must be a finite value greater than 1, got "
            f"{max_condition_number}."
        )
        raise ValueError(msg)
    eigvals = np.linalg.eigvalsh(covariance)
    epsilon = np.maximum(
        (eigvals[..., -1] - max_condition_number * eigvals[..., 0]) / (max_condition_number - 1.0),
        0.0,
    )
    return _epsilon_result(epsilon)


def _residual_variance(
    covariance: FloatArray, *, scale: float, rank: int | None
) -> float | FloatArray:
    _ = validate_positive_finite(scale, name="scale")
    if rank is None:
        msg = "ResidualVariance.rank must be provided when using the residual_variance heuristic."
        raise ValueError(msg)
    rank = validate_positive_int(rank, name="rank")
    eigvals = np.linalg.eigvalsh(covariance)
    if rank > eigvals.shape[-1]:
        msg = f"rank must not exceed covariance dimension {eigvals.shape[-1]}, got {rank}."
        raise ValueError(msg)
    n_discarded = max(eigvals.shape[-1] - rank, 0)
    if n_discarded == 0:
        return _epsilon_result(np.zeros(covariance.shape[:-2], dtype=np.float64))
    discarded = np.maximum(eigvals[..., :n_discarded], 0.0)
    return _epsilon_result(scale * np.mean(discarded, axis=-1))


def _epsilon_result(value: npt.ArrayLike) -> float | FloatArray:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array


def _as_epsilon_heuristic(value: object) -> EpsilonHeuristic:
    if not isinstance(value, (RelativeToTrace, TargetConditionNumber, ResidualVariance)):
        msg = f"Unknown epsilon heuristic: {type(value).__name__}."
        raise TypeError(msg)
    return value
