from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import numpy as np

from gmm_divergence._core._validation import validate_positive_finite
from gmm_divergence.covariance._shape import check_covariance_shape

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
        validate_positive_finite(self.c, name="c")


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
        validate_positive_finite(self.c, name="c")
        if self.r is not None and (isinstance(self.r, bool) or self.r <= 0):
            msg = f"r must be a positive integer when provided, got {self.r}."
            raise ValueError(msg)


EpsilonHeuristic: TypeAlias = RelativeToTrace | TargetConditionNumber | ResidualVariance
EpsilonSpec: TypeAlias = float | EpsilonHeuristic


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    heuristic: EpsilonHeuristic,
    batched: Literal[False] = False,
) -> float: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    heuristic: EpsilonHeuristic,
    batched: Literal[True],
) -> FloatArray: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, heuristic: EpsilonHeuristic, batched: None = None
) -> float | FloatArray: ...


def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    heuristic: EpsilonHeuristic,
    batched: bool | None = None,
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
    shape_kind = check_covariance_shape(covariance_arr, batched=batched)
    match heuristic:
        case RelativeToTrace(c=c):
            return _relative_trace(covariance_arr, c=c, batched=shape_kind)
        case TargetConditionNumber(kappa=kappa):
            return _target_condition_number(covariance_arr, kappa=kappa, batched=shape_kind)
        case ResidualVariance(c=c, r=rank):
            return _residual_variance(covariance_arr, c=c, rank=rank, batched=shape_kind)


def _relative_trace(
    covariance: FloatArray, *, c: float, batched: Literal["single", "batched"]
) -> float | FloatArray:
    validate_positive_finite(c, name="c")
    if batched == "single":
        dim = covariance.shape[0]
        scale = max(float(np.trace(covariance) / dim), 0.0)
        return float(c * scale)

    dim = covariance.shape[1]
    scale = np.maximum(np.trace(covariance, axis1=1, axis2=2) / dim, 0.0)
    return (c * scale).astype(np.float64, copy=False)


def _target_condition_number(
    covariance: FloatArray, *, kappa: float, batched: Literal["single", "batched"]
) -> float | FloatArray:
    if not np.isfinite(kappa) or kappa <= 1.0:
        msg = f"kappa must be a finite value greater than 1, got {kappa}."
        raise ValueError(msg)

    symmetrized = _symmetrize(covariance)
    eigvals = np.linalg.eigvalsh(symmetrized)

    if batched == "single":
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])
        eps = max((lambda_max - kappa * lambda_min) / (kappa - 1.0), 0.0)
        return float(eps)

    lambda_min = eigvals[:, 0]
    lambda_max = eigvals[:, -1]
    eps = np.maximum((lambda_max - kappa * lambda_min) / (kappa - 1.0), 0.0)
    return eps.astype(np.float64, copy=False)


def _residual_variance(
    covariance: FloatArray, *, c: float, rank: int | None, batched: Literal["single", "batched"]
) -> float | FloatArray:
    validate_positive_finite(c, name="c")
    if rank is None:
        msg = "ResidualVariance.r must be provided when using the residual_variance heuristic."
        raise ValueError(msg)
    if rank <= 0:
        msg = f"rank must be a positive integer, got {rank}."
        raise ValueError(msg)

    symmetrized = _symmetrize(covariance)
    eigvals = np.linalg.eigvalsh(symmetrized)
    n_discarded = _n_discarded(eigvals.shape[-1], rank=rank)

    if batched == "single":
        discarded = eigvals[:n_discarded]
        if discarded.size == 0:
            return 0.0
        return float(c * np.mean(np.maximum(discarded, 0.0)))

    if n_discarded == 0:
        return np.zeros(covariance.shape[0], dtype=np.float64)
    discarded = np.maximum(eigvals[:, :n_discarded], 0.0)
    return (c * np.mean(discarded, axis=1)).astype(np.float64, copy=False)


def _n_discarded(dim: int, *, rank: int) -> int:
    return max(dim - min(rank, dim), 0)


def _symmetrize(covariance: FloatArray) -> FloatArray:
    return 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
