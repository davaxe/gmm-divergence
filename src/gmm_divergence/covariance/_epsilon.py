from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, cast, overload

import numpy as np

from gmm_divergence._core._dispatch import MethodSpec, Registry

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


EpsilonMethodName: TypeAlias = Literal[
    "relative_trace", "target_condition_number", "residual_variance"
]
EpsilonMethod: TypeAlias = (
    EpsilonMethodName | RelativeToTrace | TargetConditionNumber | ResidualVariance
)
EpsilonSpec: TypeAlias = float | EpsilonMethod

OptionsT = TypeVar("OptionsT")

_EPSILON_REGISTRY = Registry(
    label="covariance epsilon heuristic",
    specs=(
        MethodSpec(name="relative_trace", option_type=RelativeToTrace, default=RelativeToTrace()),
        MethodSpec(
            name="target_condition_number",
            option_type=TargetConditionNumber,
            default=TargetConditionNumber(),
        ),
        MethodSpec(
            name="residual_variance", option_type=ResidualVariance, default=ResidualVariance()
        ),
    ),
)


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    method: EpsilonMethod = "relative_trace",
    batched: Literal[False] = False,
) -> float: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    method: EpsilonMethod = "relative_trace",
    batched: Literal[True],
) -> FloatArray: ...


@overload
def estimate_epsilon(
    covariance: npt.ArrayLike, /, *, method: EpsilonMethod = "relative_trace", batched: None = None
) -> float | FloatArray: ...


def estimate_epsilon(
    covariance: npt.ArrayLike,
    /,
    *,
    method: EpsilonMethod = "relative_trace",
    batched: bool | None = None,
) -> float | FloatArray:
    """Estimate a diagonal-loading epsilon from covariance scale or spectrum.

    Parameters
    ----------
    covariance : array-like
        Covariance matrix with shape `(d, d)` or batch of matrices with shape
        `(n, d, d)`.
    method : str or epsilon heuristic configuration, default="relative_trace"
        Heuristic used to estimate the epsilon value.
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
    shape_kind = _check_covariance_shape(covariance_arr, batched=batched)
    spec, options = _EPSILON_REGISTRY.resolve(method)

    match spec.name:
        case "relative_trace":
            options = _cast_options(options, RelativeToTrace)
            return _relative_trace(covariance_arr, c=options.c, batched=shape_kind)
        case "target_condition_number":
            options = _cast_options(options, TargetConditionNumber)
            return _target_condition_number(covariance_arr, kappa=options.kappa, batched=shape_kind)
        case "residual_variance":
            options = _cast_options(options, ResidualVariance)
            return _residual_variance(
                covariance_arr, c=options.c, rank=options.r, batched=shape_kind
            )
        case _:
            msg = "Unhandled covariance epsilon heuristic registry entry."
            raise AssertionError(msg)


def _relative_trace(
    covariance: FloatArray, *, c: float, batched: Literal["single", "batched"]
) -> float | FloatArray:
    _validate_positive_finite(c, name="c")
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
    _validate_positive_finite(c, name="c")
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
        discarded = cast("FloatArray", eigvals[:n_discarded])
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


def _validate_positive_finite(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        msg = f"{name} must be a positive finite value, got {value}."
        raise ValueError(msg)


def _cast_options(options: object, option_type: type[OptionsT]) -> OptionsT:
    if not isinstance(options, option_type):
        msg = "Dispatcher returned an option object with the wrong type."
        raise TypeError(msg)
    return options


def _check_covariance_shape(
    covariance: FloatArray, *, batched: bool | None = None
) -> Literal["single", "batched"]:
    if covariance.ndim == 2:
        if batched is True:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)
        if covariance.shape[0] != covariance.shape[1]:
            msg = f"Expected covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)
        return "single"

    if covariance.ndim == 3:
        if batched is False:
            msg = f"Expected single covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)
        if covariance.shape[1] != covariance.shape[2]:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)
        return "batched"

    msg = f"covariance must have shape (d, d) or (n, d, d), got {covariance.shape}."
    raise ValueError(msg)
