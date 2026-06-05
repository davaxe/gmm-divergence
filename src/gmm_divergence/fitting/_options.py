from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class SoftmaxLBFGSB:
    """L-BFGS-B optimizer over unconstrained softmax logits."""

    tol: float = 1e-8
    max_iterations: int = 1000


@dataclass(frozen=True, slots=True)
class SimplexSLSQP:
    """SLSQP optimizer over simplex-constrained weights."""

    tol: float = 1e-8
    max_iterations: int = 1000


@dataclass(frozen=True, slots=True)
class ForwardKL:
    """Forward KL fitting objective configuration."""

    sampling: npt.ArrayLike | int = 10_000
    rng: np.random.Generator | int | None = None


@dataclass(frozen=True, slots=True)
class ReverseKL:
    """Reverse KL fitting objective configuration."""

    p_sampling: npt.ArrayLike | int = 10_000
    q_sampling: npt.ArrayLike | int = 10_000
    rng: np.random.Generator | int | None = None


@dataclass(frozen=True, slots=True)
class BidirectionalKL:
    """Bidirectional KL fitting objective configuration."""

    p_sampling: npt.ArrayLike | int = 10_000
    q_sampling: npt.ArrayLike | int = 10_000
    alpha: float = 0.5
    rng: np.random.Generator | int | None = None


@dataclass(frozen=True, slots=True)
class MomentMatching:
    """Moment-matching fitting objective configuration."""

    fit_second_moments: bool = False


FitMethodName: TypeAlias = Literal["softmax-lbfgsb", "simplex-slsqp"]
FitObjective: TypeAlias = Literal["forward", "reverse", "bidirectional", "moment_matching"]
FitParameterization: TypeAlias = Literal["simplex", "softmax"]
WeightFitMethod: TypeAlias = FitMethodName | SoftmaxLBFGSB | SimplexSLSQP
WeightFitObjective: TypeAlias = (
    FitObjective | ForwardKL | ReverseKL | BidirectionalKL | MomentMatching
)
