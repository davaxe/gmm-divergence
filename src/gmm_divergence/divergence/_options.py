from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


Approximation: TypeAlias = Literal["nearest", "moment_matching"]


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    """Monte Carlo KL estimator configuration."""

    sampling: npt.ArrayLike | int = 10_000
    rng: np.random.Generator | int | None = None


@dataclass(frozen=True, slots=True)
class Unscented:
    """Unscented-transform KL estimator configuration."""


@dataclass(frozen=True, slots=True)
class GaussianApproximation:
    """Gaussian-approximation KL estimator configuration."""

    approximation: Approximation = "moment_matching"


@dataclass(frozen=True, slots=True)
class ClosedForm:
    """Closed-form Gaussian KL configuration."""


EstimationMethod: TypeAlias = Literal[
    "monte_carlo", "unscented", "gaussian_approximation", "closed_form"
]
KLMethod: TypeAlias = EstimationMethod | MonteCarlo | Unscented | GaussianApproximation | ClosedForm
