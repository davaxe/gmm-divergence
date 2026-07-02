"""Candidate selection strategies for GMM fitting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, Protocol

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar, override

from gmm_divergence._core._sampling import Draw
from gmm_divergence._core._validation import validate_nonnegative_finite
from gmm_divergence.distributions._base import Distribution
from gmm_divergence.divergence._api import kl_divergence
from gmm_divergence.divergence._options import MonteCarlo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.divergence._options import KLMethod

DistributionT = TypeVar("DistributionT", bound=Distribution, default=Distribution)


@dataclass(frozen=True, slots=True)
class CandidateSelection(Generic[DistributionT]):
    selected_indices: npt.NDArray[np.int_]
    rejected_indices: npt.NDArray[np.int_]
    scores: FloatArray | None = None


class CandidateSelector(Protocol, Generic[DistributionT]):
    """Protocol for filtering candidate distributions during fitting."""

    def select(
        self, p: DistributionT, q_i: Sequence[DistributionT]
    ) -> CandidateSelection[DistributionT]:
        """Filter candidate distributions based on the target distribution."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _KLSelectorBase(CandidateSelector[DistributionT], ABC):
    direction: Literal["forward", "reverse", "bidirectional"] = "forward"
    alpha: float = 0.5
    kl_method: KLMethod = field(default=MonteCarlo(sampling=Draw(rng=0)))

    def __post_init__(self) -> None:
        _validate_kl_selector_base(direction=self.direction, alpha=self.alpha)

    def _kl_divergence(self, p: Distribution, q: Distribution) -> float:
        return kl_divergence(p, q, method=self.kl_method, prefer_closed_form=True).value

    def _compute_kl_values(self, p: DistributionT, q_i: Sequence[DistributionT]) -> FloatArray:
        """Compute pairwise KL divergence values between p and each q_i."""
        match self.direction:
            case "forward":
                return np.array([self._kl_divergence(p, q) for q in q_i], dtype=np.float64)
            case "reverse":
                return np.array([self._kl_divergence(q, p) for q in q_i], dtype=np.float64)
            case "bidirectional":
                forward = np.array([self._kl_divergence(p, q) for q in q_i], dtype=np.float64)
                reverse = np.array([self._kl_divergence(q, p) for q in q_i], dtype=np.float64)
                return self.alpha * forward + (1 - self.alpha) * reverse

    @override
    def select(
        self, p: DistributionT, q_i: Sequence[DistributionT]
    ) -> CandidateSelection[DistributionT]:
        kl_values = self._compute_kl_values(p, q_i)
        mask = self._select_mask(kl_values)
        return CandidateSelection(
            selected_indices=np.where(mask)[0],
            rejected_indices=np.where(~mask)[0],
            scores=kl_values,
        )

    @abstractmethod
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class ThresholdSelector(_KLSelectorBase[DistributionT]):
    """Select candidates whose KL score is at or below a fixed threshold."""

    threshold: float

    def __post_init__(self) -> None:
        _validate_kl_selector_base(direction=self.direction, alpha=self.alpha)
        validate_nonnegative_finite(self.threshold, name="threshold")

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence threshold."""
        return kl_values <= self.threshold


@dataclass(frozen=True, slots=True)
class ToleranceSelector(_KLSelectorBase[DistributionT]):
    """Select candidates within a tolerance of the best KL score.

    In absolute mode, candidates with `KL <= min(KL) + delta` are kept. In
    relative mode, candidates with `KL <= min(KL) + delta * abs(min(KL))` are
    kept, so `delta=0.5` means within 50% of the best score when the best score
    is positive.
    """

    delta: float
    mode: Literal["absolute", "relative"] = "absolute"

    def __post_init__(self) -> None:
        _validate_kl_selector_base(direction=self.direction, alpha=self.alpha)
        validate_nonnegative_finite(self.delta, name="delta")

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence delta."""
        min_kl = np.min(kl_values)
        return (
            kl_values <= (min_kl + self.delta)
            if self.mode == "absolute"
            else kl_values <= (1 + self.delta) * min_kl
        )


@dataclass(frozen=True, slots=True)
class TopKSelector(_KLSelectorBase[DistributionT]):
    k: int

    def __post_init__(self) -> None:
        _validate_kl_selector_base(direction=self.direction, alpha=self.alpha)
        if isinstance(self.k, bool) or self.k <= 0:
            msg = f"k must be positive, got {self.k}."
            raise ValueError(msg)

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on top-k KL divergence."""
        if self.k >= len(kl_values):
            return np.ones_like(kl_values, dtype=bool)
        threshold = np.partition(kl_values, self.k - 1)[self.k - 1]
        return kl_values <= threshold


@dataclass(frozen=True, slots=True)
class QuantileSelector(_KLSelectorBase[DistributionT]):
    quantile: float

    def __post_init__(self) -> None:
        _validate_kl_selector_base(direction=self.direction, alpha=self.alpha)
        if not 0 < self.quantile < 1:
            msg = f"quantile must be in (0, 1), got {self.quantile}."
            raise ValueError(msg)

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence quantile."""
        threshold = np.quantile(kl_values, self.quantile)
        return kl_values <= threshold


def _validate_kl_selector_base(
    *, direction: Literal["forward", "reverse", "bidirectional"], alpha: float
) -> None:
    if direction not in {"forward", "reverse", "bidirectional"}:
        msg = f"direction must be 'forward', 'reverse', or 'bidirectional', got {direction!r}."
        raise ValueError(msg)
    if direction == "bidirectional" and not 0 <= alpha <= 1:
        msg = f"alpha must be in [0, 1] for bidirectional KL, got {alpha}."
        raise ValueError(msg)
