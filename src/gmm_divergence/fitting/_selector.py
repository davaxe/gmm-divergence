"""Candidate selection strategies for GMM fitting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._sampling import Samples
from gmm_divergence._core._validation import (
    validate_nonnegative_finite,
    validate_positive_int,
    validate_unit_interval,
)
from gmm_divergence.divergence._api import kl_divergence
from gmm_divergence.divergence._options import KLEstimator, MonteCarlo

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._typing import GaussianDistribution


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    scores: FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_indices", _freeze_indices(self.selected_indices))
        object.__setattr__(self, "rejected_indices", _freeze_indices(self.rejected_indices))
        if self.scores is not None:
            scores = readonly_copy(self.scores, dtype=np.float64)
            object.__setattr__(self, "scores", scores)


class CandidateSelector(Protocol):
    """Protocol for filtering candidate distributions during fitting."""

    def select(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> CandidateSelection:
        """Filter candidate distributions based on the target distribution."""
        ...


def _freeze_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(index) if isinstance(index, np.integer) else index for index in indices)


class CandidateScorer(Protocol):
    """Custom scoring over the entire original candidate sequence."""

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardKLScore:
    """Score KL(target || candidate); samples belong to target."""

    estimator: KLEstimator

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        return np.array([
            kl_divergence(target, candidate, estimator=self.estimator).value
            for candidate in candidates
        ])


@dataclass(frozen=True, slots=True, kw_only=True)
class ReverseKLScore:
    """Score KL(candidate || target), with generated or per-candidate estimators."""

    estimators: KLEstimator | Sequence[KLEstimator]

    def __post_init__(self) -> None:
        if isinstance(self.estimators, Sequence):
            estimators = tuple(self.estimators)
            if not estimators:
                msg = "estimators must contain one estimator per original candidate."
                raise ValueError(msg)
            object.__setattr__(self, "estimators", estimators)
        elif isinstance(self.estimators, MonteCarlo) and isinstance(
            self.estimators.sampling, Samples
        ):
            msg = "precomputed reverse samples require one estimator per original candidate."
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        estimators = self.estimators
        if isinstance(estimators, Sequence):
            if len(estimators) != len(candidates):
                msg = "estimators must contain one estimator per original candidate."
                raise ValueError(msg)
        else:
            estimators = (estimators,) * len(candidates)
        return np.array([
            kl_divergence(candidate, target, estimator=estimator).value
            for candidate, estimator in zip(candidates, estimators, strict=True)
        ])


@dataclass(frozen=True, slots=True, kw_only=True)
class BidirectionalKLScore:
    """Weighted forward and reverse scores with independently configured sources."""

    forward: CandidateScorer
    reverse: CandidateScorer
    forward_weight: float = 0.5

    def __post_init__(self) -> None:
        _ = validate_unit_interval(self.forward_weight, name="forward_weight")
        _validate_scorer(self.forward)
        _validate_scorer(self.reverse)

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        if self.forward_weight == 1:
            return score_candidates(target, candidates, scorer=self.forward)
        if self.forward_weight == 0:
            return score_candidates(target, candidates, scorer=self.reverse)
        return self.forward_weight * score_candidates(target, candidates, scorer=self.forward) + (
            1 - self.forward_weight
        ) * score_candidates(target, candidates, scorer=self.reverse)


def _validate_scorer(scorer: CandidateScorer) -> None:
    if not callable(getattr(scorer, "score", None)):
        msg = "scorer must implement score(target, candidates)."
        raise TypeError(msg)


def score_candidates(
    target: GaussianDistribution,
    candidates: Sequence[GaussianDistribution],
    *,
    scorer: CandidateScorer,
) -> FloatArray:
    """Return one finite or infinite score per original candidate."""
    if not candidates:
        msg = "candidates must contain at least one distribution."
        raise ValueError(msg)
    if any(candidate.n_features != target.n_features for candidate in candidates):
        msg = "candidates must have the same n_features as target."
        raise ValueError(msg)
    _validate_scorer(scorer)
    scores = readonly_copy(scorer.score(target, candidates), dtype=np.float64)
    if scores.shape != (len(candidates),) or np.any(np.isnan(scores)):
        msg = "scorer must return one non-NaN score per candidate."
        raise ValueError(msg)
    return scores


def rank_candidates(
    target: GaussianDistribution,
    candidates: Sequence[GaussianDistribution],
    *,
    scorer: CandidateScorer,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Rank from lowest score to highest, retaining input order for ties."""
    if limit is not None:
        limit = validate_positive_int(limit, name="limit")
    scores = score_candidates(target, candidates, scorer=scorer)
    indices = np.argsort(scores, kind="stable")
    return [(int(index), float(scores[index])) for index in indices[:limit]]


@dataclass(frozen=True, slots=True, kw_only=True)
class _KLSelectorBase(CandidateSelector, ABC):
    scorer: CandidateScorer

    def __post_init__(self) -> None:
        _validate_scorer(self.scorer)

    def _compute_kl_values(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        """Compute pairwise KL divergence values between p and each q_i."""
        return score_candidates(target, candidates, scorer=self.scorer)

    @override
    def select(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> CandidateSelection:
        kl_values = self._compute_kl_values(target, candidates)
        mask = self._select_mask(kl_values)
        return CandidateSelection(
            selected_indices=tuple(int(index) for index in np.flatnonzero(mask)),
            rejected_indices=tuple(int(index) for index in np.flatnonzero(~mask)),
            scores=kl_values,
        )

    @abstractmethod
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdSelector(_KLSelectorBase):
    """Select candidates whose KL score is at or below a fixed threshold."""

    threshold: float

    def __post_init__(self) -> None:
        _validate_scorer(self.scorer)
        _ = validate_nonnegative_finite(self.threshold, name="threshold")

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence threshold."""
        return kl_values <= self.threshold


@dataclass(frozen=True, slots=True, kw_only=True)
class ToleranceSelector(_KLSelectorBase):
    """Select candidates within a tolerance of the best KL score.

    In absolute mode, candidates with scores at most the minimum score plus
    `delta` are kept. In relative mode, the tolerance is
    `delta * abs(minimum_score)`, so `delta=0.5` means within 50% of the best
    score when the best score is positive.
    """

    delta: float
    mode: Literal["absolute", "relative"] = "absolute"

    def __post_init__(self) -> None:
        _validate_scorer(self.scorer)
        _ = validate_nonnegative_finite(self.delta, name="delta")
        if self.mode not in {"absolute", "relative"}:
            msg = "mode must be 'absolute' or 'relative'."
            raise ValueError(msg)

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence delta."""
        min_kl = np.min(kl_values)
        return (
            kl_values <= (min_kl + self.delta)
            if self.mode == "absolute"
            else kl_values <= min_kl + self.delta * abs(min_kl)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TopKSelector(_KLSelectorBase):
    k: int

    def __post_init__(self) -> None:
        _validate_scorer(self.scorer)
        _ = validate_positive_int(self.k, name="k")

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on top-k KL divergence."""
        if self.k >= len(kl_values):
            return np.ones_like(kl_values, dtype=bool)
        selected = np.argsort(kl_values, kind="stable")[: self.k]
        mask = np.zeros_like(kl_values, dtype=bool)
        mask[selected] = True
        return mask


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantileSelector(_KLSelectorBase):
    quantile: float

    def __post_init__(self) -> None:
        _validate_scorer(self.scorer)
        _ = validate_unit_interval(self.quantile, name="quantile")
        if not 0 < self.quantile < 1:
            msg = f"quantile must be in (0, 1), got {self.quantile}."
            raise ValueError(msg)

    @override
    def _select_mask(self, kl_values: FloatArray) -> npt.NDArray[np.bool_]:
        """Predicate to filter candidates based on KL divergence quantile."""
        threshold = np.quantile(kl_values, self.quantile)
        return kl_values <= threshold
