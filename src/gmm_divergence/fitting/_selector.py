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
    """Selected and rejected positions in the original candidate sequence.

    Parameters
    ----------
    selected_indices : tuple of int
        Retained positions, in the order used for fitting. Fitting requires at
        least one retained candidate.
    rejected_indices : tuple of int
        Rejected positions. Together with selected_indices, must partition the
        original candidate sequence without overlap when used in fitting.
    scores : array-like or None, default=None
        Optional scores in original candidate order, copied into a read-only
        float64 array. Built-in selectors supply one score per candidate.
    """

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
        """Select candidates for a fixed target.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Nonempty original sequence sharing the target's feature dimension.

        Returns
        -------
        CandidateSelection
            Selected and rejected indices forming a complete, non-overlapping
            partition of the original sequence. Selected order controls fitting order.
        """
        ...


def _freeze_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(index) if isinstance(index, np.integer) else index for index in indices)


class CandidateScorer(Protocol):
    """Custom scoring over the entire original candidate sequence."""

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        """Assign scores to the original candidate sequence.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Original candidate sequence with the same feature dimension as target.

        Returns
        -------
        ndarray, shape (n_candidates,)
            One score per original candidate in input order. Lower scores are preferred.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardKLScore:
    """Score directed KL from the target to each candidate.

    Parameters
    ----------
    estimator : KLEstimator
        Estimator configuration reused for every target-to-candidate call.
        Precomputed observations must come from the target. Generated observations
        follow the configured random stream semantics on each call.
    """

    estimator: KLEstimator

    def score(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> FloatArray:
        """Evaluate target-to-candidate KL with the configured estimator.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Original candidate sequence with the same feature dimension as target.

        Returns
        -------
        ndarray, shape (n_candidates,)
            One score per original candidate in input order. Lower scores are preferred.
        """
        return np.array([
            kl_divergence(target, candidate, estimator=self.estimator).value
            for candidate in candidates
        ])


@dataclass(frozen=True, slots=True, kw_only=True)
class ReverseKLScore:
    """Score directed KL from each candidate to the target.

    Parameters
    ----------
    estimators : KLEstimator or sequence of KLEstimator
        Reusable estimator configuration, or one estimator per original candidate
        in input order. MonteCarlo with precomputed Samples requires the sequence
        form so each batch originates from its own candidate. Generated estimators
        may be reused; generator state advances as candidates are scored.
    """

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
        """Evaluate candidate-to-target KL with candidate-specific sources.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Original candidate sequence with the same feature dimension as target.

        Returns
        -------
        ndarray, shape (n_candidates,)
            One score per original candidate in input order. Lower scores are preferred.
        """
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
    """Combine forward and reverse scores with separate sample sources.

    Parameters
    ----------
    forward : CandidateScorer
        Forward scorer, normally ForwardKLScore with target observations.
    reverse : CandidateScorer
        Reverse scorer, normally ReverseKLScore with candidate observations.
    forward_weight : float, default=0.5
        Finite weight in [0, 1]. Reverse scores receive 1 - forward_weight.
        At either endpoint, only the nonzero-weight scorer is evaluated.
    """

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
        """Evaluate the weighted combination of directional scores.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Original candidate sequence with the same feature dimension as target.

        Returns
        -------
        ndarray, shape (n_candidates,)
            One score per original candidate in input order. Lower scores are preferred.
        """
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
    """Compute scores for the original candidate sequence.

    Parameters
    ----------
    target : Gaussian or GaussianMixture
        Reference distribution.
    candidates : sequence of Gaussian or GaussianMixture
        Nonempty candidate sequence sharing the target's feature dimension.
    scorer : CandidateScorer
        Configuration or custom scorer implementing score(target, candidates).
        It must return one non-NaN score per candidate; infinite scores are allowed.

    Returns
    -------
    ndarray, shape (n_candidates,)
        Independent, read-only float64 scores in original input order.
    """
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
    """Rank candidates from lowest score to highest with stable ties.

    Parameters
    ----------
    target : Gaussian or GaussianMixture
        Reference distribution.
    candidates : sequence of Gaussian or GaussianMixture
        Nonempty candidate sequence sharing the target's feature dimension.
    scorer : CandidateScorer
        Explicit scoring configuration or custom scorer; lower scores are preferred.
    limit : int or None, default=None
        Positive maximum number of entries to return. None returns all candidates;
        values above the candidate count also return all entries.

    Returns
    -------
    list of tuple
        Original candidate indices and float scores in ascending score order.
        Equal scores retain original candidate order.
    """
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
        """Compute configured scores for every original candidate against target."""
        return score_candidates(target, candidates, scorer=self.scorer)

    @override
    def select(
        self, target: GaussianDistribution, candidates: Sequence[GaussianDistribution]
    ) -> CandidateSelection:
        """Partition candidates using the configured score rule.

        Parameters
        ----------
        target : Gaussian or GaussianMixture
            Reference distribution.
        candidates : sequence of Gaussian or GaussianMixture
            Nonempty original sequence sharing the target's feature dimension.

        Returns
        -------
        CandidateSelection
            Selected and rejected indices, each in original input order, with an
            independent read-only score array for all original candidates.
        """
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
    """Retain candidates with scores at or below a fixed threshold.

    Parameters
    ----------
    scorer : CandidateScorer
        Explicit scoring configuration; lower scores are preferred.
    threshold : float
        Finite, nonnegative upper score threshold, inclusive. Selection may be
        empty; fitting rejects a selection retaining no candidates.
    """

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
    """Retain candidates within a tolerance of the minimum score.

    Parameters
    ----------
    scorer : CandidateScorer
        Explicit scoring configuration; lower scores are preferred.
    delta : float
        Finite, nonnegative tolerance amount.
    mode : {"absolute", "relative"}, default="absolute"
        Absolute mode retains scores at most minimum_score + delta. Relative mode
        uses minimum_score + delta * abs(minimum_score), including for negative
        estimates. A zero minimum gives zero relative tolerance.
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
    """Retain up to k candidates with the lowest scores.

    Parameters
    ----------
    scorer : CandidateScorer
        Explicit scoring configuration; lower scores are preferred.
    k : int
        Positive maximum number to retain. Values above the candidate count retain
        all candidates. Ties favor earlier input positions. Retained indices are
        returned in original input order rather than score order.
    """

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
    """Retain candidates at or below an interpolated score quantile.

    Parameters
    ----------
    scorer : CandidateScorer
        Explicit scoring configuration; lower scores are preferred.
    quantile : float
        Finite quantile strictly between zero and one, passed to numpy.quantile.
        Ties are included, so the retained count need not equal this fraction of
        the candidate count.
    """

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
