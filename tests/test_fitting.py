from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

import gmm_divergence as gd
from gmm_divergence.fitting import CandidateSelection, CandidateSelector


def _fixture() -> tuple[gd.GaussianMixture, list[gd.Gaussian]]:
    candidates = [gd.Gaussian.univariate(-1.0), gd.Gaussian.univariate(1.0)]
    return gd.GaussianMixture.from_components(candidates, weights=[0.3, 0.7]), candidates


def test_explicit_fitting_configs_and_immutable_result() -> None:
    p, candidates = _fixture()
    result = gd.fit_mixture_weights(
        p,
        candidates,
        method=gd.fitting.SimplexSLSQP(initial_weights=[0.5, 0.5]),
        objective=gd.fitting.MomentMatching(fit_second_moments=True),
    )
    assert result.weights == pytest.approx([0.3, 0.7], abs=1e-6)
    assert result.active_candidate_indices == (0, 1)
    assert not result.weights.flags.writeable
    assert not hasattr(result, "scipy_result")


def test_fitting_validates_bounds_and_initial_values() -> None:
    p, candidates = _fixture()
    with pytest.raises(ValueError, match="min_weight"):
        _ = gd.fitting.SimplexSLSQP(min_weight=-1.0)
    with pytest.raises(ValueError, match="infeasible"):
        _ = gd.fit_mixture_weights(
            p,
            candidates,
            method=gd.fitting.SimplexSLSQP(min_weight=0.6),
            objective=gd.fitting.MomentMatching(),
        )
    with pytest.raises(ValueError, match="length 2"):
        _ = gd.fit_mixture_weights(
            p,
            candidates,
            method=gd.fitting.SoftmaxLBFGSB(initial_logits=[0.0]),
            objective=gd.fitting.MomentMatching(),
        )


def test_selectors_are_explicit_and_top_k_is_exact() -> None:
    p = gd.Gaussian.univariate()
    candidates = [gd.Gaussian.univariate(), gd.Gaussian.univariate()]
    estimator = gd.divergence.ClosedForm()
    selection = gd.fitting.TopKSelector(k=1, estimator=estimator).select(p, candidates)
    assert len(selection.selected_indices) == 1
    assert selection.selected_indices == (0,)
    with pytest.raises(ValueError, match="mode must be"):
        _ = gd.fitting.ToleranceSelector(
            delta=1.0,
            mode=cast("Literal['absolute', 'relative']", cast("object", "bad")),
            estimator=estimator,
        )


@dataclass(frozen=True, slots=True)
class _FirstOnly(CandidateSelector):
    @override
    def select(
        self, p: gd.distributions.GaussianLike, q_i: Sequence[gd.distributions.GaussianLike]
    ) -> CandidateSelection:
        assert p.dim == q_i[0].dim
        return CandidateSelection(selected_indices=(1,), rejected_indices=(0,))


def test_selection_result_aligns_weights_and_mapping_to_original_candidates() -> None:
    p, candidates = _fixture()
    result = gd.fit_mixture_weights(
        p,
        candidates,
        method=gd.fitting.SoftmaxLBFGSB(),
        objective=gd.fitting.MomentMatching(),
        candidate_selector=_FirstOnly(),
    )
    assert result.weights == pytest.approx([0.0, 1.0])
    assert result.active_candidate_indices == (1,)
    assert np.all(result.fitted_mixture.mapping.source_index == 1)


def test_candidate_selection_requires_a_complete_partition() -> None:
    p, candidates = _fixture()

    @dataclass(frozen=True, slots=True)
    class InvalidSelector(CandidateSelector):
        @override
        def select(
            self, p: gd.distributions.GaussianLike, q_i: Sequence[gd.distributions.GaussianLike]
        ) -> CandidateSelection:
            assert p.dim == q_i[0].dim
            return CandidateSelection(selected_indices=(0, 0), rejected_indices=())

    with pytest.raises(ValueError, match="complete non-overlapping"):
        _ = gd.fit_mixture_weights(
            p,
            candidates,
            method=gd.fitting.SoftmaxLBFGSB(),
            objective=gd.fitting.MomentMatching(),
            candidate_selector=InvalidSelector(),
        )
