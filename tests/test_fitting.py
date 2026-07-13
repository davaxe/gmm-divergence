from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

import gmm_divergence as gd
from gmm_divergence.fitting import CandidateSelection, CandidateSelector


def _fixture() -> tuple[gd.GaussianMixture, list[gd.Gaussian]]:
    candidates = [gd.Gaussian.univariate(-1.0), gd.Gaussian.univariate(1.0)]
    return gd.GaussianMixture.from_components(candidates, weights=[0.3, 0.7]), candidates


@dataclass(slots=True)
class _CountingSamples:
    samples: npt.NDArray[np.float64]
    calls: int = 0

    def sample(self, distribution: gd.distributions.GaussianLike) -> npt.NDArray[np.float64]:
        assert distribution.dim == self.samples.shape[1]
        self.calls += 1
        return self.samples


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


def test_prepared_fit_reuses_density_data_and_supports_warm_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p, candidates = _fixture()
    original_logpdf = gd.Gaussian.logpdf
    logpdf_calls = 0
    sampling = _CountingSamples(
        samples=np.array([[-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5]], dtype=np.float64)
    )

    def counting_logpdf(self: gd.Gaussian, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nonlocal logpdf_calls
        logpdf_calls += 1
        return original_logpdf(self, x)

    monkeypatch.setattr(gd.Gaussian, "logpdf", counting_logpdf)
    prepared = gd.fitting.prepare_mixture_weight_fit(
        p, candidates, objective=gd.fitting.ForwardKL(sampling=sampling)
    )

    assert prepared.n_active_candidates == 2
    assert sampling.calls == 1
    assert logpdf_calls == 2
    assert sampling.calls == 1

    weights = np.array([0.4, 0.6], dtype=np.float64)
    _, gradient = prepared.evaluate(weights)
    step = 1e-6
    finite_difference = np.empty_like(weights)
    for index in range(weights.shape[0]):
        lower = weights.copy()
        upper = weights.copy()
        lower[index] -= step
        upper[index] += step
        lower_value, _ = prepared.evaluate(lower)
        upper_value, _ = prepared.evaluate(upper)
        finite_difference[index] = (upper_value - lower_value) / (2.0 * step)
    assert gradient == pytest.approx(finite_difference, rel=1e-5)

    first_solution = prepared.solve(method=gd.fitting.SimplexSLSQP())
    warm_solution = prepared.solve(
        method=gd.fitting.SimplexSLSQP(initial_weights=first_solution.active_weights)
    )
    result = prepared.report(warm_solution)

    assert logpdf_calls == 2
    assert first_solution.converged
    assert warm_solution.converged
    assert not first_solution.parameters.flags.writeable
    assert not first_solution.active_weights.flags.writeable
    assert result.weights == pytest.approx(warm_solution.active_weights)


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
            mode="bad",  # pyright: ignore[reportArgumentType]
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
