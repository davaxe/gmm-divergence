from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence import Gaussian, GaussianMixture, fit_mixture_weights, prune_mixture
from gmm_divergence.fitting import (
    BidirectionalKL,
    JensenShannon,
    MomentMatching,
    SimplexSLSQP,
    SoftmaxLBFGSB,
)
from gmm_divergence.fitting._objectives import (
    forward_kl,
    jensen_shannon,
    moment_matching,
    reverse_kl,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.fitting import WeightFitMethod


def _finite_difference_gradient(
    objective: Callable[[FloatArray], tuple[float, FloatArray]], weights: FloatArray
) -> FloatArray:
    eps = 1e-6
    grad = np.empty_like(weights)
    for k in range(weights.shape[0]):
        step = np.zeros_like(weights)
        step[k] = eps
        value_plus, _ = objective(weights + step)
        value_minus, _ = objective(weights - step)
        grad[k] = (value_plus - value_minus) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    "method",
    [SoftmaxLBFGSB(tol=1e-10, max_iterations=200), SimplexSLSQP(tol=1e-10, max_iterations=200)],
)
def test_fit_mixture_weights_recovers_known_component_weights(
    method: SoftmaxLBFGSB | SimplexSLSQP,
) -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[-2.0], [1.5]], covariances=[[[0.5]], [[1.2]]]
    )
    candidates = [
        Gaussian.from_arrays(mean=[-2.0], covariance=[[0.5]]),
        Gaussian.from_arrays(mean=[1.5], covariance=[[1.2]]),
    ]

    result = fit_mixture_weights(
        p, candidates, method=method, objective=MomentMatching(fit_second_moments=True)
    )

    assert result.fit_objective == "moment_matching"
    assert result.converged is True
    assert result.weights == pytest.approx([0.25, 0.75], abs=1e-7)
    assert float(np.sum(result.weights)) == pytest.approx(1.0)
    assert result.objective_value < 1e-8
    assert result.forward_kl.value == pytest.approx(0.0, abs=1e-8)


def test_fit_mixture_weights_rejects_empty_or_incompatible_candidates() -> None:
    p = Gaussian.from_arrays(mean=[0.0], covariance=[[1.0]])
    q_wrong_dim = Gaussian.from_arrays(mean=[0.0, 1.0], covariance=np.eye(2))

    with pytest.raises(ValueError, match="q_i must contain at least one distribution"):
        _ = fit_mixture_weights(p, [], objective=MomentMatching())

    with pytest.raises(ValueError, match="must have the same dimensionality"):
        _ = fit_mixture_weights(p, [q_wrong_dim], objective=MomentMatching())

    with pytest.raises(ValueError, match="Unknown fit optimizer method"):
        _ = fit_mixture_weights(
            p,
            [p],
            method=cast("WeightFitMethod", cast("object", "unknown")),
            objective=MomentMatching(),
        )


def test_prune_mixture_removes_small_weights_and_keeps_valid_mixture() -> None:
    mixture = GaussianMixture.from_arrays(
        weights=[0.8, 0.00001, 0.19999],
        means=[[0.0], [10.0], [2.0]],
        covariances=[[[1.0]], [[1.0]], [[1.0]]],
    )

    pruned = prune_mixture(mixture, min_weight=1e-4)

    assert pruned.n_components == 2
    assert pruned.weights == pytest.approx([0.8 / 0.99999, 0.19999 / 0.99999])
    assert pruned.means[:, 0] == pytest.approx([0.0, 2.0])

    with pytest.raises(ValueError, match="All components were pruned"):
        _ = prune_mixture(mixture, min_weight=0.9)

    with pytest.raises(ValueError, match="min_weight must be a nonnegative finite value"):
        _ = prune_mixture(mixture, min_weight=-1.0)


def test_fit_options_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="tol must be a positive finite value"):
        _ = SoftmaxLBFGSB(tol=0.0)

    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        _ = SimplexSLSQP(max_iterations=0)

    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        _ = BidirectionalKL(alpha=1.5)


def test_fit_mixture_weights_accepts_jensen_shannon_objective() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[-2.0], [1.5]], covariances=[[[0.5]], [[1.2]]]
    )
    candidates = [
        Gaussian.from_arrays(mean=[-2.0], covariance=[[0.5]]),
        Gaussian.from_arrays(mean=[1.5], covariance=[[1.2]]),
    ]

    result = fit_mixture_weights(
        p,
        candidates,
        objective=JensenShannon(
            p_sampling=gd.sampling.Draw(2_000, rng=123), q_sampling=gd.sampling.Draw(2_000, rng=123)
        ),
    )

    assert result.fit_objective == "jensen_shannon"
    assert result.converged is True
    assert result.weights == pytest.approx([0.25, 0.75], abs=0.08)
    assert float(np.sum(result.weights)) == pytest.approx(1.0)

    default_result = fit_mixture_weights(p, candidates, objective="jensen_shannon")
    assert default_result.fit_objective == "jensen_shannon"


def test_fit_mixture_weights_accepts_stratified_candidate_sampling() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.25, 0.75], means=[[-2.0], [1.5]], covariances=[[[0.5]], [[1.2]]]
    )
    candidates = [
        GaussianMixture.from_components([Gaussian.from_arrays(mean=[-2.0], covariance=[[0.5]])]),
        GaussianMixture.from_components([Gaussian.from_arrays(mean=[1.5], covariance=[[1.2]])]),
    ]

    result = fit_mixture_weights(
        p,
        candidates,
        objective=JensenShannon(
            p_sampling=gd.sampling.Stratified(2_000, rng=123),
            q_sampling=gd.sampling.Stratified(2_000, rng=123),
        ),
    )

    assert result.fit_objective == "jensen_shannon"
    assert result.converged is True
    assert result.weights == pytest.approx([0.25, 0.75], abs=0.08)


def test_fit_objective_gradients_match_finite_differences() -> None:
    p = GaussianMixture.from_arrays(
        weights=[0.45, 0.55], means=[[-1.0], [1.3]], covariances=[[[0.6]], [[1.2]]]
    )
    candidates = [
        Gaussian.univariate(mean=-0.8, variance=0.7),
        Gaussian.univariate(mean=1.6, variance=1.1),
    ]
    p_samples = np.array([[-1.5], [-0.2], [0.4], [1.0], [2.2]], dtype=np.float64)
    q_samples = np.array(
        [[[-1.4], [-0.8], [0.0], [0.6]], [[0.8], [1.2], [1.8], [2.5]]], dtype=np.float64
    )
    weights = np.array([0.37, 0.63], dtype=np.float64)

    objectives = [
        forward_kl(p, candidates, p_samples),
        reverse_kl(p, candidates, q_samples),
        jensen_shannon(p, candidates, p_samples, q_samples),
        moment_matching(p, candidates, second_moments=True),
    ]

    for objective in objectives:
        _value, analytical = objective(weights)
        numerical = _finite_difference_gradient(objective, weights)
        assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-5)
