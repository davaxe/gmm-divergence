from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from gmm_divergence import (
    Gaussian,
    GaussianMixture,
    MomentMatching,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    fit_mixture_weights,
    prune_mixture,
)

if TYPE_CHECKING:
    from gmm_divergence.fitting import WeightFitMethod


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
    pruned_without_explicit_renormalization = prune_mixture(
        mixture, min_weight=1e-4, renormalize=False
    )

    assert pruned.n_components == 2
    assert pruned.weights == pytest.approx([0.8 / 0.99999, 0.19999 / 0.99999])
    assert pruned.means[:, 0] == pytest.approx([0.0, 2.0])
    assert pruned_without_explicit_renormalization.n_components == 2
    assert float(np.sum(pruned_without_explicit_renormalization.weights)) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="All components were pruned"):
        _ = prune_mixture(mixture, min_weight=0.9)
