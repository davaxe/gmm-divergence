from __future__ import annotations

import numpy as np
import pytest

from gmm_divergence import GaussianMixture, fit_mixture_weights


def _p_and_q_i() -> tuple[GaussianMixture, list[GaussianMixture]]:
    p = GaussianMixture.from_arrays(
        weights=[0.6, 0.4], means=[[0.0], [2.0]], covariances=[[[0.5]], [[0.5]]]
    )
    q_i = [
        GaussianMixture.from_arrays(weights=[1.0], means=[[0.0]], covariances=[[[0.5]]]),
        GaussianMixture.from_arrays(weights=[1.0], means=[[2.0]], covariances=[[[0.5]]]),
    ]
    return p, q_i


def test_fit_mixture_weights_result_reports_fit_objective_and_diagnostics() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(p, q_i, objective="forward", p_sampling=500, rng=9126)

    assert result.fit_objective == "forward"
    assert result.alpha is None
    assert np.isfinite(result.objective_value)
    assert result.forward_kl.method == "monte_carlo"
    assert result.reverse_kl.method == "monte_carlo"
    assert result.forward_kl.num_samples == 500
    assert result.reverse_kl.num_samples == 500


def test_fit_mixture_weights_accepts_reverse_objective() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(
        p, q_i, objective="reverse", p_sampling=500, q_sampling=500, rng=9126
    )

    assert result.converged
    assert result.fit_objective == "reverse"
    assert result.alpha is None
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)


def test_fit_mixture_weights_accepts_forward_reverse_objective() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(
        p, q_i, objective="bidirectional", p_sampling=500, q_sampling=500, rng=9126
    )

    assert result.converged
    assert result.fit_objective == "bidirectional"
    assert result.alpha == pytest.approx(0.5)
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)
