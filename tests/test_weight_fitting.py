from __future__ import annotations

import numpy as np
import pytest

from gmm_divergence import (
    BidirectionalKL,
    ForwardKL,
    GaussianMixture,
    ReverseKL,
    SimplexSLSQP,
    SoftmaxLBFGSB,
    fit_mixture_weights,
)

UNKNOWN_FIT_METHOD_MESSAGE = (
    r"Unknown fit optimizer method 'bad'\. "
    r"Supported methods are: simplex-slsqp, softmax-lbfgsb\."
)


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

    result = fit_mixture_weights(p, q_i, objective=ForwardKL(sampling=500, rng=9126))

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
        p, q_i, objective=ReverseKL(p_sampling=500, q_sampling=500, rng=9126)
    )

    assert result.converged
    assert result.fit_objective == "reverse"
    assert result.alpha is None
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)


def test_fit_mixture_weights_accepts_forward_reverse_objective() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(
        p, q_i, objective=BidirectionalKL(p_sampling=500, q_sampling=500, rng=9126)
    )

    assert result.converged
    assert result.fit_objective == "bidirectional"
    assert result.alpha == pytest.approx(0.5)
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)


def test_fit_string_method_and_objective_use_defaults() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(p, q_i, method="softmax-lbfgsb", objective="forward")

    assert result.converged
    assert result.fit_objective == "forward"


def test_fit_accepts_configured_optimizer() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(
        p,
        q_i,
        method=SimplexSLSQP(tol=1e-8, max_iterations=1000),
        objective=ForwardKL(sampling=500, rng=9126),
    )

    assert result.converged
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)


def test_fit_accepts_configured_softmax_optimizer() -> None:
    p, q_i = _p_and_q_i()

    result = fit_mixture_weights(
        p,
        q_i,
        method=SoftmaxLBFGSB(tol=1e-8, max_iterations=1000),
        objective=ForwardKL(sampling=500, rng=9126),
    )

    assert result.converged
    np.testing.assert_allclose(result.weights, [0.6, 0.4], atol=0.08)


def test_fit_rejects_unknown_method() -> None:
    p, q_i = _p_and_q_i()

    with pytest.raises(ValueError, match=UNKNOWN_FIT_METHOD_MESSAGE):
        _ = fit_mixture_weights(p, q_i, method="bad")  # pyright: ignore[reportArgumentType]


def test_fit_rejects_legacy_dispatcher_kwargs() -> None:
    p, q_i = _p_and_q_i()

    with pytest.raises(TypeError):
        fit_mixture_weights(p, q_i, p_sampling=500)  # pyright: ignore[reportCallIssue]
