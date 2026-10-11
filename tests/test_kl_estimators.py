from __future__ import annotations

import numpy as np
import pytest

import gmm_divergence as gd


def test_explicit_estimators_and_closed_form_require_gaussians() -> None:
    p = gd.Gaussian.univariate(0.0, 1.0)
    q = gd.Gaussian.univariate(1.0, 2.0)

    closed_form = gd.kl_divergence(p, q, estimator=gd.divergence.ClosedForm())
    monte_carlo = gd.kl_divergence(
        p, q, estimator=gd.divergence.MonteCarlo(sampling=gd.sampling.Draw(n_samples=50_000, rng=1))
    )

    assert isinstance(closed_form.estimator, gd.divergence.ClosedForm)
    assert monte_carlo.value == pytest.approx(closed_form.value, abs=0.02)
    with pytest.raises(TypeError, match="ClosedForm requires"):
        _ = gd.kl_divergence(
            gd.GaussianMixture.from_components([p]), p, estimator=gd.divergence.ClosedForm()
        )


def test_adaptive_monte_carlo_and_configuration_validation() -> None:
    p = gd.Gaussian.univariate(0.0, 1.0)
    q = gd.Gaussian.univariate(0.2, 1.0)
    estimator = gd.divergence.MonteCarlo(
        sampling=gd.sampling.Draw(n_samples=10, rng=1),
        target_standard_error=0.1,
        max_samples=1_000,
        batch_size=10,
    )
    result = gd.kl_divergence(p, q, estimator=estimator)
    assert result.n_evaluations is not None
    assert 10 <= result.n_evaluations <= 1_000

    with pytest.raises(TypeError, match=r"requires sampling\.Draw"):
        _ = gd.divergence.MonteCarlo(
            sampling=gd.sampling.Samples([[0.0]]), target_standard_error=0.1, max_samples=10
        )


def test_component_matrix_and_estimator_validation() -> None:
    p = gd.Gaussian.univariate()
    q = gd.GaussianMixture.from_components([p, gd.Gaussian.univariate(1.0)])
    assert gd.component_kl_matrix(p, q).shape == (1, 2)
    with pytest.raises(ValueError, match="dimensions must match"):
        _ = gd.kl_divergence(p, gd.Gaussian.standard(2), estimator=gd.divergence.Unscented())


def test_monte_carlo_rejects_empty_precomputed_samples() -> None:
    with pytest.raises(ValueError, match="samples must be a nonempty"):
        _ = gd.divergence.MonteCarlo(sampling=gd.sampling.Samples(np.empty((0, 1))))


def test_stratified_monte_carlo_does_not_claim_zero_uncertainty_for_singletons() -> None:
    p = gd.GaussianMixture.from_components([
        gd.Gaussian.univariate(-1.0),
        gd.Gaussian.univariate(1.0),
    ])
    q = gd.Gaussian.univariate()
    result = gd.kl_divergence(
        p,
        q,
        estimator=gd.divergence.MonteCarlo(sampling=gd.sampling.Stratified(n_samples=2, rng=0)),
    )

    assert result.monte_carlo is not None
    assert result.monte_carlo.standard_error is None
