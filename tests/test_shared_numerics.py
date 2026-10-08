from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import gmm_divergence as gd
from gmm_divergence._core._sampling import sample_batches_with_weights, stratified_mixture_samples

if TYPE_CHECKING:
    import numpy.typing as npt


@pytest.mark.parametrize("dim", [1, 3, 5])
def test_scalar_pairwise_and_aligned_kl_agree_with_independent_formula(dim: int) -> None:
    rng = np.random.default_rng(dim)
    components: list[gd.Gaussian] = []
    for _ in range(5):
        factor = rng.normal(size=(dim, dim))
        components.append(
            gd.Gaussian.from_arrays(rng.normal(size=dim), factor @ factor.T + np.eye(dim))
        )
    p = gd.GaussianMixture.from_components(components[:2], weights=[0.3, 0.7])
    q = gd.GaussianMixture.from_components(components[2:], weights=[0.2, 0.5, 0.3])
    matrix = gd.divergence.component_kl_matrix(p, q)
    for i, p_component in enumerate(components[:2]):
        for j, q_component in enumerate(components[2:]):
            delta = q_component.mean - p_component.mean
            _, p_logdet = np.linalg.slogdet(p_component.covariance)
            _, q_logdet = np.linalg.slogdet(q_component.covariance)
            expected = 0.5 * (
                np.trace(np.linalg.solve(q_component.covariance, p_component.covariance))
                + delta @ np.linalg.solve(q_component.covariance, delta)
                - dim
                + q_logdet
                - p_logdet
            )
            actual = gd.kl_divergence(
                p_component, q_component, estimator=gd.divergence.ClosedForm()
            ).value
            assert actual == pytest.approx(expected, abs=1e-12)
            assert matrix[i, j] == pytest.approx(expected, abs=1e-12)
    aligned_q = gd.GaussianMixture.from_components(components[2:4], weights=[0.6, 0.4])
    aligned = gd.aligned_component_kl(p, aligned_q)
    np.testing.assert_allclose(
        aligned.component_kls, np.diag(gd.divergence.component_kl_matrix(p, aligned_q))
    )
    assert not aligned.component_kls.flags.writeable


def test_responsibilities_and_density_share_cached_component_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = [gd.Gaussian.univariate(-1, 0.5), gd.Gaussian.univariate(2, 2)]
    mixture = gd.GaussianMixture.from_components(components, weights=[0.4, 0.6])
    x = np.linspace(-4, 4, 17)[:, None]
    joint = np.column_stack([q.pdf(x) for q in components]) * mixture.weights
    expected_density = joint.sum(axis=1)
    expected_responsibilities = joint / expected_density[:, None]

    def fail_component(self: gd.GaussianMixture, index: int) -> gd.Gaussian:
        pytest.fail(f"Density evaluation reconstructed component {index} of {self!r}")

    monkeypatch.setattr(gd.GaussianMixture, "get_component", fail_component)
    np.testing.assert_allclose(mixture.pdf(x), expected_density)
    cached_chol = mixture.chol()
    np.testing.assert_allclose(mixture.responsibilities(x), expected_responsibilities)
    assert mixture.chol() is cached_chol


def test_stratified_metadata_and_batches_preserve_rng_and_component_mass() -> None:
    components = [gd.Gaussian.univariate(-2), gd.Gaussian.univariate(3), gd.Gaussian.univariate(8)]
    p = gd.GaussianMixture.from_components(components, weights=[0.99, 0.01, 0])
    q = gd.GaussianMixture.from_components(components, weights=[0.2, 0.7, 0.1])
    spec = gd.sampling.Stratified(7, rng=17)
    result = stratified_mixture_samples(p, spec)
    for k, weight in enumerate(p.weights):
        assert result.integration_weights[result.component_ids == k].sum() == pytest.approx(weight)
    assert result.counts[2] == 0
    assert np.isfinite(result.integration_weights).all()
    batches, integration_weights = sample_batches_with_weights([p, q], spec)
    np.testing.assert_array_equal(batches, spec.sample_batches([p, q]))
    assert integration_weights is not None
    np.testing.assert_allclose(integration_weights.sum(axis=1), [1, 1])
    # Batch generation consumes one generator across candidates rather than resetting it.
    rng = np.random.default_rng(17)
    first = stratified_mixture_samples(p, spec, rng=rng)
    second = stratified_mixture_samples(q, spec, rng=rng)
    np.testing.assert_array_equal(batches, np.stack([first.samples, second.samples]))


@pytest.mark.parametrize("probabilities", [[-5e-13, 1 + 5e-13], [0.5, 0.50001]])
def test_probability_policy_is_shared_across_divergence_fitting_and_statistics(
    probabilities: npt.ArrayLike,
) -> None:
    rows = np.asarray(probabilities, dtype=np.float64)[None, :]
    with pytest.raises(ValueError, match=r"nonnegative|sum to one"):
        _ = gd.divergence.categorical_kl_divergence(probabilities, [0.5, 0.5])
    with pytest.raises(ValueError, match=r"nonnegative|sum to one"):
        _ = gd.fitting.fit_categorical_mixture_weights(
            probabilities, [[0.5, 0.5]], method=gd.fitting.SoftmaxLBFGSB()
        )
    with pytest.raises(ValueError, match=r"nonnegative|sum to one"):
        _ = gd.fitting.fit_categorical_mixture_weights(
            [0.5, 0.5], rows, method=gd.fitting.SoftmaxLBFGSB()
        )
    with pytest.raises(ValueError, match=r"nonnegative|sum to one"):
        _ = gd.fitting.component_statistics([[0]], responsibilities=rows)


def test_probability_sum_tolerance_and_input_ownership_are_consistent() -> None:
    probabilities = np.array([0.5, 0.500000005])
    before = probabilities.copy()
    assert np.isfinite(gd.divergence.categorical_kl_divergence(probabilities, [0.5, 0.5]).value)
    fit = gd.fitting.fit_categorical_mixture_weights(
        probabilities, [[0.5, 0.5]], method=gd.fitting.SoftmaxLBFGSB()
    )
    assert fit.converged
    stats = gd.fitting.component_statistics([[0]], responsibilities=probabilities[None, :])
    assert np.isfinite(stats.weights).all()
    np.testing.assert_array_equal(probabilities, before)
    assert probabilities.flags.writeable


def test_custom_sampling_subclasses_are_still_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomDraw(gd.sampling.Draw):
        pass

    def malformed_batches(self: CustomDraw, distributions: object) -> npt.NDArray[np.float64]:
        del self, distributions
        return np.empty((1, 0, 1), dtype=np.float64)

    monkeypatch.setattr(CustomDraw, "sample_batches", malformed_batches)
    with pytest.raises(ValueError, match="at least one sample"):
        _ = sample_batches_with_weights([gd.Gaussian.univariate()], CustomDraw())
