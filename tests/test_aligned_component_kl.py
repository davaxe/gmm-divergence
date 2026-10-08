from __future__ import annotations

import numpy as np
import pytest

import gmm_divergence as gd


def test_joint_kl_matches_analytic_decomposition() -> None:
    p = gd.GaussianMixture.from_arrays([0.25, 0.75], [[0], [1]], [[[1]], [[4]]])
    q = gd.GaussianMixture.from_arrays([0.5, 0.5], [[1], [3]], [[[2]], [[1]]])
    result = gd.aligned_component_kl(p, q)
    component_kls = np.array([0.5 * np.log(2), 0.5 * (7 - np.log(4))])
    weight_kl = 0.25 * np.log(0.5) + 0.75 * np.log(1.5)
    weighted_component_kl = 0.25 * component_kls[0] + 0.75 * component_kls[1]
    assert isinstance(result, gd.AlignedKLResult)
    assert gd.aligned_component_kl is gd.divergence.aligned_component_kl
    np.testing.assert_allclose(result.component_kls, component_kls)
    assert result.weight_kl == pytest.approx(weight_kl)
    assert result.weighted_component_kl == pytest.approx(weighted_component_kl)
    assert result.value == pytest.approx(weight_kl + weighted_component_kl)
    reverse = gd.aligned_component_kl(q, p)
    assert reverse.value != pytest.approx(result.value)
    assert not result.component_kls.flags.writeable


def test_identical_mixtures_have_zero_joint_kl() -> None:
    mixture = gd.GaussianMixture.from_arrays([0.2, 0.8], [[-1], [2]], [[[1]], [[2]]])
    result = gd.aligned_component_kl(mixture, mixture)
    assert result.value == pytest.approx(0, abs=1e-12)
    assert result.weight_kl == 0
    np.testing.assert_allclose(result.component_kls, 0, atol=1e-12)


def test_shared_components_reduce_to_categorical_kl() -> None:
    components = [gd.Gaussian.univariate(-1), gd.Gaussian.univariate(1)]
    p = gd.GaussianMixture.from_components(components, [0.8, 0.2])
    q = gd.GaussianMixture.from_components(components, [0.5, 0.5])
    result = gd.aligned_component_kl(p, q)
    assert result.value == pytest.approx(
        gd.divergence.categorical_kl_divergence(p.weights, q.weights).value
    )
    assert result.weighted_component_kl == 0


def test_single_multivariate_component_matches_manual_gaussian_kl() -> None:
    p = gd.GaussianMixture.from_arrays([1], [[0, 1]], [np.diag([1, 4])])
    q = gd.GaussianMixture.from_arrays([1], [[2, -1]], [np.diag([2, 1])])
    expected = 0.5 * (np.log(2 / 4) + (1 + 4) / 2 + (4 + 4) - 2)
    result = gd.aligned_component_kl(p, q)
    assert result.weight_kl == 0
    assert result.value == pytest.approx(expected)
    np.testing.assert_allclose(result.component_kls, [expected])


def test_component_order_controls_correspondence() -> None:
    p = gd.GaussianMixture.from_arrays([0.5, 0.5], [[-1], [1]], [[[1]], [[1]]])
    swapped = p.select_components([1, 0])
    points = [[-2], [0], [2]]
    np.testing.assert_allclose(p.logpdf(points), swapped.logpdf(points))
    assert gd.aligned_component_kl(p, swapped).value == pytest.approx(2)


def test_shared_reordering_preserves_joint_kl() -> None:
    p = gd.GaussianMixture.from_arrays([0.2, 0.8], [[0], [1]], [[[1]], [[2]]])
    q = gd.GaussianMixture.from_arrays([0.6, 0.4], [[1], [2]], [[[2]], [[1]]])
    result = gd.aligned_component_kl(p, q)
    reordered = gd.aligned_component_kl(p.select_components([1, 0]), q.select_components([1, 0]))
    assert reordered.value == pytest.approx(result.value)
    np.testing.assert_allclose(reordered.component_kls, result.component_kls[::-1])


@pytest.mark.parametrize("q_weights", [[0, 1], [0.5, 0.5], [1, 0]])
def test_zero_weights_follow_support_conventions(q_weights: list[float]) -> None:
    p = gd.GaussianMixture.from_arrays([1, 0], [[0], [100]], [[[1]], [[1]]])
    q = gd.GaussianMixture.from_arrays(q_weights, [[0], [0]], [[[1]], [[1]]])
    with np.errstate(all="raise"):
        result = gd.aligned_component_kl(p, q)
    assert result.weighted_component_kl == 0
    expected = np.inf if q_weights[0] == 0 else -np.log(q_weights[0])
    assert result.value == pytest.approx(expected)


def test_mismatched_component_counts_rejected() -> None:
    p = gd.GaussianMixture.from_components([gd.Gaussian.univariate()])
    q = gd.GaussianMixture.from_components([gd.Gaussian.univariate(), gd.Gaussian.univariate(1)])
    with pytest.raises(ValueError, match="equal numbers of components"):
        _ = gd.aligned_component_kl(p, q)


def test_mismatched_feature_dimensions_rejected() -> None:
    p = gd.GaussianMixture.from_arrays([1], [[0]], [[[1]]])
    q = gd.GaussianMixture.from_arrays([1], [[0, 0]], [np.eye(2)])
    with pytest.raises(ValueError, match="equal feature dimensions"):
        _ = gd.aligned_component_kl(p, q)
