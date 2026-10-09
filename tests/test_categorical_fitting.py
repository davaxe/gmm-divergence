"""Tests for categorical mixture-weight fitting."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from gmm_divergence.divergence import categorical_kl_divergence
from gmm_divergence.fitting import SimplexSLSQP, SoftmaxLBFGSB, fit_categorical_mixture_weights
from gmm_divergence.results import CategoricalMixtureFitResult


@pytest.fixture(params=[SimplexSLSQP(tol=1e-10), SoftmaxLBFGSB(tol=1e-10)], ids=["slsqp", "lbfgsb"])
def method(request: pytest.FixtureRequest) -> SimplexSLSQP | SoftmaxLBFGSB:
    """Run fitting tests with both supported optimizers."""
    return request.param


@pytest.fixture
def candidates() -> npt.NDArray[np.float64]:
    """Three categorical distributions with three categories."""
    return np.array([[0.1, 0.6, 0.3], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]], dtype=np.float64)


def _assert_valid_result(
    result: CategoricalMixtureFitResult, target: npt.ArrayLike, candidates: npt.ArrayLike
) -> None:
    """Check common result invariants."""
    target = np.asarray(target, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)

    assert isinstance(result, CategoricalMixtureFitResult)
    assert result.converged

    assert result.weights.shape == (len(candidates),)
    assert result.fitted_probabilities.shape == target.shape

    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= -1e-12)

    np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    np.testing.assert_allclose(result.fitted_probabilities, result.weights @ candidates, atol=1e-10)

    np.testing.assert_allclose(result.fitted_probabilities.sum(), 1.0, atol=1e-10)

    assert np.all(result.fitted_probabilities >= -1e-12)

    # The reported objective must equal the actual forward KL.
    expected_kl = categorical_kl_divergence(target, result.fitted_probabilities).value

    np.testing.assert_allclose(result.objective_value, expected_kl, atol=1e-9, rtol=1e-7)

    assert result.objective_value >= -1e-9
    assert result.iterations >= 0
    assert isinstance(result.optimizer_message, str)
    assert result.optimizer_message


def test_recovers_known_mixture_weights(
    method: SimplexSLSQP | SoftmaxLBFGSB, candidates: npt.NDArray[np.float64]
) -> None:
    """Recover known weights when the target is an exact candidate mixture."""
    expected_weights = np.array([0.6, 0.2, 0.2])
    target = expected_weights @ candidates

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, expected_weights, atol=1e-3)

    np.testing.assert_allclose(result.fitted_probabilities, target, atol=1e-5)

    assert result.objective_value < 1e-8


def test_identical_target_and_candidate(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """Fitting an identical candidate should give zero KL."""
    target = np.array([0.2, 0.5, 0.3])
    candidates = target[None, :]

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, [1.0])
    np.testing.assert_allclose(result.fitted_probabilities, target)

    assert abs(result.objective_value) < 1e-10


def test_single_candidate_matches_categorical_kl(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """With one candidate, the optimized weight must be one."""
    target = np.array([0.2, 0.5, 0.3])
    candidate = np.array([0.4, 0.3, 0.3])
    candidates = candidate[None, :]

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    expected_kl = categorical_kl_divergence(target, candidate).value

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, [1.0])
    np.testing.assert_allclose(result.fitted_probabilities, candidate)

    np.testing.assert_allclose(result.objective_value, expected_kl, atol=1e-10)


def test_identical_candidates(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """Identical candidates should produce the same fitted distribution."""
    target = np.array([0.2, 0.5, 0.3])
    candidate = np.array([0.4, 0.3, 0.3])

    candidates = np.stack([candidate, candidate, candidate])

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.fitted_probabilities, candidate, atol=1e-10)

    # The individual candidate weights are not identifiable.
    # Only their sum and the fitted distribution are meaningful.
    np.testing.assert_allclose(result.weights.sum(), 1.0)


def test_target_outside_candidate_convex_hull(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """Find the best approximation when exact recovery is impossible."""
    target = np.array([0.8, 0.1, 0.1])

    candidates = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    # The first candidate is the optimum: it provides the largest
    # achievable mass on the target's dominant category.
    np.testing.assert_allclose(result.weights, [1.0, 0.0], atol=1e-3)

    np.testing.assert_allclose(result.fitted_probabilities, candidates[0], atol=1e-3)

    assert result.objective_value > 0.0


def test_zero_target_probability(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """Zero target probabilities must contribute zero to forward KL."""
    target = np.array([0.7, 0.3, 0.0])

    candidates = np.array([[0.7, 0.3, 0.0], [0.2, 0.3, 0.5]])

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.fitted_probabilities, target, atol=1e-5)

    assert result.objective_value < 1e-8


def test_zero_candidate_probability(method: SimplexSLSQP | SoftmaxLBFGSB) -> None:
    """Individual candidates may have zero mass on target categories."""
    target = np.array([0.5, 0.5, 0.0])

    candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-4)

    np.testing.assert_allclose(result.fitted_probabilities, target, atol=1e-5)

    assert result.objective_value < 1e-8


def test_unsupported_target_category() -> None:
    """Reject a target category absent from every candidate."""
    target = np.array([0.3, 0.4, 0.3])

    candidates = np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]])

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        _ = fit_categorical_mixture_weights(target, candidates, method=SimplexSLSQP())


def test_does_not_modify_inputs(
    method: SimplexSLSQP | SoftmaxLBFGSB, candidates: npt.NDArray[np.float64]
) -> None:
    """Fitting must not modify caller-owned arrays."""
    target = np.array([0.2, 0.5, 0.3])

    original_target = target.copy()
    original_candidates = candidates.copy()

    _ = fit_categorical_mixture_weights(target, candidates, method=method)

    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(candidates, original_candidates)


def test_accepts_python_lists() -> None:
    """The public API should accept general array-like inputs."""
    target = [0.5, 0.5]

    candidates = [[0.8, 0.2], [0.2, 0.8]]

    result = fit_categorical_mixture_weights(target, candidates, method=SimplexSLSQP())

    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-4)

    np.testing.assert_allclose(result.fitted_probabilities, target, atol=1e-5)


@pytest.mark.parametrize(
    "target",
    [[], [0.5], [[0.5, 0.5]], [0.0, 0.0], [-0.1, 1.1], [0.4, 0.4], [np.nan, 0.5], [np.inf, 0.5]],
)
def test_invalid_target_rejected(target: npt.ArrayLike) -> None:
    """Reject malformed target categorical distributions."""
    candidates = np.array([[0.5, 0.5], [0.2, 0.8]])

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        _ = fit_categorical_mixture_weights(target, candidates, method=SimplexSLSQP())


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [0.5, 0.5],
        [[0.5]],
        [[0.5, 0.5, 0.0]],
        [[0.4, 0.4]],
        [[-0.1, 1.1]],
        [[np.nan, 0.5]],
        [[np.inf, 0.5]],
        [[0.5, 0.5], [0.2, 0.2]],
        [[0.5, 0.5], [0.2, -0.2]],
    ],
)
def test_invalid_candidates_rejected(candidates: npt.ArrayLike) -> None:
    """Reject malformed candidate categorical distributions."""
    target = np.array([0.5, 0.5])

    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        _ = fit_categorical_mixture_weights(target, candidates, method=SimplexSLSQP())


def test_custom_slsqp_initial_weights(candidates: npt.NDArray[np.float64]) -> None:
    """Custom initial simplex weights should be supported."""
    target = np.array([0.2, 0.5, 0.3])

    method = SimplexSLSQP(tol=1e-10, initial_weights=np.array([0.2, 0.3, 0.5]))

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, [0.6, 0.2, 0.2], atol=1e-3)

    assert result.fit_method is method


def test_custom_lbfgsb_initial_logits(candidates: npt.NDArray[np.float64]) -> None:
    """Custom initial softmax logits should be supported."""
    target = np.array([0.2, 0.5, 0.3])

    method = SoftmaxLBFGSB(tol=1e-10, initial_logits=np.array([-1.0, 0.0, 1.0]))

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    _assert_valid_result(result, target, candidates)

    np.testing.assert_allclose(result.weights, [0.6, 0.2, 0.2], atol=1e-3)

    assert result.fit_method is method


def test_both_optimizers_agree(candidates: npt.NDArray[np.float64]) -> None:
    """Both optimizers should converge to the same fitted distribution."""
    target = np.array([0.2, 0.5, 0.3])

    slsqp = fit_categorical_mixture_weights(target, candidates, method=SimplexSLSQP(tol=1e-10))

    lbfgsb = fit_categorical_mixture_weights(target, candidates, method=SoftmaxLBFGSB(tol=1e-10))

    assert slsqp.converged
    assert lbfgsb.converged

    np.testing.assert_allclose(slsqp.fitted_probabilities, lbfgsb.fitted_probabilities, atol=1e-5)

    np.testing.assert_allclose(slsqp.objective_value, lbfgsb.objective_value, atol=1e-8)


def test_optimizer_metadata(candidates: npt.NDArray[np.float64]) -> None:
    """The result should preserve optimizer configuration and metadata."""
    target = np.array([0.2, 0.5, 0.3])
    method = SimplexSLSQP(tol=1e-10)

    result = fit_categorical_mixture_weights(target, candidates, method=method)

    assert result.fit_method is method
    assert result.converged is True
    assert isinstance(result.iterations, int)
    assert result.iterations >= 0
    assert isinstance(result.optimizer_message, str)
    assert len(result.optimizer_message) > 0
