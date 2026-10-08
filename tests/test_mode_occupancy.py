from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import gmm_divergence as gd

if TYPE_CHECKING:
    import numpy.typing as npt


def test_mode_occupancy_matches_manual_categorical_kl() -> None:
    result = gd.divergence.mode_occupancy_kl([0.8, 0.2], (0.5, 0.5))
    expected = 0.8 * np.log(0.8 / 0.5) + 0.2 * np.log(0.2 / 0.5)
    assert result.value == pytest.approx(expected)
    assert result.method == "mode_occupancy"
    assert result.num_samples is None
    assert result.monte_carlo_stats is None
    reverse = gd.divergence.mode_occupancy_kl([0.5, 0.5], [0.8, 0.2])
    assert reverse.value != pytest.approx(result.value)


@pytest.mark.parametrize("weights", [[1.0], [0.0, 1.0], [0.2, 0.8]])
def test_identical_occupancies_have_zero_kl(weights: npt.ArrayLike) -> None:
    assert gd.divergence.mode_occupancy_kl(weights, weights).value == 0


def test_zero_probability_conventions() -> None:
    with np.errstate(all="raise"):
        finite = gd.divergence.mode_occupancy_kl([0.0, 1.0], [0.5, 0.5])
        infinite = gd.divergence.mode_occupancy_kl([1.0, 0.0], [0.0, 1.0])
    assert finite.value == pytest.approx(np.log(2))
    assert infinite.value == np.inf


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 2.0])
def test_smoothing_preserves_inputs_and_matches_formula(epsilon: float) -> None:
    p = np.array([0.8, 0.2])
    q = np.array([0.5, 0.5])
    p_before, q_before = p.copy(), q.copy()
    p.setflags(write=False)
    q.setflags(write=False)
    result = gd.divergence.mode_occupancy_kl(p, q, epsilon=epsilon)
    smoothed_p = (p + epsilon) / (1 + 2 * epsilon)
    smoothed_q = (q + epsilon) / (1 + 2 * epsilon)
    expected = np.sum(smoothed_p * np.log(smoothed_p / smoothed_q))
    assert result.value == pytest.approx(expected)
    np.testing.assert_array_equal(p, p_before)
    np.testing.assert_array_equal(q, q_before)


def test_smoothing_removes_support_mismatch_and_handles_large_epsilon() -> None:
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        result = gd.divergence.mode_occupancy_kl([1, 0], [0, 1], epsilon=0.1)
        large = gd.divergence.mode_occupancy_kl([1, 0], [0, 1], epsilon=1e308)
    assert result.value == pytest.approx(np.log(11) / 1.2)
    assert large.value == 0


def test_tiny_probabilities_do_not_overflow_ratio() -> None:
    with np.errstate(all="raise"):
        result = gd.divergence.mode_occupancy_kl([1.0, 0.0], [1e-320, 1.0])
    assert np.isfinite(result.value)
    assert result.value == pytest.approx(-np.log(1e-320))


@pytest.mark.parametrize(
    ("p", "q", "message"),
    [
        ([], [], "nonempty"),
        ([[1]], [1], "weight vectors"),
        ([1], [0.5, 0.5], "equal length"),
        (1, 1, "weight vectors"),
        ([np.nan, 1], [0, 1], "finite"),
        ([0, 1], [np.inf, 0], "finite"),
        ([-0.1, 1.1], [0, 1], "nonnegative"),
        ([0, 1], [1.1, -0.1], "nonnegative"),
        ([0, 0], [0, 1], "sum to one"),
        ([0, 1], [1, 1], "sum to one"),
    ],
)
def test_invalid_occupancies_rejected(p: npt.ArrayLike, q: npt.ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ = gd.divergence.mode_occupancy_kl(p, q)


@pytest.mark.parametrize("epsilon", [-1.0, np.nan, np.inf])
def test_invalid_smoothing_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon must be finite and nonnegative"):
        _ = gd.divergence.mode_occupancy_kl([1], [1], epsilon=epsilon)
