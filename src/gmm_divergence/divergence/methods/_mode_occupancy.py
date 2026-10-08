from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    import numpy.typing as npt


def kl_mode_occupancy(
    p_w: npt.ArrayLike, q_w: npt.ArrayLike, /, *, epsilon: float = 0.0
) -> DivergenceResult:
    r"""Compute the KL between two discrete distributions based on their mode occupancy.

    Computes

    $$
    D_{\mathrm{KL}}(p \| q) = \sum_{k=1}^K \pi_{p,k} \log \frac{\pi_{p,k}}{\pi_{q,k}}
    $$

    where index $k$ represents the same category in both probability vectors.
    This is categorical KL, not KL between Gaussian mixture densities.

    Parameters
    ----------
    p_w : array-like
        Weights of the reference distribution.
    q_w : array-like
        Weights of the approximating distribution.
    epsilon : float, optional
        Finite, nonnegative additive smoothing amount, by default 0.0.
        Smoothing changes the distributions being compared.

    Returns
    -------
    DivergenceResult
        Result object containing the KL divergence based on mode occupancy.
    """
    p_w = np.asarray(p_w, dtype=np.float64)
    q_w = np.asarray(q_w, dtype=np.float64)
    if p_w.ndim != 1 or q_w.ndim != 1 or p_w.shape != q_w.shape or p_w.size == 0:
        msg = "Expected nonempty weight vectors of equal length."
        raise ValueError(msg)

    if not np.all(np.isfinite(p_w)) or not np.all(np.isfinite(q_w)):
        msg_0 = "Weights must be finite."
        raise ValueError(msg_0)

    if np.any(p_w < 0) or np.any(q_w < 0):
        msg_1 = "Weights must be nonnegative."
        raise ValueError(msg_1)

    if not np.isclose(p_w.sum(), 1.0) or not np.isclose(q_w.sum(), 1.0):
        msg_2 = "Weights must sum to one."
        raise ValueError(msg_2)

    if not np.isfinite(epsilon) or epsilon < 0:
        msg_3 = "epsilon must be finite and nonnegative."
        raise ValueError(msg_3)

    if epsilon > 0:
        if epsilon <= 1:
            p_w = (p_w + epsilon) / (1 + len(p_w) * epsilon)
            q_w = (q_w + epsilon) / (1 + len(q_w) * epsilon)
        else:
            # Equivalent smoothing avoids overflow for large finite epsilon.
            p_w = (p_w / epsilon + 1) / (1 / epsilon + len(p_w))
            q_w = (q_w / epsilon + 1) / (1 / epsilon + len(q_w))

    if np.any((p_w > 0) & (q_w == 0)):
        return DivergenceResult(value=np.inf, method="mode_occupancy")

    mask = p_w > 0
    kl = np.sum(p_w[mask] * (np.log(p_w[mask]) - np.log(q_w[mask])))
    return DivergenceResult(value=float(max(0.0, kl)), method="mode_occupancy")
