from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_probabilities
from gmm_divergence.results import DivergenceResult

if TYPE_CHECKING:
    import numpy.typing as npt


def kl_categorical(
    p_w: npt.ArrayLike, q_w: npt.ArrayLike, /, *, epsilon: float = 0.0
) -> DivergenceResult:
    r"""Compute categorical KL between two probability vectors over shared categories.

    Computes

    $$
    D_{\mathrm{KL}}(p \| q) = \sum_{k=1}^K \pi_{p,k} \log \frac{\pi_{p,k}}{\pi_{q,k}}
    $$

    where index $k$ represents the same category in both probability vectors.
    This is categorical KL, not KL between Gaussian mixture densities.

    Parameters
    ----------
    p_w, q_w : array-like, shape (n_categories,)
        Nonempty, finite, nonnegative reference and comparison probabilities,
        summing to one within rtol=1e-7, atol=1e-8. Category order must align.
        Probabilities are not normalized or clipped during validation.
    epsilon : float, default=0.0
        Finite, nonnegative additive smoothing amount. Each vector becomes
        (probabilities + epsilon) / (1 + n_categories * epsilon).

    Returns
    -------
    DivergenceResult
        Exact categorical KL in nats. Without smoothing, a positive reference
        mass with zero comparison mass gives infinity. Tiny negative roundoff
        in the finite result is clipped to zero.
    """
    p_w = np.asarray(p_w, dtype=np.float64)
    q_w = np.asarray(q_w, dtype=np.float64)
    if p_w.ndim != 1 or q_w.ndim != 1 or p_w.shape != q_w.shape or p_w.size == 0:
        msg = "Expected nonempty weight vectors of equal length."
        raise ValueError(msg)

    p_w = as_probabilities(p_w)
    q_w = as_probabilities(q_w)

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
        return DivergenceResult(value=np.inf, divergence="categorical_kl")

    mask = p_w > 0
    kl = np.sum(p_w[mask] * (np.log(p_w[mask]) - np.log(q_w[mask])))
    return DivergenceResult(value=float(max(0.0, kl)), divergence="categorical_kl")
