from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._base import Distribution


def resolve_samples(
    distribution: Distribution,
    samples: npt.ArrayLike | int,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Return provided samples or draw samples from the distribution."""
    dim = distribution.dim
    if not isinstance(samples, int):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != dim:
            msg = f"Expected samples of shape (n_samples, n_features), got {samples.shape}"
            raise ValueError(msg)
        return samples
    return distribution.sample(n_samples=samples or 10_000, rng=rng)


def resolve_sample_batches(
    distributions: Sequence[Distribution],
    samples: npt.ArrayLike | int | None,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Return provided sample batches or draw one batch from each distribution."""
    if isinstance(samples, int) or samples is None:
        n_samples = samples or 10_000
        rng = np.random.default_rng(rng)
        return np.asarray(
            [distribution.sample(n_samples, rng=rng) for distribution in distributions],
            dtype=np.float64,
        )

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0] != len(distributions):
        msg = (
            "Expected sample batches with shape "
            f"({len(distributions)}, n_samples, n_features), got {samples.shape}."
        )
        raise ValueError(msg)

    expected_dim = distributions[0].dim if distributions else 0
    if samples.shape[2] != expected_dim:
        msg = (
            "Expected sample batches with feature dimension "
            f"{expected_dim}, got {samples.shape[2]}."
        )
        raise ValueError(msg)
    return samples
