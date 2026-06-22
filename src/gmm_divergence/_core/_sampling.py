from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_points, as_positive_sample_count, as_sample_batches

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
        return as_points(samples, n_features=dim, name="samples")
    return distribution.sample(n_samples=as_positive_sample_count(samples, name="samples"), rng=rng)


def resolve_sample_batches(
    distributions: Sequence[Distribution],
    samples: npt.ArrayLike | int | None,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Return provided sample batches or draw one batch from each distribution."""
    if isinstance(samples, int) or samples is None:
        n_samples = 10_000 if samples is None else as_positive_sample_count(samples, name="samples")
        rng = np.random.default_rng(rng)
        return np.asarray(
            [distribution.sample(n_samples, rng=rng) for distribution in distributions],
            dtype=np.float64,
        )

    expected_dim = distributions[0].dim if distributions else 0
    return as_sample_batches(
        samples, n_distributions=len(distributions), n_features=expected_dim, name="samples"
    )
