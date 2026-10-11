"""Combine Gaussian-family distributions and preserve component provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._validation import as_weights, validate_nonnegative_finite
from gmm_divergence.distributions._mapping import MixtureMapping
from gmm_divergence.distributions._mixture import GaussianMixture

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence.distributions._typing import GaussianDistribution


def combine_distributions(
    sources: Sequence[GaussianDistribution], *, weights: npt.ArrayLike | None = None
) -> GaussianMixture:
    """Combine fixed distributions with nonnegative source weights.

    Coefficients are normalized; omitted weights give equal source mass.
    The returned mixture works directly with divergence and fitting APIs.
    Its mapping identifies immediate source indices and local component indices,
    even when an input source already carries provenance.
    """
    if not sources:
        msg = "sources must contain at least one distribution."
        raise ValueError(msg)
    if len({source.n_features for source in sources}) != 1:
        msg = "sources must have the same n_features."
        raise ValueError(msg)
    source_weights = as_weights(
        np.ones(len(sources)) if weights is None else weights,
        expected_length=len(sources),
        name="weights",
    )
    arrays = [source.component_arrays() for source in sources]
    return GaussianMixture(
        weights=np.concatenate([
            weight * parts[0] for weight, parts in zip(source_weights, arrays, strict=True)
        ]),
        means=np.concatenate([parts[1] for parts in arrays]),
        covariances=np.concatenate([parts[2] for parts in arrays]),
        mapping=MixtureMapping(
            source_index=np.concatenate([
                np.full(len(parts[0]), index, dtype=np.intp) for index, parts in enumerate(arrays)
            ]),
            local_component_index=np.concatenate([
                np.arange(len(parts[0]), dtype=np.intp) for parts in arrays
            ]),
        ),
    )


def prune_mixture(mixture: GaussianMixture, *, min_weight: float = 1e-4) -> GaussianMixture:
    """Remove components below a threshold, preserving provenance and normalizing weights."""
    _ = validate_nonnegative_finite(min_weight, name="min_weight")
    indices = np.flatnonzero(mixture.weights >= min_weight)
    if indices.size == 0:
        msg = "All components were pruned; decrease min_weight."
        raise ValueError(msg)
    return mixture.select_components(indices)
