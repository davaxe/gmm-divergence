"""Sampling records, protocols, and validated integration helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._arrays import readonly_copy
from gmm_divergence._core._validation import as_points, as_weights, validate_positive_int
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._typing import GaussianDistribution


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SampleBatch:
    """Owned observations (N, D), optional normalized integration weights and strata."""

    samples: FloatArray
    integration_weights: FloatArray | None
    component_ids: npt.NDArray[np.intp] | None

    def __init__(
        self,
        samples: npt.ArrayLike,
        *,
        integration_weights: npt.ArrayLike | None = None,
        component_ids: npt.ArrayLike | None = None,
    ) -> None:
        points = readonly_copy(samples, dtype=np.float64)
        if points.ndim != 2 or 0 in points.shape or not np.all(np.isfinite(points)):
            msg = "samples must be a nonempty finite array of shape (n_samples, n_features)."
            raise ValueError(msg)
        weights = None
        if integration_weights is not None:
            weights = as_weights(
                integration_weights,
                expected_length=len(points),
                normalize=False,
                name="integration_weights",
            )
            if not np.isclose(weights.sum(), 1.0, rtol=1e-7, atol=1e-8):
                msg = "integration_weights must sum to one."
                raise ValueError(msg)
        ids = None
        if component_ids is not None:
            array = np.asarray(component_ids)
            if array.ndim != 1 or len(array) != len(points):
                msg = "component_ids must have one entry per sample."
                raise ValueError(msg)
            if array.dtype.kind not in "iu":
                msg = "component_ids must contain integers."
                raise TypeError(msg)
            if np.any(array < 0) or np.any(array > np.iinfo(np.intp).max):
                msg = "component_ids must contain nonnegative component positions."
                raise IndexError(msg)
            ids = readonly_copy(array, dtype=np.intp)
        object.__setattr__(self, "samples", points)
        object.__setattr__(self, "integration_weights", weights)
        object.__setattr__(self, "component_ids", ids)


class SampleSpec(Protocol):
    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Return a batch originating from distribution."""
        ...


class BatchSampleSpec(Protocol):
    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Return one batch per distribution, preserving input order."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Draw:
    """Fresh iid samples. Seeds restart streams; generator objects advance them."""

    n_samples: int = 10_000
    rng: np.random.Generator | int | None = None

    def __post_init__(self) -> None:
        _ = validate_positive_int(self.n_samples, name="n_samples")

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Return observations (n_samples, n_features) with their integration design."""
        _validate_distribution(distribution)
        return SampleBatch(distribution.sample(self.n_samples, rng=self.rng))

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Return one immutable batch per source, consuming a shared generator."""
        _validate_distributions(distributions)
        rng = np.random.default_rng(self.rng)
        return tuple(SampleBatch(d.sample(self.n_samples, rng=rng)) for d in distributions)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stratified:
    """Fixed component allocations with integration mass preserved in the batch."""

    n_samples: int = 10_000
    rng: np.random.Generator | int | None = None

    def __post_init__(self) -> None:
        _ = validate_positive_int(self.n_samples, name="n_samples")

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Return observations (n_samples, n_features) with their integration design."""
        _validate_distribution(distribution)
        return stratified_mixture_samples(distribution, self)

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Return one immutable batch per source, consuming a shared generator."""
        _validate_distributions(distributions)
        rng = np.random.default_rng(self.rng)
        return tuple(stratified_mixture_samples(d, self, rng=rng) for d in distributions)


@dataclass(frozen=True, slots=True, init=False)
class Samples:
    """Precomputed observations or a batch retaining its integration design."""

    samples: SampleBatch

    def __init__(self, samples: SampleBatch | npt.ArrayLike) -> None:
        batch = samples if isinstance(samples, SampleBatch) else SampleBatch(samples)
        object.__setattr__(self, "samples", batch)

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Return observations (n_samples, n_features) with their integration design."""
        batch = self.samples
        return validate_sample_batch(distribution, batch)


@dataclass(frozen=True, slots=True, init=False)
class SampleBatches:
    """One precomputed batch per original candidate, before filtering or reordering."""

    samples: tuple[SampleBatch, ...]

    def __init__(self, samples: Sequence[SampleBatch] | npt.ArrayLike) -> None:
        values = samples
        if isinstance(values, Sequence) and all(isinstance(batch, SampleBatch) for batch in values):
            batches = tuple(values)
        else:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 3 or 0 in array.shape:
                msg = "samples must have shape (n_distributions, n_samples, n_features)."
                raise ValueError(msg)
            batches = tuple(SampleBatch(batch) for batch in array)
        if not batches:
            msg = "samples must contain at least one batch."
            raise ValueError(msg)
        object.__setattr__(self, "samples", batches)

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Return one immutable batch per source, consuming a shared generator."""
        _validate_distributions(distributions)
        batches = self.samples
        if len(batches) != len(distributions):
            msg = "samples must contain one batch per original candidate distribution."
            raise ValueError(msg)
        return tuple(starmap(validate_sample_batch, zip(distributions, batches, strict=True)))


def validate_sample_batch(distribution: GaussianDistribution, batch: object) -> SampleBatch:
    _validate_distribution(distribution)
    if not isinstance(batch, SampleBatch):
        msg = "sampling specifications must return SampleBatch records."
        raise TypeError(msg)
    _ = as_points(
        batch.samples, n_features=distribution.n_features, name="samples", require_nonempty=True
    )
    if batch.component_ids is not None:
        weights, _, _ = distribution.component_arrays()
        ids = batch.component_ids
        if np.any(ids >= len(weights)):
            msg = "component_ids must reference components of the source distribution."
            raise IndexError(msg)
        if batch.integration_weights is None:
            return batch
        counts = np.bincount(ids, minlength=len(weights))
        if np.any(counts[weights > 0] == 0):
            msg = "component_ids must cover every positive-weight source component."
            raise ValueError(msg)
        if not np.allclose(batch.integration_weights, weights[ids] / counts[ids]):
            msg = "integration_weights must match source component mass divided by stratum count."
            raise ValueError(msg)
    return batch


def stratified_mixture_samples(
    distribution: GaussianDistribution, spec: Stratified, *, rng: np.random.Generator | None = None
) -> SampleBatch:
    if isinstance(distribution, Gaussian):
        distribution = GaussianMixture.from_components([distribution])
    counts = stratified_component_counts(distribution.weights, spec.n_samples)
    rng = np.random.default_rng(spec.rng) if rng is None else rng
    samples: list[FloatArray] = []
    ids: list[npt.NDArray[np.intp]] = []
    for index, count in enumerate(counts):
        if count:
            samples.append(distribution.get_component(index).sample(int(count), rng=rng))
            ids.append(np.full(int(count), index, dtype=np.intp))
    component_ids = np.concatenate(ids)
    return SampleBatch(
        np.vstack(samples),
        component_ids=component_ids,
        integration_weights=distribution.weights[component_ids] / counts[component_ids],
    )


def sample_with_weights(
    distribution: GaussianDistribution, sampling: SampleSpec
) -> tuple[FloatArray, FloatArray | None]:
    if not callable(getattr(sampling, "sample", None)):
        msg = "sampling must implement sample(distribution) -> SampleBatch."
        raise TypeError(msg)
    batch = validate_sample_batch(distribution, sampling.sample(distribution))
    return batch.samples, batch.integration_weights


def sample_batches_with_weights(
    distributions: Sequence[GaussianDistribution], sampling: BatchSampleSpec
) -> tuple[FloatArray, FloatArray | None]:
    if not distributions:
        msg = "distributions must contain at least one distribution."
        raise ValueError(msg)
    if not callable(getattr(sampling, "sample_batches", None)):
        msg = "candidate_sampling must implement sample_batches(distributions)."
        raise TypeError(msg)
    batches = sampling.sample_batches(distributions)
    if len(batches) != len(distributions):
        msg = "candidate_sampling must return one batch per distribution."
        raise ValueError(msg)
    batches = tuple(starmap(validate_sample_batch, zip(distributions, batches, strict=True)))
    if len({b.samples.shape[0] for b in batches}) != 1:
        msg = "All sample batches must have the same number of samples."
        raise ValueError(msg)
    samples = np.stack([b.samples for b in batches])
    if all(b.integration_weights is None for b in batches):
        return samples, None
    weights = np.stack([
        b.integration_weights
        if b.integration_weights is not None
        else np.full(len(b.samples), 1 / len(b.samples))
        for b in batches
    ])
    return samples, weights


def stratified_component_counts(weights: npt.ArrayLike, n_samples: int) -> npt.NDArray[np.intp]:
    """Allocate exact stratified sample counts from mixture weights."""
    n_samples = validate_positive_int(n_samples, name="n_samples")
    weights_arr = as_weights(weights, name="weights", normalize=True)
    positive = weights_arr > 0.0
    n_positive = int(np.count_nonzero(positive))
    if n_samples < n_positive:
        msg = (
            "sampling.Stratified requires at least one sample per positive-weight component, "
            f"got n_samples={n_samples} for {n_positive} positive components."
        )
        raise ValueError(msg)
    normalized = weights_arr / float(np.sum(weights_arr))
    expected = normalized * n_samples
    counts = np.floor(expected).astype(np.intp)
    counts[positive & (counts == 0)] = 1
    while int(np.sum(counts)) > n_samples:
        adjustable = np.flatnonzero(counts > 1)
        excess = counts[adjustable] - expected[adjustable]
        counts[adjustable[int(np.argmax(excess))]] -= 1

    while int(np.sum(counts)) < n_samples:
        candidates = np.flatnonzero(positive)
        residual = expected[candidates] - counts[candidates]
        counts[candidates[int(np.argmax(residual))]] += 1
    return counts


def _validate_distribution(distribution: object) -> None:
    if not isinstance(distribution, (Gaussian, GaussianMixture)):
        msg = "distribution must be a Gaussian or GaussianMixture."
        raise TypeError(msg)


def _validate_distributions(distributions: Sequence[GaussianDistribution]) -> None:
    if not distributions:
        msg = "distributions must contain at least one distribution."
        raise ValueError(msg)
    for distribution in distributions:
        _validate_distribution(distribution)
