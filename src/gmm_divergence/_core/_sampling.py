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
    """Immutable observations with optional integration weights and strata.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_features)
        Nonempty, finite observations, copied into a read-only float64 array.
    integration_weights : array-like, shape (n_samples,), or None, default=None
        Finite, nonnegative weights summing to one within rtol=1e-7, atol=1e-8.
        Values are copied without normalization. None requests an ordinary mean.
    component_ids : array-like, shape (n_samples,), or None, default=None
        Nonnegative integer component positions, copied into a read-only intp
        array. Source bounds are checked when the batch is used.

    Notes
    -----
    When both weights and component IDs are provided, reuse validates that each
    positive-weight source component is represented and each observation's weight
    equals its component mass divided by the number of observations in its stratum.
    """

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
    """Protocol for a sample source returning observations and integration metadata."""

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Return a batch originating from the supplied distribution.

        Parameters
        ----------
        distribution : Gaussian or GaussianMixture
            Source distribution; any precomputed observations must come from it.

        Returns
        -------
        SampleBatch
            Finite observations with matching feature dimensions and optional
            integration weights and source component IDs.
        """
        ...


class BatchSampleSpec(Protocol):
    """Protocol for one sample batch per source distribution."""

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Return one batch per distribution, preserving input order.

        Parameters
        ----------
        distributions : sequence of Gaussian or GaussianMixture
            Nonempty sequence of source distributions.

        Returns
        -------
        tuple of SampleBatch
            One batch per source. Fitting requires equal sample counts and
            feature dimensions across batches.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Draw:
    """Draw fresh, independent observations from each source distribution.

    Parameters
    ----------
    n_samples : int, default=10000
        Positive number of observations per distribution.
    rng : numpy.random.Generator, int, or None, default=None
        Random source. Integer seeds restart the stream on each method call;
        generators advance their state. None uses fresh entropy. Batch sampling
        uses one shared stream across sources within each call.
    """

    n_samples: int = 10_000
    rng: np.random.Generator | int | None = None

    def __post_init__(self) -> None:
        _ = validate_positive_int(self.n_samples, name="n_samples")

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Draw an independent, equally weighted batch from the source.

        Parameters
        ----------
        distribution : Gaussian or GaussianMixture
            Source distribution with the required feature dimensions.

        Returns
        -------
        SampleBatch
            New float64 observations with no integration weights or component IDs.
        """
        _validate_distribution(distribution)
        return SampleBatch(distribution.sample(self.n_samples, rng=self.rng))

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Draw independent batches using one shared random stream.

        Parameters
        ----------
        distributions : sequence of Gaussian or GaussianMixture
            Nonempty sequence of source distributions in the desired batch order.

        Returns
        -------
        tuple of SampleBatch
            One batch per source in input order, each containing n_samples observations.
            A shared generator advances across sources within this call.
        """
        _validate_distributions(distributions)
        rng = np.random.default_rng(self.rng)
        return tuple(SampleBatch(d.sample(self.n_samples, rng=rng)) for d in distributions)


@dataclass(frozen=True, slots=True, kw_only=True)
class Stratified:
    """Draw fixed component allocations while preserving mixture integration mass.

    Parameters
    ----------
    n_samples : int, default=10000
        Positive number of observations per distribution. Must allow at least
        one draw for every positive-weight component of each source.
    rng : numpy.random.Generator, int, or None, default=None
        Random source. Integer seeds restart the stream on each method call;
        generators advance their state. Batch sampling shares one stream.

    Notes
    -----
    Each observation from component k has integration weight `weights[k] / n_k`,
    where n_k is its allocated count. Batches also carry component IDs. Counts
    sum to n_samples; zero-weight components receive no draws. A Gaussian is
    treated as a one-component mixture. Observations are grouped by component.
    """

    n_samples: int = 10_000
    rng: np.random.Generator | int | None = None

    def __post_init__(self) -> None:
        _ = validate_positive_int(self.n_samples, name="n_samples")

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Draw a component-stratified batch from the source.

        Parameters
        ----------
        distribution : Gaussian or GaussianMixture
            Source distribution with the required feature dimensions.

        Returns
        -------
        SampleBatch
            New observations with normalized integration weights and component IDs.
        """
        _validate_distribution(distribution)
        return stratified_mixture_samples(distribution, self)

    def sample_batches(
        self, distributions: Sequence[GaussianDistribution]
    ) -> tuple[SampleBatch, ...]:
        """Draw stratified batches using one shared random stream.

        Parameters
        ----------
        distributions : sequence of Gaussian or GaussianMixture
            Nonempty sequence of source distributions in the desired batch order.

        Returns
        -------
        tuple of SampleBatch
            One batch per source in input order, each containing n_samples observations.
            A shared generator advances across sources within this call.
        """
        _validate_distributions(distributions)
        rng = np.random.default_rng(self.rng)
        return tuple(stratified_mixture_samples(d, self, rng=rng) for d in distributions)


@dataclass(frozen=True, slots=True, init=False)
class Samples:
    """Reuse observations from a single source distribution without drawing again.

    Parameters
    ----------
    samples : SampleBatch or array-like, shape (n_samples, n_features)
        Nonempty, finite source observations. Arrays are copied into an immutable
        batch and treated as equally weighted observations. A SampleBatch is
        reused with its integration weights and component IDs intact.

    Notes
    -----
    Feature dimensions and component metadata are checked against the source
    when sampling. The caller must ensure observations originate from that
    source; this cannot be inferred from their values.
    """

    samples: SampleBatch

    def __init__(self, samples: SampleBatch | npt.ArrayLike) -> None:
        batch = samples if isinstance(samples, SampleBatch) else SampleBatch(samples)
        object.__setattr__(self, "samples", batch)

    def sample(self, distribution: GaussianDistribution) -> SampleBatch:
        """Validate and return the stored batch without drawing samples.

        Parameters
        ----------
        distribution : Gaussian or GaussianMixture
            Source distribution with the required feature dimensions.

        Returns
        -------
        SampleBatch
            Stored batch with its integration weights and component IDs unchanged.
        """
        batch = self.samples
        return validate_sample_batch(distribution, batch)


@dataclass(frozen=True, slots=True, init=False)
class SampleBatches:
    """Reuse one precomputed batch per source distribution.

    Parameters
    ----------
    samples : sequence of SampleBatch or array-like
        Nonempty sequence of immutable batches, or finite observations with
        shape `(n_distributions, n_samples, n_features)`. Arrays are copied;
        records preserve their integration weights and component IDs.

    Notes
    -----
    For fitting, supply one batch per original candidate before filtering or
    reordering. Batches must have equal sample counts and matching feature
    dimensions. The caller is responsible for sampling from the correct sources.
    """

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
        """Validate and return precomputed batches without drawing new samples.

        Parameters
        ----------
        distributions : sequence of Gaussian or GaussianMixture
            Original sources in the same order as the stored batches.

        Returns
        -------
        tuple of SampleBatch
            Stored batches with source dimensions and component metadata checked.
        """
        _validate_distributions(distributions)
        batches = self.samples
        if len(batches) != len(distributions):
            msg = "samples must contain one batch per original candidate distribution."
            raise ValueError(msg)
        return tuple(starmap(validate_sample_batch, zip(distributions, batches, strict=True)))


def validate_sample_batch(distribution: GaussianDistribution, batch: object) -> SampleBatch:
    """Check a SampleBatch against its source and return the same record.

    distribution determines feature count, component bounds, and stratified
    integration mass. An unsupported batch raises TypeError; invalid shapes or
    integration mass raise ValueError, and invalid component IDs raise IndexError.
    """
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
    """Draw spec.n_samples observations with fixed source component allocations.

    rng overrides spec.rng when provided, allowing a shared stream across
    sources. Returns observations grouped by component with IDs and weights
    equal to component mass divided by allocated count.
    """
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
    """Resolve and validate sampling for one distribution.

    Returns samples with shape (n_samples, n_features) and optional normalized
    integration weights with shape (n_samples,). Both are read-only batch arrays.
    """
    if not callable(getattr(sampling, "sample", None)):
        msg = "sampling must implement sample(distribution) -> SampleBatch."
        raise TypeError(msg)
    batch = validate_sample_batch(distribution, sampling.sample(distribution))
    return batch.samples, batch.integration_weights


def sample_batches_with_weights(
    distributions: Sequence[GaussianDistribution], sampling: BatchSampleSpec
) -> tuple[FloatArray, FloatArray | None]:
    """Resolve sampling and stack equal-length batches in source order.

    distributions must be nonempty; sampling must supply one valid batch per
    source. Returns observations of shape (n_distributions, n_samples, n_features)
    and optional weights of shape (n_distributions, n_samples). If some batches
    have weights and others do not, unweighted batches receive uniform weights.
    """
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
    """Allocate a total of n_samples draws among nonnegative mixture weights.

    weights are normalized. Each positive-weight component receives at least
    one draw, requiring n_samples to be at least their count. Remaining draws
    favor the largest allocation deficits, with earlier positions breaking ties.
    Returns integer counts in component order; zero-weight components get zero.
    """
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
