from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import override

from gmm_divergence._core._validation import (
    as_points,
    as_positive_sample_count,
    as_sample_batches,
    as_weights,
)
from gmm_divergence.distributions._gaussian import Gaussian
from gmm_divergence.distributions._mixture import GaussianMixture

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._typing import GaussianLike


class SampleSpec(Protocol):
    """Protocol for sampling specifications."""

    def sample(self, distribution: GaussianLike) -> FloatArray:
        """Return samples described by the specification."""
        ...


class BatchSampleSpec(Protocol):
    """Protocol for sample specifications that can sample distribution sequences."""

    def sample_batches(self, distributions: Sequence[GaussianLike]) -> FloatArray:
        """Return sample batches described by the specification."""
        ...


@dataclass(frozen=True, slots=True)
class Draw(SampleSpec, BatchSampleSpec):
    """Draw fresh samples from the Gaussian-family distribution being estimated.

    Use this when the estimator or fitting objective should own sampling.
    Passing a seed or generator through `rng` makes repeated calls
    reproducible.
    """

    n_samples: int = 10_000
    """Number of samples to draw."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _ = as_positive_sample_count(self.n_samples, name="n_samples")

    @override
    def sample(self, distribution: GaussianLike) -> FloatArray:
        """Return the batch of samples corresponding to the given distribution."""
        return distribution.sample(n_samples=self.n_samples, rng=self.rng)

    @override
    def sample_batches(self, distributions: Sequence[GaussianLike]) -> FloatArray:
        """Return one independently drawn sample batch per distribution."""
        rng = np.random.default_rng(self.rng)
        return _sample_each_distribution(
            distributions,
            lambda distribution: distribution.sample(n_samples=self.n_samples, rng=rng),
        )


@dataclass(frozen=True, slots=True)
class Stratified(SampleSpec, BatchSampleSpec):
    """Draw stratified samples from a Gaussian-family distribution.

    For a Gaussian mixture, component sample counts are allocated
    deterministically from the mixture weights, then samples are drawn from each
    component. Every positive-weight component receives at least one sample, so
    `n_samples` must be at least the number of positive-weight components. A
    single Gaussian is treated as a one-component mixture.
    """

    n_samples: int = 10_000
    """Total number of samples to draw."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _ = as_positive_sample_count(self.n_samples, name="n_samples")

    @override
    def sample(self, distribution: GaussianLike) -> FloatArray:
        """Return the batch of samples corresponding to the given distribution."""
        return stratified_mixture_samples(distribution, self).samples

    @override
    def sample_batches(self, distributions: Sequence[GaussianLike]) -> FloatArray:
        """Return one stratified sample batch per distribution."""
        rng = np.random.default_rng(self.rng)
        return _sample_each_distribution(
            distributions,
            lambda distribution: stratified_mixture_samples(distribution, self, rng=rng).samples,
        )


@dataclass(frozen=True, slots=True)
class Samples(SampleSpec):
    """Use precomputed samples from a single Gaussian-family reference distribution."""

    samples: npt.ArrayLike
    """Sample array with shape `(n_samples, n_features)`."""

    @override
    def sample(self, distribution: GaussianLike) -> FloatArray:
        """Return the batch of samples corresponding to the given distribution."""
        return as_points(self.samples, n_features=distribution.dim, name="samples")


@dataclass(frozen=True, slots=True)
class SampleBatches(BatchSampleSpec):
    """Use precomputed sample batches for a sequence of candidate distributions."""

    samples: npt.ArrayLike
    """Sample array with shape `(n_distributions, n_samples, n_features)`."""

    @override
    def sample_batches(self, distributions: Sequence[GaussianLike]) -> FloatArray:
        """Return precomputed sample batches for the given distributions."""
        expected_dim = distributions[0].dim if distributions else 0
        return as_sample_batches(
            self.samples,
            n_distributions=len(distributions),
            n_features=expected_dim,
            name="samples",
        )


def _sample_each_distribution(
    distributions: Sequence[GaussianLike], sampler: Callable[[GaussianLike], FloatArray]
) -> FloatArray:
    if len(distributions) == 0:
        msg = "distributions must contain at least one distribution."
        raise ValueError(msg)

    validated: list[FloatArray] = []
    n_samples: int | None = None
    for index, distribution in enumerate(distributions):
        batch = sampler(distribution)
        batch_arr = as_points(batch, n_features=distribution.dim, name=f"samples[{index}]")
        if n_samples is None:
            n_samples = batch_arr.shape[0]
        elif batch_arr.shape[0] != n_samples:
            msg = (
                "All sample batches must have the same number of samples, "
                f"got {n_samples} and {batch_arr.shape[0]}."
            )
            raise ValueError(msg)
        validated.append(batch_arr)

    return np.stack(validated).astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class StratifiedSampleResult:
    """Samples and component metadata produced by stratified mixture sampling."""

    samples: FloatArray
    """Stacked sample array with shape `(n_samples, n_features)`."""
    component_ids: npt.NDArray[np.intp]
    """Component index for each sample."""
    counts: npt.NDArray[np.intp]
    """Number of samples drawn from each component."""


def stratified_mixture_samples(
    distribution: GaussianLike, spec: Stratified, *, rng: np.random.Generator | None = None
) -> StratifiedSampleResult:
    """Draw stratified samples from a Gaussian mixture."""
    if isinstance(distribution, Gaussian):
        distribution = GaussianMixture.from_components([distribution], weights=[1.0])

    counts = stratified_component_counts(distribution.weights, spec.n_samples)
    rng = np.random.default_rng(spec.rng) if rng is None else rng
    samples: list[FloatArray] = []
    component_ids: list[npt.NDArray[np.intp]] = []

    for component_index, count in enumerate(counts):
        if count == 0:
            continue
        component = distribution.get_component(int(component_index))
        samples.append(component.sample(int(count), rng=rng))
        component_ids.append(np.full(int(count), component_index, dtype=np.intp))

    return StratifiedSampleResult(
        samples=np.vstack(samples).astype(np.float64, copy=False),
        component_ids=np.concatenate(component_ids).astype(np.intp, copy=False),
        counts=counts,
    )


def stratified_component_counts(weights: npt.ArrayLike, n_samples: int) -> npt.NDArray[np.intp]:
    """Allocate exact stratified sample counts from mixture weights."""
    n_samples = as_positive_sample_count(n_samples, name="n_samples")
    weights_arr = as_weights(weights, name="weights", normalize=True)
    expected = weights_arr * n_samples
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
