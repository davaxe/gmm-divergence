from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from gmm_divergence._core._validation import as_points, as_positive_sample_count, as_sample_batches
from gmm_divergence.distributions._mixture import GaussianMixture

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._base import Distribution


@dataclass(frozen=True, slots=True)
class DrawSamples:
    """Draw fresh samples from the distribution being estimated.

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


@dataclass(frozen=True, slots=True)
class StratifiedSamples:
    """Draw stratified samples from a Gaussian mixture.

    Component sample counts are allocated deterministically from the mixture
    weights, then samples are drawn from each component. Every positive-weight
    component receives at least one sample, so `n_samples` must be at least the
    number of positive-weight components. This is only valid for
    `GaussianMixture` distributions.
    """

    n_samples: int = 10_000
    """Total number of samples to draw."""
    rng: np.random.Generator | int | None = None
    """Random generator or seed used when drawing samples."""

    def __post_init__(self) -> None:
        _ = as_positive_sample_count(self.n_samples, name="n_samples")


@dataclass(frozen=True, slots=True)
class UseSamples:
    """Use precomputed samples from a single reference distribution."""

    samples: npt.ArrayLike
    """Sample array with shape `(n_samples, n_features)`."""


@dataclass(frozen=True, slots=True)
class UseSampleBatches:
    """Use precomputed sample batches for a sequence of candidate distributions."""

    samples: npt.ArrayLike
    """Sample array with shape `(n_distributions, n_samples, n_features)`."""


SampleSpec: TypeAlias = DrawSamples | StratifiedSamples | UseSamples
SampleBatchSpec: TypeAlias = DrawSamples | StratifiedSamples | UseSampleBatches


def resolve_samples(distribution: Distribution, spec: SampleSpec) -> FloatArray:
    """Return samples described by a single-distribution sample specification."""
    match spec:
        case DrawSamples(n_samples=n_samples, rng=rng):
            return distribution.sample(n_samples=n_samples, rng=rng)
        case StratifiedSamples():
            return stratified_mixture_samples(distribution, spec).samples
        case UseSamples(samples=samples):
            return as_points(samples, n_features=distribution.dim, name="samples")


def resolve_sample_batches(
    distributions: Sequence[Distribution], spec: SampleBatchSpec
) -> FloatArray:
    """Return sample batches described by a sample-batch specification."""
    match spec:
        case DrawSamples(n_samples=n_samples, rng=rng):
            rng = np.random.default_rng(rng)
            return np.asarray(
                [distribution.sample(n_samples, rng=rng) for distribution in distributions],
                dtype=np.float64,
            )
        case StratifiedSamples():
            return np.asarray(
                [
                    stratified_mixture_samples(distribution, spec).samples
                    for distribution in distributions
                ],
                dtype=np.float64,
            )
        case UseSampleBatches(samples=samples):
            expected_dim = distributions[0].dim if distributions else 0
            return as_sample_batches(
                samples, n_distributions=len(distributions), n_features=expected_dim, name="samples"
            )


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
    distribution: Distribution, spec: StratifiedSamples
) -> StratifiedSampleResult:
    """Draw stratified samples from a Gaussian mixture."""
    if not isinstance(distribution, GaussianMixture):
        msg = (
            "StratifiedSamples requires a GaussianMixture distribution, "
            f"got {type(distribution).__name__}."
        )
        raise TypeError(msg)

    counts = stratified_component_counts(distribution.weights, spec.n_samples)
    rng = np.random.default_rng(spec.rng)
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
    """Allocate exact stratified sample counts from mixture weights.

    Every component with positive weight receives at least one sample. This
    avoids silently biasing stratified estimates by assigning zero samples to a
    component that still contributes mass to the mixture.
    """
    n_samples = as_positive_sample_count(n_samples, name="n_samples")
    weights_arr = np.asarray(weights, dtype=np.float64)

    if weights_arr.ndim != 1 or weights_arr.shape[0] == 0:
        msg = "weights must be a non-empty 1D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0.0):
        msg = "weights must contain finite nonnegative values."
        raise ValueError(msg)

    positive = weights_arr > 0.0
    n_positive = int(np.count_nonzero(positive))
    if n_positive == 0:
        msg = "weights must contain at least one positive value."
        raise ValueError(msg)
    if n_samples < n_positive:
        msg = (
            "StratifiedSamples requires at least one sample per positive-weight component, "
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
