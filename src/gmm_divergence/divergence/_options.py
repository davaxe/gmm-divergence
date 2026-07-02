from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from gmm_divergence._core._sampling import Draw, SampleSpec
from gmm_divergence._core._validation import validate_positive_finite as _validate_positive_float
from gmm_divergence._core._validation import validate_positive_int as _validate_positive_int

Approximation: TypeAlias = Literal["nearest", "moment_matching"]


@dataclass(frozen=True, slots=True, init=False)
class MonteCarlo:
    r"""Monte Carlo KL estimator configuration.

    Estimates the KL divergence with samples from the reference distribution
    $p$:

    $$
    D_{\mathrm{KL}}(p \| q)
    =
    \mathbb{E}_{X \sim p}
    \left[
        \log p(X) - \log q(X)
    \right].
    $$

    This is the most general estimator: it can be used whenever `p` can be
    sampled and both `p` and `q` can evaluate log densities.

    Sampling is configured explicitly with `sampling.Draw(...)`,
    `sampling.Samples(...)`, or `sampling.Stratified(...)`. Adaptive standard-error
    control requires `sampling.Draw(...)` because it must draw additional
    batches.
    """

    sampling: SampleSpec = field(default_factory=Draw)
    """Sampling specification used to estimate the expectation under p."""
    target_standard_error: float | None = None
    """Optional standard-error target for adaptive sampling."""
    max_samples: int | None = None
    """Maximum sample count when adaptive sampling is enabled.

    If omitted, adaptive Monte Carlo uses ten times the initial draw count.
    """
    batch_size: int | None = None
    """Batch size for additional adaptive samples.

    If omitted, adaptive Monte Carlo uses the initial draw count as the batch size.
    """

    def __init__(
        self,
        sampling: int | SampleSpec | None = None,
        target_standard_error: float | None = None,
        max_samples: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if isinstance(sampling, int):
            sampling = Draw(n_samples=sampling)
        object.__setattr__(self, "sampling", sampling or Draw())
        object.__setattr__(self, "target_standard_error", target_standard_error)
        object.__setattr__(self, "max_samples", max_samples)
        object.__setattr__(self, "batch_size", batch_size)
        if not isinstance(self.sampling, (Draw,)):
            if self.target_standard_error is None:
                return
            msg = "target_standard_error requires sampling=sampling.Draw(...)."
            raise ValueError(msg)

        sampling_count = self.sampling.n_samples
        if self.target_standard_error is not None:
            _validate_positive_float(self.target_standard_error, name="target_standard_error")
        if self.max_samples is not None:
            _validate_positive_int(self.max_samples, name="max_samples")
            if self.max_samples < sampling_count:
                msg = (
                    "max_samples must be greater than or equal to the initial sampling count, "
                    f"got max_samples={self.max_samples} and sampling={sampling_count}."
                )
                raise ValueError(msg)
        if self.batch_size is not None:
            _validate_positive_int(self.batch_size, name="batch_size")


@dataclass(frozen=True, slots=True)
class Unscented:
    r"""Unscented-transform KL estimator configuration.

    Estimates the KL divergence by replacing random samples from $p$ with
    deterministic sigma points generated from the Gaussian components of $p$.
    For each component, the sigma points are chosen to capture its mean and
    covariance structure, then the estimator averages

    $$
    \log p(x) - \log q(x)
    $$

    over those points. This can be useful when deterministic, low-sample-count
    estimates are preferred over Monte Carlo sampling.
    """


@dataclass(frozen=True, slots=True)
class MomentMatchedGaussian:
    r"""Moment-matched Gaussian KL estimator configuration.

    Approximates one or both inputs with Gaussian summaries and then computes a
    Gaussian KL surrogate for

    $$
    D_{\mathrm{KL}}(p \| q).
    $$

    This is a fast heuristic rather than an exact Gaussian-mixture KL
    computation. It is mainly useful as a rough baseline or when a cheap
    approximation is preferable to a sampled estimate.

    The `"moment_matching"` strategy replaces each Gaussian mixture by a single
    Gaussian with the same mean and covariance, then computes the closed-form
    Gaussian KL divergence between those summaries. This preserves global first
    and second moments but discards multimodality.

    The `"nearest"` strategy computes pairwise closed-form KL divergences
    between Gaussian components and returns the smallest component-level value.
    This captures the closest local match between the mixtures but ignores
    mixture weights and the full mixture shape.
    """

    approximation: Approximation = "moment_matching"
    """Gaussian approximation strategy to use."""

    def __post_init__(self) -> None:
        if self.approximation not in {"nearest", "moment_matching"}:
            msg = "approximation must be 'nearest' or 'moment_matching'."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Variational:
    r"""Variational KL estimator configuration."""


@dataclass(frozen=True, slots=True)
class ClosedForm:
    r"""Closed-form Gaussian KL configuration.

    Computes the exact KL divergence between two Gaussian distributions:

    $$
    D_{\mathrm{KL}}(p \| q) =
    \frac{1}{2}
    \left[
        \mathrm{tr}(\Sigma_q^{-1}\Sigma_p)
        + (\mu_q-\mu_p)^\top\Sigma_q^{-1}(\mu_q-\mu_p)
        - d
        + \log\frac{\det\Sigma_q}{\det\Sigma_p}
    \right].
    $$

    This method is only valid when both inputs are single Gaussian
    distributions.
    """


KLMethod: TypeAlias = (
    Literal["monte_carlo", "unscented", "gaussian_approximation", "closed_form", "variational"]
    | MonteCarlo
    | Unscented
    | MomentMatchedGaussian
    | ClosedForm
    | Variational
)


@dataclass(frozen=True, slots=True)
class KLDivergence:
    """Configuration for the directed Kullback-Leibler divergence."""

    method: KLMethod = "monte_carlo"
    """Method used to estimate ``KL(p || q)``."""
    prefer_closed_form: bool = True
    """Use closed-form Gaussian KL when both inputs are single Gaussian."""


@dataclass(frozen=True, slots=True)
class SymmetricKLDivergence:
    """Configuration for the symmetric Kullback-Leibler divergence."""

    method: KLMethod = "monte_carlo"
    """Method used for each directed KL estimate."""
    prefer_closed_form: bool = True
    """Use closed-form Gaussian KL in each direction when possible."""


@dataclass(frozen=True, slots=True)
class JensenShannonDivergence:
    """Configuration for the Jensen-Shannon divergence."""

    method: KLMethod = "monte_carlo"
    """Method used for the KL estimates against the midpoint mixture."""
    prefer_closed_form: bool = True
    """Pass through closed-form preference to the underlying KL estimates."""


DivergenceName: TypeAlias = Literal["kl", "symmetric_kl", "jensen_shannon"]
DivergenceSpec: TypeAlias = (
    DivergenceName | KLDivergence | SymmetricKLDivergence | JensenShannonDivergence
)
