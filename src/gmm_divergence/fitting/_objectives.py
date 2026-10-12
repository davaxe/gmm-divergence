"""Objective functions for fitting Gaussian mixtures via divergence minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import logsumexp
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmm_divergence._core._types import FloatArray
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture
    from gmm_divergence.distributions._typing import GaussianDistribution
    from gmm_divergence.fitting._simplex import ObjectiveFn


def _sample_mean(
    values: FloatArray, integration_weights: FloatArray | None, *, axis: int
) -> FloatArray:
    """Average samples using their integration weights when supplied."""
    if integration_weights is None:
        return np.mean(values, axis=axis)
    # Weights describe the leading sample dimensions; trailing axes hold densities.
    expanded_weights = integration_weights.reshape((
        *integration_weights.shape,
        *((1,) * (values.ndim - integration_weights.ndim)),
    ))
    return np.sum(values * expanded_weights, axis=axis)


def logpdf_matrix(components: Sequence[GaussianDistribution], samples: FloatArray) -> FloatArray:
    """Evaluate candidate log densities at fixed samples.

    components is the ordered candidate sequence; samples has shape
    (n_samples, n_features). Returns float64 log densities with shape
    (n_samples, n_candidates). Candidates
    may themselves be mixtures, rather than individual Gaussian components.
    """
    samples = np.asarray(samples, dtype=np.float64)
    log_q = np.empty((samples.shape[0], len(components)), dtype=np.float64)
    for k, qk in enumerate(components):
        log_q[:, k] = qk.logpdf(samples)
    return log_q


def mixture_stats(
    log_q: FloatArray, weights: FloatArray, *, eps: float = 1e-300
) -> tuple[FloatArray, FloatArray]:
    """Compute log mixture densities and candidate responsibilities.

    log_q has shape (n_samples, n_candidates), and weights has shape
    (n_candidates,). eps floors weights only when taking their logarithms.
    Returns log densities (n_samples,) and candidate responsibilities
    (n_samples, n_candidates); caller-supplied weights are not normalized here.
    """
    log_w = np.log(np.maximum(np.asarray(weights, dtype=np.float64), eps))
    log_terms = np.asarray(log_q, dtype=np.float64) + log_w[None, :]
    log_qw = logsumexp(log_terms, axis=1)
    responsibilities = np.exp(log_terms - log_qw[:, None])
    return log_qw.astype(np.float64), responsibilities.astype(np.float64)


@dataclass(frozen=True, slots=True)
class _ForwardKL:
    """Monte Carlo forward KL objective over simplex weights."""

    log_q_on_p_samples: FloatArray
    log_p_on_p_samples: FloatArray | None = None
    p_integration_weights: FloatArray | None = None
    eps: float = 1e-300
    include_constant: bool = False

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        weights = np.asarray(weights, dtype=np.float64)
        log_qw, responsibilities = mixture_stats(self.log_q_on_p_samples, weights, eps=self.eps)

        value = -float(_sample_mean(log_qw, self.p_integration_weights, axis=0))
        if self.include_constant and self.log_p_on_p_samples is not None:
            value += float(
                _sample_mean(self.log_p_on_p_samples, self.p_integration_weights, axis=0)
            )

        weights_safe = np.maximum(weights, self.eps)
        grad = -_sample_mean(
            responsibilities / weights_safe[None, :], self.p_integration_weights, axis=0
        )
        return float(value), grad.astype(np.float64)


def forward_kl(
    target: GaussianDistribution | None,
    q_components: Sequence[GaussianDistribution],
    p_samples: FloatArray,
    *,
    p_integration_weights: FloatArray | None = None,
    include_constant: bool = False,
    eps: float = 1e-300,
) -> ObjectiveFn:
    r"""Build a simplex objective for forward KL, $D_{\mathrm{KL}}(p \| q_{\mathbf{w}})$.

    Parameters
    ----------
    target : Gaussian or GaussianMixture or None
        Reference distribution, required when include_constant is True.
    q_components : sequence of Gaussian or GaussianMixture
        Fixed candidates in weight-vector order.
    p_samples : ndarray, shape (n_samples, n_features)
        Fixed reference observations used to cache candidate densities.
    p_integration_weights : ndarray or None, default=None
        Normalized reference integration weights of shape (n_samples,).
        None uses an ordinary sample mean.
    include_constant : bool, default=False
        Include mean log target density in the value. False reports
        cross-entropy; public fitting sets True to report mathematical KL.
    eps : float, default=1e-300
        Numerical floor for weights when computing logarithms and gradients.

    Returns
    -------
    callable
        Cached objective returning a scalar value and weight gradient.
    """
    p_samples = np.asarray(p_samples, dtype=np.float64)
    log_q_on_p_samples = logpdf_matrix(q_components, p_samples)

    log_p_on_p_samples = None
    if include_constant:
        if target is None:
            msg = "p is required when include_constant=True."
            raise ValueError(msg)
        log_p_on_p_samples = np.asarray(target.logpdf(p_samples), dtype=np.float64)

    return _ForwardKL(
        log_q_on_p_samples=log_q_on_p_samples,
        log_p_on_p_samples=log_p_on_p_samples,
        p_integration_weights=p_integration_weights,
        eps=eps,
        include_constant=include_constant,
    )


@dataclass(frozen=True, slots=True)
class _ReverseKL:
    """Fixed-sample reverse KL objective over simplex weights."""

    log_q_on_q_samples: FloatArray
    log_p_on_q_samples: FloatArray
    q_integration_weights: FloatArray | None = None
    eps: float = 1e-300

    @property
    def n_components(self) -> int:
        """Number of candidate batches represented by this objective."""
        return int(self.log_q_on_q_samples.shape[0])

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        weights = np.asarray(weights, dtype=np.float64)
        log_w = np.log(np.maximum(weights, self.eps))

        log_qw = logsumexp(self.log_q_on_q_samples + log_w[None, None, :], axis=2)
        component_terms = _sample_mean(
            log_qw - self.log_p_on_q_samples, self.q_integration_weights, axis=1
        )
        value = float(np.dot(weights, component_terms))

        correction = np.zeros(self.n_components, dtype=np.float64)
        for i in range(self.n_components):
            ratio_i = np.exp(self.log_q_on_q_samples[i] - log_qw[i, :, None])
            correction += weights[i] * _sample_mean(
                ratio_i,
                None if self.q_integration_weights is None else self.q_integration_weights[i],
                axis=0,
            )

        grad = component_terms + correction
        return float(value), grad.astype(np.float64)


def reverse_kl(
    target: GaussianDistribution,
    q_components: Sequence[GaussianDistribution],
    q_samples: FloatArray,
    *,
    q_integration_weights: FloatArray | None = None,
    eps: float = 1e-300,
) -> ObjectiveFn:
    r"""Build a simplex objective for reverse KL, $D_{\mathrm{KL}}(q_{\mathbf{w}} \| p)$.

    Uses fixed samples from each candidate distribution. q_components lists the
    fixed candidates in weight order; q_samples has shape
    (n_candidates, n_samples, n_features), with batch i drawn from candidate i.
    target is evaluated on those same observations. q_integration_weights
    optionally has shape (n_candidates, n_samples) with normalized rows;
    None uses ordinary within-batch means. eps floors weights for logarithms.
    Returns a cached callable producing the scalar KL estimate and gradient.
    """
    log_p_on_q_samples, log_q_on_q_samples = _candidate_sample_logpdfs(
        target, q_components, q_samples
    )

    return _ReverseKL(
        log_q_on_q_samples=log_q_on_q_samples,
        log_p_on_q_samples=log_p_on_q_samples,
        q_integration_weights=q_integration_weights,
        eps=eps,
    )


@dataclass(frozen=True, slots=True)
class _JensenShannon:
    """Fixed-sample Jensen-Shannon objective over simplex weights."""

    log_q_on_p_samples: FloatArray
    log_p_on_p_samples: FloatArray
    log_q_on_q_samples: FloatArray
    log_p_on_q_samples: FloatArray
    p_integration_weights: FloatArray | None = None
    q_integration_weights: FloatArray | None = None
    eps: float = 1e-300

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        w = np.asarray(weights, dtype=np.float64)
        log_2: float = np.log(2.0)
        log_qw_p, _ = mixture_stats(self.log_q_on_p_samples, w, eps=self.eps)
        log_m_p = np.logaddexp(self.log_p_on_p_samples, log_qw_p) - log_2
        log_w = np.log(np.maximum(w, self.eps))
        log_qw_q = logsumexp(self.log_q_on_q_samples + log_w, axis=-1)
        log_m_q = np.logaddexp(self.log_p_on_q_samples, log_qw_q) - log_2
        forward_kl = _sample_mean(
            self.log_p_on_p_samples - log_m_p, self.p_integration_weights, axis=0
        )
        reverse_kl_per_component = _sample_mean(
            log_qw_q - log_m_q, self.q_integration_weights, axis=1
        )
        value = 0.5 * (forward_kl + np.dot(w, reverse_kl_per_component))
        forward_grad = -0.25 * _sample_mean(
            np.exp(self.log_q_on_p_samples - log_m_p[:, None]), self.p_integration_weights, axis=0
        )
        mean_q_over_qw = _sample_mean(
            np.exp(self.log_q_on_q_samples - log_qw_q[..., None]),
            self.q_integration_weights,
            axis=1,
        )
        mean_q_over_m = _sample_mean(
            np.exp(self.log_q_on_q_samples - log_m_q[..., None]), self.q_integration_weights, axis=1
        )
        reverse_grad = (
            0.5 * reverse_kl_per_component + 0.5 * (w @ mean_q_over_qw) - 0.25 * (w @ mean_q_over_m)
        )
        return float(value), forward_grad + reverse_grad


def jensen_shannon(
    target: GaussianDistribution,
    q_components: Sequence[GaussianDistribution],
    p_samples: FloatArray,
    q_samples: FloatArray,
    *,
    p_integration_weights: FloatArray | None = None,
    q_integration_weights: FloatArray | None = None,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a cached simplex objective for Jensen-Shannon divergence in nats.

    target and q_components are fixed reference and candidate distributions.
    p_samples has shape (n_target_samples, n_features) and originates from
    target; q_samples has shape (n_candidates, n_candidate_samples, n_features),
    with one batch from each candidate. p_integration_weights and
    q_integration_weights optionally weight those observations with shapes
    (n_target_samples,) and (n_candidates, n_candidate_samples), normalized
    separately for the target and each candidate. eps floors weights in
    logarithms. Returns a callable producing
    the scalar divergence estimate and weight gradient, without a square root.
    """
    p_samples = np.asarray(p_samples, dtype=np.float64)
    log_p_on_q_samples, log_q_on_q_samples = _candidate_sample_logpdfs(
        target, q_components, q_samples
    )

    log_q_on_p_samples = logpdf_matrix(q_components, p_samples)
    log_p_on_p_samples = np.asarray(target.logpdf(p_samples), dtype=np.float64)

    return _JensenShannon(
        log_q_on_p_samples=log_q_on_p_samples,
        log_p_on_p_samples=log_p_on_p_samples,
        p_integration_weights=p_integration_weights,
        log_q_on_q_samples=log_q_on_q_samples,
        log_p_on_q_samples=log_p_on_q_samples,
        q_integration_weights=q_integration_weights,
        eps=eps,
    )


def _candidate_sample_logpdfs(
    target: GaussianDistribution,
    q_components: Sequence[GaussianDistribution],
    q_samples: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    q_samples = np.asarray(q_samples, dtype=np.float64)
    n_components, n_samples, n_features = q_samples.shape
    q_samples_flat = q_samples.reshape(-1, n_features)

    log_p_on_q_samples = np.asarray(target.logpdf(q_samples_flat), dtype=np.float64).reshape(
        n_components, n_samples
    )
    log_q_flat = logpdf_matrix(q_components, q_samples_flat)
    log_q_on_q_samples = log_q_flat.reshape(n_components, n_samples, n_components)
    return log_p_on_q_samples, log_q_on_q_samples


@dataclass(frozen=True, slots=True)
class _WeightedSum:
    """Weighted sum of objective functions."""

    objectives: tuple[ObjectiveFn, ...]
    weights: FloatArray

    def __call__(self, w: FloatArray) -> tuple[float, FloatArray]:
        total_value = 0.0
        total_grad = np.zeros_like(w, dtype=np.float64)

        for objective, forward_weight in zip(self.objectives, self.weights, strict=True):
            value, grad = objective(w)
            total_value += float(forward_weight) * float(value)
            total_grad += float(forward_weight) * np.asarray(grad, dtype=np.float64)

        return float(total_value), total_grad.astype(np.float64)


def bidirectional_kl(
    target: GaussianDistribution,
    q_components: Sequence[GaussianDistribution],
    p_samples: FloatArray | None = None,
    q_samples: FloatArray | None = None,
    *,
    p_integration_weights: FloatArray | None = None,
    q_integration_weights: FloatArray | None = None,
    forward_weight: float = 0.5,
    include_forward_constant: bool = True,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a weighted forward/reverse KL objective over simplex weights.

    target and q_components are the fixed reference and candidate distributions.
    forward_weight is in [0, 1]; the reverse term receives 1 - forward_weight.
    p_samples has shape (n_target_samples, n_features) and is required for a
    nonzero forward term. q_samples has shape
    (n_candidates, n_candidate_samples, n_features) and is required for a
    nonzero reverse term. Optional p_integration_weights and
    q_integration_weights have shapes (n_target_samples,) and
    (n_candidates, n_candidate_samples), normalized separately for the target
    and each candidate. include_forward_constant controls whether the target-density
    constant is included; eps floors candidate weights for numerical stability.
    Returns a callable producing the combined value and weight gradient.
    """
    objectives: list[ObjectiveFn] = []
    objective_weights: list[float] = []
    reverse_weight = 1.0 - forward_weight
    if forward_weight:
        if p_samples is None:
            msg = "p_samples is required when forward_weight is nonzero."
            raise ValueError(msg)
        objectives.append(
            forward_kl(
                target=target,
                q_components=q_components,
                p_samples=p_samples,
                p_integration_weights=p_integration_weights,
                include_constant=include_forward_constant,
                eps=eps,
            )
        )
        objective_weights.append(float(forward_weight))

    if reverse_weight:
        if q_samples is None:
            msg = "q_samples is required when reverse_weight is nonzero."
            raise ValueError(msg)
        objectives.append(
            reverse_kl(
                target=target,
                q_components=q_components,
                q_samples=q_samples,
                q_integration_weights=q_integration_weights,
                eps=eps,
            )
        )
        objective_weights.append(float(reverse_weight))

    return _WeightedSum(
        objectives=tuple(objectives), weights=np.asarray(objective_weights, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class _MomentMatching:
    """Moment-matching objective over simplex weights."""

    p_moments: FloatArray
    q_moments: FloatArray

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        weights = np.asarray(weights, dtype=np.float64)
        mixture_moments = weights @ self.q_moments
        residual = mixture_moments - self.p_moments
        value = float(residual @ residual)
        grad = 2.0 * (self.q_moments @ residual)
        return value, grad.astype(np.float64)


def moment_matching(
    target: GaussianDistribution,
    q_components: Sequence[GaussianDistribution],
    *,
    second_moments: bool = False,
) -> ObjectiveFn:
    """Build an exact squared raw-moment residual objective.

    target and q_components supply fixed reference and candidate moments.
    second_moments adds flattened covariance + outer(mean, mean) to each mean
    vector. Returns a cached value-and-gradient callable; values are not in nats.
    """
    p_moments = gaussian_family_raw_moment_vector(target, second_moments=second_moments)
    q_moments = np.asarray(
        [gaussian_family_raw_moment_vector(q, second_moments=second_moments) for q in q_components],
        dtype=np.float64,
    )

    if q_moments.ndim != 2:
        msg = f"Expected q_moments to be two-dimensional, got shape {q_moments.shape}."
        raise ValueError(msg)

    if q_moments.shape[1] != p_moments.shape[0]:
        msg = (
            "Moment dimensionality mismatch: "
            f"p has {p_moments.shape[0]} moments, "
            f"q_i has {q_moments.shape[1]}."
        )
        raise ValueError(msg)

    return _MomentMatching(p_moments=p_moments, q_moments=q_moments)


def build_gaussian_mixture_objective(
    *,
    objective: ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching,
    target: Gaussian | GaussianMixture,
    candidates: Sequence[Gaussian | GaussianMixture],
    p_samples: FloatArray | None,
    q_samples: FloatArray | None,
    p_integration_weights: FloatArray | None = None,
    q_integration_weights: FloatArray | None = None,
) -> ObjectiveFn:
    """Dispatch an objective configuration to its cached implementation.

    target and candidates are fixed distributions. p_samples and q_samples are
    optional validated reference and candidate batches, shaped
    (n_target_samples, n_features) and
    (n_candidates, n_candidate_samples, n_features); required sources depend on
    objective. Corresponding integration weights have shapes (n_target_samples,)
    and (n_candidates, n_candidate_samples), or None for ordinary means. Public KL
    objectives include their target-density constants in reported values.
    """
    match objective:
        case ForwardKL():
            if p_samples is None:
                msg = "p_samples is required for forward KL."
                raise ValueError(msg)
            return forward_kl(
                target,
                candidates,
                p_samples,
                p_integration_weights=p_integration_weights,
                include_constant=True,
            )
        case ReverseKL():
            if q_samples is None:
                msg = "q_samples is required for reverse KL."
                raise ValueError(msg)
            return reverse_kl(
                target, candidates, q_samples, q_integration_weights=q_integration_weights
            )
        case BidirectionalKL(forward_weight=forward_weight):
            return bidirectional_kl(
                target,
                candidates,
                p_samples=p_samples,
                q_samples=q_samples,
                forward_weight=forward_weight,
                p_integration_weights=p_integration_weights,
                q_integration_weights=q_integration_weights,
            )
        case JensenShannon():
            if q_samples is None or p_samples is None:
                msg = "Both p_samples and q_samples are required for Jensen-Shannon."
                raise ValueError(msg)
            return jensen_shannon(
                target,
                candidates,
                p_samples=p_samples,
                q_samples=q_samples,
                p_integration_weights=p_integration_weights,
                q_integration_weights=q_integration_weights,
            )
        case MomentMatching(fit_second_moments=fit_second_moments):
            return moment_matching(target, candidates, second_moments=fit_second_moments)


def gaussian_family_raw_moment_vector(
    distribution: GaussianDistribution, /, *, second_moments: bool = False
) -> FloatArray:
    """Return the raw moment vector used by moment-matching objectives.

    distribution supplies aggregate mean and covariance. With second_moments
    False, returns the mean (n_features,). Otherwise returns the mean concatenated
    with flattened covariance + outer(mean, mean), of length
    n_features + n_features squared. Both quantities describe the full distribution.
    """
    mean, covariance = (distribution.mean, distribution.covariance)
    if not second_moments:
        return mean

    raw_second = covariance + np.outer(mean, mean)
    return np.concatenate([mean.ravel(), raw_second.ravel()]).astype(np.float64)
