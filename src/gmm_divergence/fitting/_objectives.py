"""Objective functions for fitting Gaussian mixtures via divergence minimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence._core._numeric import logsumexp
from gmm_divergence._core._types import FloatArray
from gmm_divergence.distributions._base import GaussianFamily, gaussian_family_raw_moment_vector
from gmm_divergence.fitting._options import (
    BidirectionalKL,
    FitParameterization,
    ForwardKL,
    JensenShannon,
    MomentMatching,
    ReverseKL,
)

if TYPE_CHECKING:
    from gmm_divergence._core._types import Weights
    from gmm_divergence.distributions._gaussian import Gaussian
    from gmm_divergence.distributions._mixture import GaussianMixture


GaussianLike = GaussianFamily
ObjectiveFn = Callable[[FloatArray], tuple[float, FloatArray]]


def softmax(theta: FloatArray) -> Weights:
    """Numerically stable softmax."""
    theta = np.asarray(theta, dtype=np.float64)
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float64)


def logpdf_matrix(components: Sequence[GaussianLike], samples: FloatArray) -> FloatArray:
    """Evaluate each component log-density at each sample."""
    samples = np.asarray(samples, dtype=np.float64)
    log_q = np.empty((samples.shape[0], len(components)), dtype=np.float64)
    for k, qk in enumerate(components):
        log_q[:, k] = qk.logpdf(samples)
    return log_q


def mixture_stats(
    log_q: FloatArray, weights: FloatArray, *, eps: float = 1e-300
) -> tuple[FloatArray, FloatArray]:
    """Return mixture log-density and component responsibilities."""
    log_w = np.log(np.maximum(np.asarray(weights, dtype=np.float64), eps))
    log_terms = np.asarray(log_q, dtype=np.float64) + log_w[None, :]
    log_qw = logsumexp(log_terms, axis=1)
    responsibilities = np.exp(log_terms - log_qw[:, None])
    return log_qw.astype(np.float64), responsibilities.astype(np.float64)


def with_softmax(simplex_objective: ObjectiveFn) -> ObjectiveFn:
    """Wrap a simplex objective as an objective over softmax logits."""

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        weights = softmax(theta)
        value, grad_w = simplex_objective(weights)
        grad_w = np.asarray(grad_w, dtype=np.float64)
        grad_theta = weights * (grad_w - np.dot(weights, grad_w))
        return float(value), grad_theta.astype(np.float64)

    return objective


@dataclass(frozen=True, slots=True)
class _ForwardKL:
    """Monte Carlo forward KL objective over simplex weights."""

    log_q_on_p_samples: FloatArray
    log_p_on_p_samples: FloatArray | None = None
    eps: float = 1e-300
    include_constant: bool = False

    @property
    def n_components(self) -> int:
        return int(self.log_q_on_p_samples.shape[1])

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        weights = np.asarray(weights, dtype=np.float64)
        log_qw, responsibilities = mixture_stats(self.log_q_on_p_samples, weights, eps=self.eps)

        value = -float(np.mean(log_qw))
        if self.include_constant and self.log_p_on_p_samples is not None:
            value += float(np.mean(self.log_p_on_p_samples))

        weights_safe = np.maximum(weights, self.eps)
        grad = -np.mean(responsibilities / weights_safe[None, :], axis=0)
        return float(value), grad.astype(np.float64)


def forward_kl(
    p: GaussianLike | None,
    q_components: Sequence[GaussianLike],
    p_samples: FloatArray,
    *,
    include_constant: bool = False,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a simplex objective for forward KL, `KL(p || q_w)`."""
    p_samples = np.asarray(p_samples, dtype=np.float64)
    log_q_on_p_samples = logpdf_matrix(q_components, p_samples)

    log_p_on_p_samples = None
    if include_constant:
        if p is None:
            msg = "p is required when include_constant=True."
            raise ValueError(msg)
        log_p_on_p_samples = np.asarray(p.logpdf(p_samples), dtype=np.float64)

    return _ForwardKL(
        log_q_on_p_samples=log_q_on_p_samples,
        log_p_on_p_samples=log_p_on_p_samples,
        eps=eps,
        include_constant=include_constant,
    )


@dataclass(frozen=True, slots=True)
class _ReverseKL:
    """Fixed-sample reverse KL objective over simplex weights."""

    log_q_on_q_samples: FloatArray
    log_p_on_q_samples: FloatArray
    eps: float = 1e-300

    @property
    def n_components(self) -> int:
        return int(self.log_q_on_q_samples.shape[0])

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        weights = np.asarray(weights, dtype=np.float64)
        log_w = np.log(np.maximum(weights, self.eps))

        log_qw = logsumexp(self.log_q_on_q_samples + log_w[None, None, :], axis=2)
        component_terms = np.mean(log_qw - self.log_p_on_q_samples, axis=1)
        value = float(np.dot(weights, component_terms))

        correction = np.zeros(self.n_components, dtype=np.float64)
        for i in range(self.n_components):
            ratio_i = np.exp(self.log_q_on_q_samples[i] - log_qw[i, :, None])
            correction += weights[i] * np.mean(ratio_i, axis=0)

        grad = component_terms + correction
        return float(value), grad.astype(np.float64)


def reverse_kl(
    p: GaussianLike,
    q_components: Sequence[GaussianLike],
    q_samples: FloatArray,
    *,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a simplex objective for fixed-sample reverse KL, `KL(q_w || p)`."""
    log_p_on_q_samples, log_q_on_q_samples = _candidate_sample_logpdfs(p, q_components, q_samples)

    return _ReverseKL(
        log_q_on_q_samples=log_q_on_q_samples, log_p_on_q_samples=log_p_on_q_samples, eps=eps
    )


@dataclass(frozen=True, slots=True)
class _JensenShannon:
    """Fixed-sample Jensen-Shannon objective over simplex weights."""

    log_q_on_p_samples: FloatArray
    log_p_on_p_samples: FloatArray
    log_q_on_q_samples: FloatArray
    log_p_on_q_samples: FloatArray
    eps: float = 1e-300

    @property
    def n_components(self) -> int:
        return int(self.log_q_on_q_samples.shape[0])

    def __call__(self, weights: FloatArray) -> tuple[float, FloatArray]:
        w = np.asarray(weights, dtype=np.float64)
        log_2: float = np.log(2.0)
        log_qw_p, _ = mixture_stats(self.log_q_on_p_samples, w, eps=self.eps)
        log_m_p = np.logaddexp(self.log_p_on_p_samples, log_qw_p) - log_2
        log_w = np.log(np.maximum(w, self.eps))
        log_qw_q = logsumexp(self.log_q_on_q_samples + log_w, axis=-1)
        log_m_q = np.logaddexp(self.log_p_on_q_samples, log_qw_q) - log_2
        forward_kl = np.mean(self.log_p_on_p_samples - log_m_p)
        reverse_kl_per_component = np.mean(log_qw_q - log_m_q, axis=1)
        value = 0.5 * (forward_kl + np.dot(w, reverse_kl_per_component))
        forward_grad = -0.25 * np.mean(np.exp(self.log_q_on_p_samples - log_m_p[:, None]), axis=0)
        mean_q_over_qw = np.mean(np.exp(self.log_q_on_q_samples - log_qw_q[..., None]), axis=1)
        mean_q_over_m = np.mean(np.exp(self.log_q_on_q_samples - log_m_q[..., None]), axis=1)
        reverse_grad = (
            0.5 * reverse_kl_per_component + 0.5 * (w @ mean_q_over_qw) - 0.25 * (w @ mean_q_over_m)
        )
        return float(value), forward_grad + reverse_grad


def jensen_shannon(
    p: GaussianLike,
    q_components: Sequence[GaussianLike],
    p_samples: FloatArray,
    q_samples: FloatArray,
    *,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a simplex objective for Jensen-Shannon divergence."""
    p_samples = np.asarray(p_samples, dtype=np.float64)
    log_p_on_q_samples, log_q_on_q_samples = _candidate_sample_logpdfs(p, q_components, q_samples)

    log_q_on_p_samples = logpdf_matrix(q_components, p_samples)
    log_p_on_p_samples = np.asarray(p.logpdf(p_samples), dtype=np.float64)

    return _JensenShannon(
        log_q_on_p_samples=log_q_on_p_samples,
        log_p_on_p_samples=log_p_on_p_samples,
        log_q_on_q_samples=log_q_on_q_samples,
        log_p_on_q_samples=log_p_on_q_samples,
        eps=eps,
    )


def _candidate_sample_logpdfs(
    p: GaussianLike, q_components: Sequence[GaussianLike], q_samples: FloatArray
) -> tuple[FloatArray, FloatArray]:
    q_samples = np.asarray(q_samples, dtype=np.float64)
    n_components, n_samples, dim = q_samples.shape
    q_samples_flat = q_samples.reshape(-1, dim)

    log_p_on_q_samples = np.asarray(p.logpdf(q_samples_flat), dtype=np.float64).reshape(
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

        for objective, alpha in zip(self.objectives, self.weights, strict=True):
            value, grad = objective(w)
            total_value += float(alpha) * float(value)
            total_grad += float(alpha) * np.asarray(grad, dtype=np.float64)

        return float(total_value), total_grad.astype(np.float64)


def bidirectional_kl(
    p: GaussianLike,
    q_components: Sequence[GaussianLike],
    p_samples: FloatArray | None = None,
    q_samples: FloatArray | None = None,
    *,
    alpha: float = 0.5,
    include_forward_constant: bool = False,
    eps: float = 1e-300,
) -> ObjectiveFn:
    """Build a weighted forward/reverse KL objective over simplex weights."""
    objectives: list[ObjectiveFn] = []
    objective_weights: list[float] = []
    forward_weight = alpha
    reverse_weight = 1.0 - alpha
    if forward_weight:
        if p_samples is None:
            msg = "p_samples is required when forward_weight is nonzero."
            raise ValueError(msg)
        objectives.append(
            forward_kl(
                p=p,
                q_components=q_components,
                p_samples=p_samples,
                include_constant=include_forward_constant,
                eps=eps,
            )
        )
        objective_weights.append(float(forward_weight))

    if reverse_weight:
        if q_samples is None:
            msg = "q_samples is required when reverse_weight is nonzero."
            raise ValueError(msg)
        objectives.append(reverse_kl(p=p, q_components=q_components, q_samples=q_samples, eps=eps))
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
    p: GaussianLike, q_components: Sequence[GaussianLike], *, second_moments: bool = False
) -> ObjectiveFn:
    """Build a simplex objective for moment matching."""
    p_moments = gaussian_family_raw_moment_vector(p, second_moments=second_moments)
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


def _build_simplex_objective(
    *,
    objective: ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    p_samples: FloatArray,
    q_samples: FloatArray | None,
) -> ObjectiveFn:
    match objective:
        case ForwardKL():
            return forward_kl(p, q_i, p_samples)
        case ReverseKL():
            if q_samples is None:
                msg = "q_samples is required for reverse KL."
                raise ValueError(msg)
            return reverse_kl(p, q_i, q_samples)
        case BidirectionalKL(alpha=alpha):
            return bidirectional_kl(p, q_i, p_samples=p_samples, q_samples=q_samples, alpha=alpha)
        case JensenShannon():
            if q_samples is None:
                msg = "q_samples is required for Jensen-Shannon."
                raise ValueError(msg)
            return jensen_shannon(p, q_i, p_samples=p_samples, q_samples=q_samples)
        case MomentMatching(fit_second_moments=fit_second_moments):
            return moment_matching(p, q_i, second_moments=fit_second_moments)


def build_objective(
    *,
    parameterization: FitParameterization,
    objective: ForwardKL | ReverseKL | BidirectionalKL | JensenShannon | MomentMatching,
    p: Gaussian | GaussianMixture,
    q_i: Sequence[Gaussian | GaussianMixture],
    p_samples: FloatArray,
    q_samples: FloatArray | None,
) -> ObjectiveFn:
    simplex_objective = _build_simplex_objective(
        objective=objective, p=p, q_i=q_i, p_samples=p_samples, q_samples=q_samples
    )
    match parameterization:
        case "softmax":
            return with_softmax(simplex_objective)
        case "simplex":
            return simplex_objective
