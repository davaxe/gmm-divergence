"""Mixture-weight fitting routines.

This module provides methods for fitting nonnegative weights for a fixed set
of Gaussian or Gaussian-mixture components so that their weighted combination
best matches a target distribution in forward KL divergence.

The exported routines differ only in how the simplex-constrained weights are
optimized:

- ``fit_mixture_weights_softmax``:
  Uses an unconstrained softmax parameterization with L-BFGS-B.
- ``fit_mixture_weights_simplex``:
  Optimizes the weights directly on the simplex with SLSQP constraints.
- ``fit_mixture_weights_em``:
  Uses EM-style responsibility updates with fixed component parameters.
"""

from gmm_divergence.fitting.weights import (
    fit_mixture_weights_em,
    fit_mixture_weights_simplex,
    fit_mixture_weights_softmax,
)

__all__ = ["fit_mixture_weights_em", "fit_mixture_weights_simplex", "fit_mixture_weights_softmax"]
