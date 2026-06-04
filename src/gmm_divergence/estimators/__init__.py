"""KL divergence estimators.

This modules contains the available strategies for estimating
``D_KL(p || q)`` between Gaussian and Gaussian-mixture distributions.

The main estimators are:

- ``kl_closed_form``:
  Exact KL divergence for the special case where both inputs are single
  Gaussian distributions.
- ``kl_gaussian_approximation``:
  Fast approximation based on reducing mixtures to Gaussian surrogates.
- ``kl_monte_carlo``:
  Sampling-based estimator using points drawn from the reference
  distribution.
- ``kl_unscented``:
  Deterministic estimator based on sigma points from the reference
  distribution.
"""

from gmm_divergence.estimators.closed_form import kl_closed_form
from gmm_divergence.estimators.gaussian_approx import kl_gaussian_approximation
from gmm_divergence.estimators.monte_carlo import kl_monte_carlo
from gmm_divergence.estimators.unscented import kl_unscented

__all__ = ["kl_closed_form", "kl_gaussian_approximation", "kl_monte_carlo", "kl_unscented"]
