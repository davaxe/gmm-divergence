"""Public divergence API."""

from gmm_divergence.divergence._api import (
    component_kl_matrix,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.divergence._options import (
    ClosedForm,
    GaussianApproximation,
    KLMethod,
    MonteCarlo,
    Unscented,
    Variational,
)

__all__ = [
    "ClosedForm",
    "GaussianApproximation",
    "KLMethod",
    "MonteCarlo",
    "Unscented",
    "Variational",
    "component_kl_matrix",
    "jensen_shannon_divergence",
    "kl_divergence",
    "symmetric_kl_divergence",
]
