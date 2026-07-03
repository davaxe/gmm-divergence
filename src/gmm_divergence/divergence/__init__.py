"""Public divergence API."""

from gmm_divergence.divergence._api import (
    component_kl_matrix,
    estimate_divergence,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.divergence._options import (
    ClosedForm,
    DivergenceName,
    DivergenceSpec,
    JensenShannonDivergence,
    KLDivergence,
    KLMethod,
    MomentMatchedGaussian,
    MonteCarlo,
    SymmetricKLDivergence,
    Unscented,
    Variational,
)

__all__ = [
    "ClosedForm",
    "DivergenceName",
    "DivergenceSpec",
    "JensenShannonDivergence",
    "KLDivergence",
    "KLMethod",
    "MomentMatchedGaussian",
    "MonteCarlo",
    "SymmetricKLDivergence",
    "Unscented",
    "Variational",
    "component_kl_matrix",
    "estimate_divergence",
    "jensen_shannon_divergence",
    "kl_divergence",
    "symmetric_kl_divergence",
]
