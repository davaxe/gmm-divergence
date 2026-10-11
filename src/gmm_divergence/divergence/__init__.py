"""Public divergence API."""

from gmm_divergence.divergence._api import (
    aligned_joint_kl_divergence,
    categorical_kl_divergence,
    component_kl_matrix,
    jensen_shannon_divergence,
    kl_divergence,
    symmetric_kl_divergence,
)
from gmm_divergence.divergence._options import (
    ClosedForm,
    KLEstimator,
    MomentMatchedGaussian,
    MonteCarlo,
    Unscented,
    Variational,
)
from gmm_divergence.results import AlignedJointKLResult, DivergenceResult, MonteCarloStatistics

__all__ = [
    "AlignedJointKLResult",
    "ClosedForm",
    "DivergenceResult",
    "KLEstimator",
    "MomentMatchedGaussian",
    "MonteCarlo",
    "MonteCarloStatistics",
    "Unscented",
    "Variational",
    "aligned_joint_kl_divergence",
    "categorical_kl_divergence",
    "component_kl_matrix",
    "jensen_shannon_divergence",
    "kl_divergence",
    "symmetric_kl_divergence",
]
