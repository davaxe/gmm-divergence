from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from gmm_divergence.covariance._epsilon import EpsilonSpec


@dataclass(frozen=True, slots=True)
class DiagonalLoading:
    r"""Diagonal loading regularization configuration.

    Applies the regularizer

    $$
    \Sigma_{\mathrm{reg}} = \Sigma + \varepsilon I,
    $$

    where $\Sigma$ is the input covariance, $I$ is the identity matrix, and
    $\varepsilon \ge 0$ is the diagonal loading amount.
    """

    eps: EpsilonSpec = 1e-6
    """Diagonal loading amount or epsilon heuristic."""


@dataclass(frozen=True, slots=True)
class LinearShrinkage:
    r"""Linear shrinkage toward an isotropic covariance target.

    Applies the shrinkage

    $$
    \Sigma_{\mathrm{reg}}
    =
    (1-\alpha)\,\Sigma + \alpha\,\tau I,
    \qquad
    \tau = \frac{\mathrm{tr}(\Sigma)}{d},
    $$

    where $d$ is the dimensionality and $\tau I$ is an isotropic covariance
    with the same average marginal variance as $\Sigma$.
    """

    alpha: float = 1e-2
    """Interpolation weight between the covariance and isotropic target."""


@dataclass(frozen=True, slots=True)
class DiagonalShrinkage:
    r"""Shrinkage toward the covariance diagonal.

    Applies the shrinkage

    $$
    \Sigma_{\mathrm{reg}}
    =
    (1-\alpha)\,\Sigma + \alpha\,\mathrm{diag}(\Sigma),
    $$

    where $\mathrm{diag}(\Sigma)$ is the diagonal matrix containing the
    marginal variances of $\Sigma$.
    """

    alpha: float = 1e-2
    """Interpolation weight between the covariance and diagonal target."""


@dataclass(frozen=True, slots=True)
class EigenvalueClipping:
    r"""Eigenvalue-clipping regularization configuration.

    Given an eigendecomposition

    $$
    \Sigma = Q \, \mathrm{diag}(\lambda_1, \ldots, \lambda_d) \, Q^T,
    $$

    this method returns

    $$
    \Sigma_{\mathrm{reg}}
    =
    Q \, \mathrm{diag}(\max(\lambda_i, \lambda_{\min})) \, Q^T,
    $$

    where $\lambda_{\min} > 0$ is the eigenvalue floor.
    """

    min_eigenvalue: float = 1e-6
    """Smallest allowed eigenvalue after clipping."""


@dataclass(frozen=True, slots=True)
class LowRank:
    r"""Low-rank covariance approximation with diagonal loading.

    For an eigendecomposition

    $$
    \Sigma = U \, \mathrm{diag}(\lambda_1, \ldots, \lambda_d) \, U^T,
    $$

    the approximation keeps the leading `rank` eigenpairs and returns

    $$
    \Sigma_{\mathrm{reg}}
    =
    U_r \, \Lambda_r \, U_r^T + \varepsilon I,
    $$

    where $U_r$ contains the leading eigenvectors, $\Lambda_r$ the
    corresponding eigenvalues, and $\varepsilon \ge 0$ is the diagonal loading
    applied after truncation.
    """

    rank: int = 1
    """Target rank for the approximation."""
    eps: EpsilonSpec = 1e-6
    """Diagonal loading amount or epsilon heuristic."""


CovarianceRegularizationMethod: TypeAlias = Literal[
    "diagonal_loading", "linear_shrinkage", "diagonal_shrinkage", "eigenvalue_clipping", "lowrank"
]

CovarianceRegularizer: TypeAlias = (
    CovarianceRegularizationMethod
    | DiagonalLoading
    | LinearShrinkage
    | DiagonalShrinkage
    | EigenvalueClipping
    | LowRank
)
