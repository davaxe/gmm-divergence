---
icon: lucide/badge-plus
---

# Covariance regularization

Covariance regularization utilities are imported from
`gmm_divergence.covariance`.

Regularization can improve covariance conditioning and repair singular empirical
estimates, for example when observations are sparse relative to feature dimension.
It does not guarantee positive definiteness for every input and configuration;
the returned matrices are validated, and invalid results are rejected.

Following the [notation conventions](index.md#notation-and-terminology),
$\Sigma$ is a symmetric $d\times d$ input matrix, $I_d$ is the identity,
$\varepsilon\geq0$ is diagonal loading, $\eta\in[0,1]$ is shrinkage strength,
$s>0$ is heuristic scale, and $r$ is retained rank. Eigenvalues are ordered
$\lambda_1\leq\cdots\leq\lambda_d$. Symbols are mapped to their Python
configuration fields in the generated docstrings below.

Regularization never modifies the input array. Every regularizer returns an
independent, read-only `float64` array.

::: gmm_divergence.covariance
    options:
        members: false
        show_root_full_path: true

Input shape determines single versus batched matrix behavior: `(n_features, n_features)`
or `(n_matrices, n_features, n_features)`. `estimate_epsilon` returns a scalar or
one value per matrix. Configure `epsilon`, heuristic `scale` and `rank`, shrinkage
`strength`, or `max_condition_number` explicitly by keyword. Shrinkage defaults
to `strength=1e-2` for both isotropic and diagonal targets.

Trace scaling and residual-variance heuristics clip negative contributions to
zero. Retaining full rank gives zero residual variance. The condition-number
heuristic computes loading from the extreme eigenvalues; an all-zero covariance
still receives zero loading, so use positive loading or eigenvalue clipping
when that case must be repaired.

```python
import gmm_divergence as gd

covariance = gd.covariance.regularize_covariance(
    [[1, 0], [0, 0]], regularizer=gd.covariance.DiagonalLoading(epsilon=0.1)
)
gaussian = gd.Gaussian([0, 0], covariance)
assert gaussian.n_features == 2
```

## Functions

::: gmm_divergence.covariance.regularize_covariance

::: gmm_divergence.covariance.estimate_epsilon

## Regularizers

::: gmm_divergence.covariance.DiagonalLoading

::: gmm_divergence.covariance.IsotropicShrinkage

::: gmm_divergence.covariance.DiagonalShrinkage

::: gmm_divergence.covariance.EigenvalueClipping

::: gmm_divergence.covariance.LowRank

## Epsilon Heuristics

::: gmm_divergence.covariance.RelativeToTrace

::: gmm_divergence.covariance.TargetConditionNumber

::: gmm_divergence.covariance.ResidualVariance


::: gmm_divergence.covariance.CovarianceRegularizer

::: gmm_divergence.covariance.EpsilonHeuristic

::: gmm_divergence.covariance.EpsilonSpec
