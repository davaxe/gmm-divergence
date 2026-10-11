---
icon: lucide/badge-plus
---

# Covariance regularization

Covariance regularization utilities are imported from
`gmm_divergence.covariance`.

Covariance regularization strategies for covariance matrices are techniques used
to ensure that the covariance matrices of the components in a GMM (or single
Gaussians) are well-conditioned and do not lead to numerical instability during
estimation. These strategies can help prevent issues such as singular covariance
matrices, which can arise when the data is not sufficiently diverse or when
there are too few data points relative to the number of parameters being
estimated.

Regularization never modifies the input array. Every regularizer returns an
independent, read-only `float64` array.

::: gmm_divergence.covariance
    options:
        members: false
        show_root_full_path: true

Input shape determines scalar versus batched behavior: `(n_features, n_features)`
or `(n_matrices, n_features, n_features)`. `estimate_epsilon` returns a scalar or
one value per matrix. Configure `epsilon`, heuristic `scale` and `rank`, shrinkage
`strength`, or `max_condition_number` explicitly by keyword. Shrinkage defaults
to `strength=1e-2` for both isotropic and diagonal targets.

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
