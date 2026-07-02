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

::: gmm_divergence.covariance
    options:
        members: false
        show_root_full_path: true

## Functions

::: gmm_divergence.covariance.regularize_covariance

::: gmm_divergence.covariance.estimate_epsilon

## Regularizers

::: gmm_divergence.covariance.DiagonalLoading

::: gmm_divergence.covariance.LinearShrinkage

::: gmm_divergence.covariance.DiagonalShrinkage

::: gmm_divergence.covariance.EigenvalueClipping

::: gmm_divergence.covariance.LowRank

## Epsilon Heuristics

::: gmm_divergence.covariance.RelativeToTrace

::: gmm_divergence.covariance.TargetConditionNumber

::: gmm_divergence.covariance.ResidualVariance
