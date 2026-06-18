---
icon: lucide/badge-plus
---

# Covariance regularization

Symbols are grouped by topic here, but are imported from `gmm_divergence`.

Covariance regularization strategies for covariance matrices are techniques used
to ensure that the covariance matrices of the components in a GMM (or single
Gaussians) are well-conditioned and do not lead to numerical instability during
estimation. These strategies can help prevent issues such as singular covariance
matrices, which can arise when the data is not sufficiently diverse or when
there are too few data points relative to the number of parameters being
estimated.

## ::: gmm_divergence.regularize_covariance

## ::: gmm_divergence.estimate_epsilon

## ::: gmm_divergence.DiagonalLoading

## ::: gmm_divergence.LinearShrinkage

## ::: gmm_divergence.DiagonalShrinkage

## ::: gmm_divergence.EigenvalueClipping

## ::: gmm_divergence.LowRank

## ::: gmm_divergence.RelativeToTrace

## ::: gmm_divergence.TargetConditionNumber

## ::: gmm_divergence.ResidualVariance
