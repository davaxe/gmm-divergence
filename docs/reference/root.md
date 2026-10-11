---
icon: lucide/package
---

# Top-level API

The top-level `gmm_divergence` namespace is intentionally small. It exposes the
classes and functions most users reach for first, plus namespace modules for
configuration objects.

## What Lives At The Root

| Symbol | Purpose |
| --- | --- |
| `Gaussian`, `GaussianMixture` | Core Gaussian-family classes |
| `combine_distributions` | Build a combined Gaussian mixture |
| `kl_divergence`, `categorical_kl_divergence`, `symmetric_kl_divergence`, `jensen_shannon_divergence` | Main divergence helpers |
| `component_kl_matrix` | Pairwise component KL diagnostics |
| `aligned_joint_kl_divergence` | Joint KL for aligned component labels and observations |
| `fit_gaussian_weights`, `fit_categorical_weights`, `prepare_gaussian_weight_fit`, `prune_mixture` | Main fitting helpers |
| `DivergenceResult`, `GaussianWeightFitResult`, `CategoricalWeightFitResult` | Result containers |

## What Lives In Namespaces

Configuration and specialized helpers are grouped by namespace:

| Namespace | Contains |
| --- | --- |
| `gmm_divergence.divergence` | Estimator configuration such as `MonteCarlo` |
| `gmm_divergence.fitting` | Prepared fits, objectives, optimizers, and candidate selectors |
| `gmm_divergence.sampling` | Sampling specifications such as `Draw` and `Samples` |
| `gmm_divergence.covariance` | Covariance regularizers and epsilon heuristics |
| `gmm_divergence.distributions` | Gaussian-family details and combination metadata |

## Example

```python
import gmm_divergence as gd

p = gd.Gaussian.univariate(mean=0.0, variance=1.0)
q = gd.Gaussian.univariate(mean=1.0, variance=2.0)
estimator = gd.divergence.MonteCarlo(sampling=gd.sampling.Draw(n_samples=50_000, rng=0))
result = gd.kl_divergence(p, q, estimator=estimator)
```

The sections below document the root exports explicitly. Namespace-only
configuration objects are documented on their namespace pages.

## Distributions

::: gmm_divergence.Gaussian

::: gmm_divergence.GaussianMixture

::: gmm_divergence.combine_distributions

## Divergence Helpers

::: gmm_divergence.categorical_kl_divergence

::: gmm_divergence.kl_divergence

::: gmm_divergence.symmetric_kl_divergence

::: gmm_divergence.jensen_shannon_divergence

::: gmm_divergence.component_kl_matrix

::: gmm_divergence.aligned_joint_kl_divergence

## Fitting Helpers

::: gmm_divergence.fit_categorical_weights

::: gmm_divergence.prepare_gaussian_weight_fit

::: gmm_divergence.fit_gaussian_weights

::: gmm_divergence.prune_mixture

## Results

::: gmm_divergence.CategoricalWeightFitResult

::: gmm_divergence.DivergenceResult

::: gmm_divergence.GaussianWeightFitResult
