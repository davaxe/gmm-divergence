---
icon: material/rocket-launch
---

# Getting started

## Install

`gmm-divergence` is under active development. For local development, install it
from a checkout:

```bash
python -m pip install -e .
```

## Create Gaussian mixtures

Define Gaussian mixtures from component weights, means, and covariances:

```python
from gmm_divergence import GaussianMixture

p = GaussianMixture.from_arrays(
    weights=[0.5, 0.5], means=[[0.0], [1.0]], covariances=[[[1.0]], [[1.0]]]
)

q = GaussianMixture.from_arrays(
    weights=[0.5, 0.5], means=[[0.5], [2.5]], covariances=[[[1.0]], [[0.5]]]
)
```

## Estimate KL divergence

Use `kl_divergence` to estimate \(D_{\mathrm{KL}}(p \| q)\):

```python
from gmm_divergence import kl_divergence

result = kl_divergence(p, q, method="monte_carlo", rng=9126)
print(result.value)
```

The result includes the estimated value and metadata about the estimator.

## Sample from a mixture

Mixtures can generate samples and evaluate log densities:

```python
samples = p.sample(1000, rng=9126)
log_density = p.logpdf(samples)
```

## Fit mixture weights

Use `fit_mixture_weights` to combine candidate mixtures against a reference distribution:

```python
from gmm_divergence import fit_mixture_weights

p = GaussianMixture.from_arrays(
    weights=[0.6, 0.4], means=[[0.0], [2.0]], covariances=[[[0.5]], [[0.5]]]
)

left = GaussianMixture.from_arrays(weights=[1.0], means=[[0.0]], covariances=[[[0.5]]])

right = GaussianMixture.from_arrays(weights=[1.0], means=[[2.0]], covariances=[[[0.5]]])

fit = fit_mixture_weights(p, [left, right], rng=9126)
print(fit.weights)
```

For the underlying formulation and more complete examples, see
[KL estimation](kl_estimation.md), [mixture weight fitting](fit_weights.md), and
the [API reference](reference/index.md).
