---
icon: material/rocket-launch
---

# Getting started

## Install

`gmm-divergence` is under active development. For local development, install it
from a checkout:

```bash
pip install -e .
```

There is also a pre-release version available on PyPI:

```bash
pip install gmm-divergence
```

!!! warning "Pre-release software"
    The PyPI version is a pre-release and **may contain bugs or incomplete features**.

    Breaking changes may also occur without warning.

## Create Gaussian mixtures

Define Gaussian mixtures from component weights, means, and covariances:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.0, variance=1.0),
        gd.Gaussian.univariate(mean=1.0, variance=1.0),
    ]
)

q = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.5, variance=1.0),
        gd.Gaussian.univariate(mean=2.5, variance=0.5),
    ]
)
```

## Estimate KL divergence

Use `kl_divergence` to estimate \(D_{\mathrm{KL}}(p \| q)\):

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.0, variance=1.0),
        gd.Gaussian.univariate(mean=1.0, variance=1.0),
    ]
)

q = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.5, variance=1.0),
        gd.Gaussian.univariate(mean=2.5, variance=0.5),
    ]
)

result = gd.kl_divergence(p, q)
print(result.value)
```

The result includes the estimated value and metadata about the estimator.

## Sample from a mixture

Mixtures can generate samples and evaluate log densities:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.0, variance=1.0),
        gd.Gaussian.univariate(mean=1.0, variance=1.0),
    ]
)

samples = p.sample(1000, rng=9126)
log_density = p.logpdf(samples)
```

## Fit mixture weights

Use `fit_mixture_weights` to combine candidate mixtures against a reference distribution:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.0, variance=0.5),
        gd.Gaussian.univariate(mean=2.0, variance=0.5),
    ],
    weights=[0.6, 0.4],
)
q1 = gd.Gaussian.univariate(mean=0.0, variance=0.5)
q2 = gd.Gaussian.univariate(mean=2.0, variance=0.5)

fit = gd.fit_mixture_weights(p, [q1, q2])
print(fit.weights)  # [~0.6, ~0.4]
```

For the underlying formulation and more complete examples, see
[KL estimation](kl_estimation.md), [mixture weight fitting](fit_weights.md), and
the [API reference](reference/index.md).
