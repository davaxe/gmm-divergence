# gmm-divergence

Utilities for estimating divergences between Gaussian mixture models.

This package is under development. APIs, estimators, and numerical behavior may change between early releases.

## Current Scope

The package currently includes:

- Typed Gaussian and Gaussian mixture representations
- Density and log-density evaluation for Gaussian mixtures
- Sampling from Gaussian mixtures
- KL divergence estimators based on closed-form Gaussian KL, Monte Carlo sampling,
  unscented sigma points, Gaussian approximations, and variational bounds
- Mixture-weight fitting with forward, reverse, bidirectional, and moment-matching objectives
- Covariance regularization utilities for diagonal loading, shrinkage, eigenvalue clipping,
  and low-rank approximation

## Installation

This project is not yet intended for stable production use. A pre-release is available from PyPI:

```bash
python -m pip install gmm-divergence
```

For development, install it from a local checkout:

```bash
python -m pip install -e .
```

## Quick Example

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    [
        gd.Gaussian.univariate(mean=-1.0, variance=0.5),
        gd.Gaussian.univariate(mean=1.5, variance=1.0),
    ],
    weights=[0.4, 0.6],
)
q = gd.GaussianMixture.from_components([
    gd.Gaussian.univariate(mean=-0.8, variance=0.7),
    gd.Gaussian.univariate(mean=1.8, variance=1.2),
])

result = gd.kl_divergence(p, q, method=gd.MonteCarlo(sampling=50_000, rng=0))
print(result.value, result.monte_carlo_stats.standard_error)
```

## Fitting Mixture Weights

```python
candidates = [
    gd.Gaussian.univariate(mean=-1.0, variance=0.5),
    gd.Gaussian.univariate(mean=1.5, variance=1.0),
]

fit = gd.fit_mixture_weights(p, candidates, objective=gd.MomentMatching(fit_second_moments=True))
print(fit.weights)
```
