# gmm-divergence

Tools for comparing Gaussian mixtures and fitting mixture weights, with support
for categorical distributions.

This package is under development. APIs, estimators, and numerical behavior may change between early releases.

[Read the documentation →](https://davaxe.github.io/gmm-divergence/)

## Current Scope

The package currently includes:

- Typed Gaussian and Gaussian mixture representations with density evaluation,
  sampling, component responsibilities, and scikit-learn conversion
- KL divergence estimators based on closed-form Gaussian KL, Monte Carlo sampling,
  unscented sigma points, Gaussian approximations, and variational bounds
- Explicit sampling controls for drawn, reused, and stratified Monte Carlo samples
- Symmetric KL, Jensen-Shannon divergence, categorical KL, and aligned-component
  joint KL with a weight/component decomposition
- Gaussian-mixture weight fitting with forward, reverse, bidirectional,
  Jensen-Shannon, and moment-matching objectives; categorical-mixture weight
  fitting with forward KL
- Softmax L-BFGS-B and simplex-constrained SLSQP optimizers, reusable Gaussian
  fitting objectives, and candidate selection
- Responsibility-weighted empirical component statistics
- Covariance regularization utilities for diagonal loading, shrinkage, eigenvalue clipping,
  and low-rank approximation

## Installation

This project is not yet intended for stable production use. A pre-release is available from PyPI:

```bash
python -m pip install gmm-divergence
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

result = gd.kl_divergence(
    p, q, estimator=gd.divergence.MonteCarlo(sampling=gd.sampling.Draw(50_000, rng=0))
)
print(result.value, result.monte_carlo_stats.standard_error)
```

The top-level module keeps the common distribution classes and primary helper
functions. Configuration objects are grouped by domain, for example
`gd.divergence.MonteCarlo`, `gd.sampling.Draw`, `gd.fitting.ForwardKL`, and
`gd.covariance.DiagonalLoading`.

## Fitting Mixture Weights

```python
candidates = [
    gd.Gaussian.univariate(mean=-1.0, variance=0.5),
    gd.Gaussian.univariate(mean=1.5, variance=1.0),
]

fit = gd.fit_gaussian_mixture_weights(
    p,
    candidates,
    method=gd.fitting.SoftmaxLBFGSB(),
    objective=gd.fitting.MomentMatching(fit_second_moments=True),
)
print(fit.weights)
```

Fit a mixture of categorical distributions using the same optimizer options:

```python
fit = gd.fitting.fit_categorical_mixture_weights(
    [0.4, 0.6],
    [[0.8, 0.2], [0.1, 0.9]],
    method=gd.fitting.SimplexSLSQP(),
)
print(fit.weights)
```
