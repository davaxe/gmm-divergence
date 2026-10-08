---
icon: lucide/chart-no-axes-combined
---

# Fitting API

Top-level fitting helpers are documented under [Top-level API](root.md). This
page documents prepared Gaussian-mixture fits, shared simplex optimization,
optimizer and objective configurations, and
candidate-selection helpers from `gmm_divergence.fitting`.


::: gmm_divergence.fitting
    options:
        members: false
        show_root_full_path: true

## Fitting From Samples

Fit mixture parameters from observations directly with scikit-learn, then
convert the fitted estimator using `GaussianMixture.from_sklearn_gmm`.
The constructor supports all four sklearn covariance types and expands them
into full covariance matrices.

```python
import numpy as np
from sklearn.mixture import GaussianMixture

import gmm_divergence as gd

rng = np.random.default_rng(0)
samples = np.concatenate([rng.normal(-2, 1, size=(100, 1)), rng.normal(2, 0.5, size=(100, 1))])
estimator = GaussianMixture(n_components=2, covariance_type="diag", n_init=3, random_state=0).fit(
    samples
)
gmm = gd.GaussianMixture.from_sklearn_gmm(estimator)
assert gmm.n_components == 2
```

See [GaussianMixture](root.md#gmm_divergence.GaussianMixture) for the conversion
constructor and other ways to construct mixtures.

## Component Statistics

Use fixed responsibilities or a shared reference mixture to estimate empirical
component statistics. Component indices stay aligned with the supplied
assignments, and components with zero responsibility mass are rejected.
Regularization is applied only when converting statistics into a mixture.

```python
import gmm_divergence as gd

stats = gd.fitting.component_statistics(
    [[0.0], [2.0], [4.0]], responsibilities=[[1, 0], [0.5, 0.5], [0, 1]]
)
population = stats.to_gaussian_mixture(regularizer=gd.covariance.DiagonalLoading())
assert population.n_components == 2
```

::: gmm_divergence.fitting.component_statistics

::: gmm_divergence.fitting.ComponentStatistics

## Probability Validation

Categorical targets, candidate probability rows, and component responsibilities
share the same validation policy: values must be finite and nonnegative, and
sums must be one within `rtol=1e-7`, `atol=1e-8`. Inputs are copied without
normalization or negative-value clipping. Categorical KL uses the same policy.
Optimizer weight vectors retain their separate validation, which permits
unnormalized positive weights when evaluating gradients.

## Objective Configuration

Sampled objectives accept `sampling.Stratified` on either side. Each observation
from component $k$ receives integration weight $\pi_k/n_k$, where $n_k$ is the
number of draws allocated to that component. Objective values and gradients
therefore preserve the mixture weights even when rounded sample counts differ
from the component proportions. `Draw`, `Samples`, and `SampleBatches` use
ordinary sample means.

::: gmm_divergence.fitting.ForwardKL

::: gmm_divergence.fitting.ReverseKL

::: gmm_divergence.fitting.BidirectionalKL

::: gmm_divergence.fitting.JensenShannon

::: gmm_divergence.fitting.MomentMatching

## Optimizers

::: gmm_divergence.fitting.SoftmaxLBFGSB

::: gmm_divergence.fitting.SimplexSLSQP

## Prepared Gaussian-Mixture Fits

::: gmm_divergence.fitting.prepare_gaussian_mixture_fit

::: gmm_divergence.fitting.PreparedGaussianMixtureFit

## Shared Simplex Optimization

`PreparedGaussianMixtureFit.solve` returns a `SimplexOptimizationResult` with
optimizer coordinates, active weights, and termination metadata. The shared
optimizer works with an objective and gradient over simplex weights; Gaussian
preparation and reporting remain separate.

::: gmm_divergence.fitting.SimplexOptimizationResult

## Fitting Results

::: gmm_divergence.results.GaussianMixtureFitResult

::: gmm_divergence.results.CategoricalMixtureFitResult

## Candidate Selectors

::: gmm_divergence.fitting.score_candidates

::: gmm_divergence.fitting.rank_candidates

::: gmm_divergence.fitting.TopKSelector

::: gmm_divergence.fitting.ThresholdSelector

::: gmm_divergence.fitting.ToleranceSelector

::: gmm_divergence.fitting.QuantileSelector
