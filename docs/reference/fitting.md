---
icon: lucide/chart-no-axes-combined
---

# Fitting API

Top-level fitting helpers are documented under [Top-level API](root.md). This
page documents prepared fits, optimizer and objective configurations, and
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

## Objective Configuration

::: gmm_divergence.fitting.ForwardKL

::: gmm_divergence.fitting.ReverseKL

::: gmm_divergence.fitting.BidirectionalKL

::: gmm_divergence.fitting.JensenShannon

::: gmm_divergence.fitting.MomentMatching

## Optimizers

::: gmm_divergence.fitting.SoftmaxLBFGSB

::: gmm_divergence.fitting.SimplexSLSQP

## Prepared Fits

::: gmm_divergence.fitting.prepare_mixture_weight_fit

::: gmm_divergence.fitting.PreparedFit

::: gmm_divergence.fitting.FitSolution

## Candidate Selectors

::: gmm_divergence.fitting.score_candidates

::: gmm_divergence.fitting.rank_candidates

::: gmm_divergence.fitting.TopKSelector

::: gmm_divergence.fitting.ThresholdSelector

::: gmm_divergence.fitting.ToleranceSelector

::: gmm_divergence.fitting.QuantileSelector
