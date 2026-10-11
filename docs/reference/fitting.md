---
icon: lucide/chart-no-axes-combined
---

# Fitting API

Top-level fitting helpers are documented under [Top-level API](root.md). This
page documents prepared Gaussian-mixture fits,
optimizer and objective configurations, and
candidate-selection helpers from `gmm_divergence.fitting`.


::: gmm_divergence.fitting
    options:
        members: false
        show_root_full_path: true

## Fitting From Samples

Fit mixture parameters from observations directly with scikit-learn, then
convert the fitted estimator using `GaussianMixture.from_sklearn`.
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
gmm = gd.GaussianMixture.from_sklearn(estimator)
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
from the component proportions. Raw `Draw`, `Samples`, and `SampleBatches`
observations use ordinary sample means. Precomputed `SampleBatch` records retain
their integration weights and component IDs.

::: gmm_divergence.fitting.ForwardKL

::: gmm_divergence.fitting.ReverseKL

::: gmm_divergence.fitting.BidirectionalKL

::: gmm_divergence.fitting.JensenShannon

::: gmm_divergence.fitting.MomentMatching

## Optimizers

::: gmm_divergence.fitting.SoftmaxLBFGSB

::: gmm_divergence.fitting.SimplexSLSQP

## Prepared Gaussian Weight Fits

`prepared.fit(optimizer=...)` is the direct route to a complete result.
`prepared.evaluate(weights)` returns the cached objective value and gradient.
For an SLSQP warm start, use the previous result's weights at
`prepared.active_candidate_indices` as `initial_weights`.
Forward and bidirectional KL values include the cached target log-density
constant, so `objective_value` estimates the mathematical objective in nats.

Gaussian results expose `mixture` directly. Its mapping refers to original
candidate indices, including after filtering or reordering. Result `weights`
include zero entries for rejected candidates. Categorical results expose
read-only `probabilities` and use the fixed `objective="forward_kl"`.
Both result types provide `optimizer`, `objective_value`, `converged`,
`n_iterations`, and `message`.

Precomputed candidate batches must contain one batch per **original** candidate.
Preparation validates the full sequence, then selects batches alongside the
retained candidates. Generated samples are drawn only from retained candidates.


::: gmm_divergence.fitting.prepare_gaussian_weight_fit

::: gmm_divergence.fitting.PreparedGaussianWeightFit

## Fitting Results

::: gmm_divergence.results.GaussianWeightFitResult

::: gmm_divergence.results.CategoricalWeightFitResult

## Candidate Scoring and Selection

Choose an explicit `scorer`. `ForwardKLScore` samples from the target;
`ReverseKLScore` samples from each candidate. Reverse scoring accepts a reusable
generated estimator or one estimator per original candidate. Precomputed
reverse observations require the per-candidate form, preventing accidental
reuse of target samples. `BidirectionalKLScore` configures both directions and
their `forward_weight`. Ranking preserves input order for tied scores.

```python
import gmm_divergence as gd

target = gd.Gaussian.univariate()
candidates = [gd.Gaussian.univariate(10), gd.Gaussian.univariate(-4)]
scorer = gd.fitting.ReverseKLScore(
    estimators=[
        gd.divergence.MonteCarlo(sampling=gd.sampling.Samples([[10]])),
        gd.divergence.MonteCarlo(sampling=gd.sampling.Samples([[-4]])),
    ]
)
assert gd.fitting.score_candidates(target, candidates, scorer=scorer).tolist() == [50, 8]
selector = gd.fitting.TopKSelector(k=1, scorer=scorer)
fit = gd.fit_gaussian_weights(
    target,
    candidates,
    objective=gd.fitting.MomentMatching(),
    optimizer=gd.fitting.SimplexSLSQP(),
    candidate_selector=selector,
)
assert fit.active_candidate_indices == (1,)
```

Custom scorers implement `CandidateScorer.score(target, candidates)` and custom
selectors implement `CandidateSelector.select(target, candidates)`, returning a
complete selected/rejected partition of the original indices.

::: gmm_divergence.fitting.ForwardKLScore

::: gmm_divergence.fitting.ReverseKLScore

::: gmm_divergence.fitting.BidirectionalKLScore

::: gmm_divergence.fitting.CandidateScorer

::: gmm_divergence.fitting.CandidateSelector

::: gmm_divergence.fitting.CandidateSelection

::: gmm_divergence.fitting.Optimizer

::: gmm_divergence.fitting.FitObjective


::: gmm_divergence.fitting.score_candidates

::: gmm_divergence.fitting.rank_candidates

::: gmm_divergence.fitting.TopKSelector

::: gmm_divergence.fitting.ThresholdSelector

::: gmm_divergence.fitting.ToleranceSelector

::: gmm_divergence.fitting.QuantileSelector
