---
icon: lucide/dices
---

# Sampling API

Sampling configuration lives in `gmm_divergence.sampling`. These objects are
used by Monte Carlo divergence estimators and sampled fitting objectives.

Distribution `.sample(n_samples, rng=...)` returns an array of shape
`(n_samples, n_features)`. Sampling specifications return immutable `SampleBatch`
records with `samples`, optional normalized `integration_weights`, and optional
`component_ids`. Batched specifications return a tuple of records in source
order. Fitting requires equal observation counts across candidate batches.

An integer `rng` seed restarts a stream on each call; a `numpy.random.Generator`
advances its state. Batched generated sampling uses one stream across sources.

```python
import gmm_divergence as gd

components = [gd.Gaussian.univariate(-2), gd.Gaussian.univariate(2)]
target = gd.GaussianMixture.from_components(components, weights=[0.99, 0.01])
batch = gd.sampling.Stratified(n_samples=20, rng=0).sample(target)
assert batch.samples.shape == (20, 1)
assert batch.integration_weights is not None
assert batch.component_ids is not None
reused = gd.sampling.Samples(batch)
result = gd.kl_divergence(
    target, gd.Gaussian.univariate(), estimator=gd.divergence.MonteCarlo(sampling=reused)
)
assert result.n_samples == 20
```

Pass records to `Samples` and `SampleBatches` to preserve stratified integration
metadata. Raw precomputed observations are interpreted as iid draws. Component
IDs with integration weights describe fixed strata: weights must match source
component mass divided by stratum count, and every positive-mass component must
be represented. Arbitrary normalized integration weights without component IDs
are supported, with uncertainty omitted.

Custom specifications implement `SampleSpec.sample` or
`BatchSampleSpec.sample_batches` and return records, not arrays.

::: gmm_divergence.sampling
    options:
        members: false
        show_root_full_path: true


::: gmm_divergence.sampling.Draw

::: gmm_divergence.sampling.Stratified

::: gmm_divergence.sampling.Samples

::: gmm_divergence.sampling.SampleBatches


::: gmm_divergence.sampling.SampleBatch

::: gmm_divergence.sampling.SampleSpec

::: gmm_divergence.sampling.BatchSampleSpec
