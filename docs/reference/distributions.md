---
icon: lucide/boxes
---

# Distributions

Common Gaussian-family classes are documented under [Top-level API](root.md). This
page documents Gaussian-family metadata objects that are available from
`gmm_divergence.distributions` but are not exported at the root.

Both constructors accept array-like parameters directly and copy them into
read-only `float64` arrays. `regularizer` is keyword-only. Mixture `weights`
are nonnegative coefficients normalized to sum to one; categorical
`probabilities` must already sum to one and are never normalized or clipped.

`Gaussian.mean` and `GaussianMixture.mean` have shape `(n_features,)`;
their `covariance` properties have shape `(n_features, n_features)`.
Mixture component arrays remain `means` `(n_components, n_features)` and
`covariances` `(n_components, n_features, n_features)`.

`Gaussian.from_samples` requires finite two-dimensional observations with at
least two rows and uses unbiased covariance. Use `regularizer` for singular
empirical covariances. Fit mixtures with explicit scikit-learn configuration,
then call `GaussianMixture.from_sklearn`. `to_sklearn` copies parameters into an
evaluable sklearn model; it does not fit observations.

`mixture.as_gaussian()` requires exactly one component. Use
`mixture.moment_match()` for an approximation matching aggregate moments.

For a mixture $p(x)=\sum_{k=1}^{K_p}\pi_{p,k}
\mathcal{N}(x;\mu_{p,k},\Sigma_{p,k})$, the aggregate moments are

$$
\mu_p=\sum_{k=1}^{K_p}\pi_{p,k}\mu_{p,k},
\qquad
\Sigma_p=\sum_{k=1}^{K_p}\pi_{p,k}
\left[\Sigma_{p,k}+(\mu_{p,k}-\mu_p)(\mu_{p,k}-\mu_p)^\top\right].
$$

Here $K_p$ counts components, $\pi_{p,k}$ are normalized component probabilities,
and each mean and covariance describes one component in $d$ features. The
between-component term is essential; covariance is not simply an average of
component covariances. See the [notation guide](index.md#notation-and-terminology).

For $N\geq2$ observations $x_n$, `Gaussian.from_samples` uses
$\widehat\mu=N^{-1}\sum_{n=1}^{N}x_n$ and

$$
\widehat\Sigma=\frac{1}{N-1}\sum_{n=1}^{N}
(x_n-\widehat\mu)(x_n-\widehat\mu)^\top.
$$

This unbiased sample covariance differs from the responsibility-weighted
maximum-likelihood covariance in
[`component_statistics`](fitting.md#component-statistics), which divides by
the soft count without an unbiasedness correction.

`combine_distributions` always returns a directly usable `GaussianMixture`.
Its `mapping` identifies the immediate source and local component of each
flattened component. Nested combination maps to its immediate inputs.
`select_components` and `prune_mixture` preserve retained mapping rows;
component indices must be integers and in bounds.

```python
import gmm_divergence as gd

target = gd.Gaussian.from_samples([[0], [1], [2]])
mixture = gd.combine_distributions([target, gd.Gaussian.univariate(3)], weights=[3, 1])
assert mixture.mapping is not None
assert mixture.mapping.source_of(1) == (1, 0)
assert mixture.mean.shape == (1,)
approximation = mixture.moment_match()
result = gd.kl_divergence(mixture, approximation, estimator=gd.divergence.Unscented())
```

::: gmm_divergence.distributions
    options:
        members: false
        show_root_full_path: true


::: gmm_divergence.distributions.GaussianDistribution

::: gmm_divergence.distributions.MixtureDiagnostics

::: gmm_divergence.distributions.MixtureMapping
