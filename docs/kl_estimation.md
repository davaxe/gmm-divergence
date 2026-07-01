---
icon: lucide/equal-approximately
---

# Kullback-Leibler divergence estimation

## Definition

Let $p$ and $q$ be two Gaussian mixture with $K_p$ and $K_q$ components respectively, defined as

$$
p(x) = \sum_{i=1}^{K_p} \pi_i \mathcal{N}(x;\mu_i,\Sigma_i),
$$

and

$$
q(x) = \sum_{j=1}^{K_q} \omega_j \mathcal{N}(x;\nu_j,\Lambda_j),
$$

where $x \in \mathbb{R}^d$ and the component weights $\pi_i$ and $\omega_j$ are non-negative and sum to one:

$$
\pi_i \geq 0,
\qquad
\sum_{i=1}^{K_p}\pi_i = 1,
\qquad
\omega_j \geq 0,
\qquad
\sum_{j=1}^{K_q}\omega_j = 1.
$$

Then the Kullback-Leibler (KL) divergence from $p$ to $q$ is defined as

$$
D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{X\sim p}
\left[
\log p(X)-\log q(X)
\right].
$$

For Gaussian mixtures, this can be expressed more explicitly as

$$
D_{\mathrm{KL}}(p\|q)=
\mathbb{E}_{X\sim p}
\left[
\log
\left(
\sum_{i=1}^{K_p}\pi_i \mathcal{N}(X;\mu_i,\Sigma_i)
\right) -
\log
\left(
\sum_{j=1}^{K_q}\omega_j \mathcal{N}(X;\nu_j,\Lambda_j)
\right)
\right].
$$

This KL divergence does not have a closed-form expression, and must be estimated
using numerical methods. There are various approaches to estimating the KL
divergence between Gaussian mixtures, including Monte Carlo sampling,
variational approximations, and unscented sigma point methods
[^hershey2007approximating].

## Example

To estimate the KL divergence between two Gaussian mixtures using the
[`kl_divergence`](../reference/kl_based_estimators.md#gmm_divergence.kl_divergence) function. For example:

``` python hl_lines="3-9"
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(  # (1)!
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

kl_estimate = gd.kl_divergence(
    p, q, method=gd.MonteCarlo(sampling=gd.DrawSamples(10_000, rng=9126))
)
assert abs(kl_estimate.value - 0.32286) < 1e-5
```

1.  This defines $p$ as a one dimensional ($d=1$) Gaussian mixture with two equally weighted components. The density of $p$ is given by
 
    $$
    p(x) = 0.5 \mathcal{N}(x;0.0,1.0) + 0.5 \mathcal{N}(x;1.0,1.0).
    $$

    !!! note "Simple constructors"
        For simple one-dimensional examples, `GaussianMixture.from_components`
        together with `Gaussian.univariate` is usually the clearest way to define
        Gaussian mixtures.
    
        The `GaussianMixture.from_arrays` constructor remains useful when you
        already have weight, mean, and covariance arrays.

    
!!! note "Other methods"
    The `kl_divergence` function also supports other estimation methods. See the [`kl_divergence`](../reference/kl_based_estimators.md#gmm_divergence.kl_divergence) for details.

## Sampling configuration

Monte Carlo sampling is configured with explicit sample specifications:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components([
    gd.Gaussian.univariate(mean=0.0, variance=1.0),
    gd.Gaussian.univariate(mean=1.0, variance=1.0),
])

drawn = gd.MonteCarlo(sampling=gd.DrawSamples(10_000, rng=9126))
reused = gd.MonteCarlo(sampling=gd.UseSamples(p.sample(10_000, rng=9126)))
stratified = gd.MonteCarlo(sampling=gd.StratifiedSamples(10_000, rng=9126))
```

`DrawSamples` is the default and works for any sampleable distribution.
`UseSamples` is useful when comparing several methods on exactly the same
reference samples. `StratifiedSamples` is only valid when the reference
distribution is a `GaussianMixture`; it allocates fixed sample counts to
positive-weight components instead of relying on random component counts.

!!! tip "Adaptive Monte Carlo"
    To spend more samples only when the estimate is still noisy, pass a target
    standard error.

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components([
    gd.Gaussian.univariate(mean=0.0, variance=1.0),
    gd.Gaussian.univariate(mean=1.0, variance=1.0),
])
q = gd.GaussianMixture.from_components([
    gd.Gaussian.univariate(mean=0.5, variance=1.0),
    gd.Gaussian.univariate(mean=2.5, variance=0.5),
])

result = gd.kl_divergence(
    p,
    q,
    method=gd.MonteCarlo(
        sampling=gd.DrawSamples(10_000, rng=9126),
        target_standard_error=1e-3,
        max_samples=100_000,
    ),
)
```

The initial `DrawSamples` count is always evaluated first. Additional batches
are drawn until the target is met or `max_samples` is reached.

[^hershey2007approximating]:
    Hershey, John R., and Peder A. Olsen. "Approximating the Kullback Leibler divergence between Gaussian mixture models." 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4. IEEE, 2007.
