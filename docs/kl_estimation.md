---
icon: lucide/equal-approximately
---

# Kullback-Leibler divergence estimation

## Definition

Following the [notation conventions](reference/index.md#notation-and-terminology),
let $p$ and $q$ be Gaussian mixture densities with $K_p$ and $K_q$ components:

$$
p(x) = \sum_{k=1}^{K_p} \pi_{p,k}\,p_k(x),
\qquad p_k(x)=\mathcal{N}(x;\mu_{p,k},\Sigma_{p,k}),
$$

and

$$
q(x) = \sum_{j=1}^{K_q} \pi_{q,j}\,q_j(x),
\qquad q_j(x)=\mathcal{N}(x;\mu_{q,j},\Sigma_{q,j}),
$$

where $x\in\mathbb{R}^d$, means are in $\mathbb{R}^d$, and covariances are
positive-definite $d\times d$ matrices. Component probabilities satisfy

$$
\pi_{p,k} \geq 0,
\qquad
\sum_{k=1}^{K_p}\pi_{p,k} = 1,
\qquad
\pi_{q,j} \geq 0,
\qquad
\sum_{j=1}^{K_q}\pi_{q,j} = 1.
$$

Then the Kullback-Leibler (KL) divergence from $p$ to $q$ is defined as

$$
D_{\mathrm{KL}}(p \| q) = \int_{\mathbb{R}^d} p(x) \log \frac{p(x)}{q(x)}\,\mathrm{d}x
= \mathbb{E}_{X\sim p}
\left[
\log p(X)-\log q(X)
\right].
$$

Throughout this library, mathematical expressions use $D_{\mathrm{KL}}(p \| q)$;
“KL divergence” is the abbreviation used in prose. All logarithms are natural,
so divergence values are in nats. The order matters: the expectation is under
the first distribution.

For Gaussian mixtures, this can be expressed more explicitly as

$$
D_{\mathrm{KL}}(p\|q)=
\mathbb{E}_{X\sim p}
\left[
\log
\left(
\sum_{k=1}^{K_p}\pi_{p,k}\mathcal{N}(X;\mu_{p,k},\Sigma_{p,k})
\right) -
\log
\left(
\sum_{j=1}^{K_q}\pi_{q,j}\mathcal{N}(X;\mu_{q,j},\Sigma_{q,j})
\right)
\right].
$$

Mixture KL generally has no closed-form expression. The Gaussian-to-Gaussian
case does have an exact formula, available through `ClosedForm` with Gaussian
objects. Mixture comparisons can instead use sampling or deterministic
approximations. There are various approaches to estimating the KL
divergence between Gaussian mixtures, including Monte Carlo sampling,
variational approximations, and deterministic sigma-point methods
[^hershey2007approximating].

## Monte Carlo estimate and uncertainty

For $N$ independent observations $x_n$ drawn from $p$, the fixed-sample estimate is

$$
\widehat D_{\mathrm{KL}}(p\|q)
=\frac{1}{N}\sum_{n=1}^{N}\ell_n,
\qquad \ell_n=\log p(x_n)-\log q(x_n).
$$

For $N\geq2$, its estimated standard error is

$$
\widehat{\mathrm{SE}}
=\sqrt{\frac{1}{N(N-1)}\sum_{n=1}^{N}(\ell_n-\bar\ell)^2},
\qquad \bar\ell=\frac{1}{N}\sum_{n=1}^{N}\ell_n.
$$

With stratified draws, let $n_k$ be the allocated count for component $k$ of $p$
and $x_{k,n}$ its observations. The integration weight per observation is
$\pi_{p,k}/n_k$, giving

$$
\widehat D_{\mathrm{KL}}(p\|q)
=\sum_{k:\pi_{p,k}>0}\frac{\pi_{p,k}}{n_k}
\sum_{n=1}^{n_k}\ell_{k,n},
\qquad \ell_{k,n}=\log p(x_{k,n})-\log q(x_{k,n}).
$$

The log densities here are full mixture densities. When every positive-weight
component has at least two observations, the estimated standard error is

$$
\widehat{\mathrm{SE}}_{\mathrm{strat}}
=\sqrt{\sum_{k:\pi_{p,k}>0}\frac{\pi_{p,k}^2s_k^2}{n_k}},
$$

where $s_k^2$ is the unbiased sample variance of the log-density ratios in
component $k$. These errors describe numerical integration with fixed
distributions, not uncertainty in fitted distribution parameters. Arbitrary
integration weights without a supported sampling design have no standard error.
An estimate can be negative even though exact KL is nonnegative; sampled values
are not clipped.

## Example

Use the [`kl_divergence`](reference/root.md#gmm_divergence.kl_divergence) function
to estimate KL between two Gaussian mixtures:

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
    p, q, estimator=gd.divergence.MonteCarlo(sampling=gd.sampling.Draw(n_samples=10_000, rng=9126))
)
assert abs(kl_estimate.value - 0.32286) < 1e-5
```

1.  This defines $p$ as a one-feature ($d=1$) Gaussian mixture with two equally weighted components. In $\mathcal{N}(x;\mu,\Sigma)$, the last argument is variance when $d=1$:
 
    $$
    p(x) = 0.5 \mathcal{N}(x;0.0,1.0) + 0.5 \mathcal{N}(x;1.0,1.0).
    $$

    !!! note "Simple constructors"
        For simple one-dimensional examples, `GaussianMixture.from_components`
        together with `Gaussian.univariate` is usually the clearest way to define
        Gaussian mixtures.
    
        The `GaussianMixture` constructor remains useful when you
        already have weight, mean, and covariance arrays.

    
!!! note "Other methods"
    The `kl_divergence` function also supports other estimation methods. See
    the [divergence API reference](reference/divergence.md) for details.

## Sampling configuration

Monte Carlo sampling is configured with explicit sample specifications:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components([
    gd.Gaussian.univariate(mean=0.0, variance=1.0),
    gd.Gaussian.univariate(mean=1.0, variance=1.0),
])

drawn = gd.divergence.MonteCarlo(sampling=gd.sampling.Draw(n_samples=10_000, rng=9126))
reused = gd.divergence.MonteCarlo(sampling=gd.sampling.Samples(p.sample(10_000, rng=9126)))
stratified = gd.divergence.MonteCarlo(sampling=gd.sampling.Stratified(n_samples=10_000, rng=9126))
```

`gd.sampling.Draw` is the default and supports Gaussian and GaussianMixture sources.
`gd.sampling.Samples` is useful when comparing several methods on exactly the same
reference samples. `gd.sampling.Stratified` allocates fixed sample counts to
positive-weight components instead of relying on random component counts; a
single Gaussian is treated as a one-component mixture.

Stratified standard errors require at least two observations from every
positive-weight component. When a component has only one observation, the KL
estimate is still returned, while `result.monte_carlo.standard_error` is
`None`. Precomputed `SampleBatch` records retain the integration design and
produce the same estimate and uncertainty when reused with `Samples`.

[^hershey2007approximating]:
    Hershey, John R., and Peder A. Olsen. "Approximating the Kullback Leibler divergence between Gaussian mixture models." 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4. IEEE, 2007.
