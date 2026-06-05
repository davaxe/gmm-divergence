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
[`kl_divergence`](../reference/kl_estimators.md#gmm_divergence.kl_divergence) function. For example:

``` python hl_lines="3-5"
from gmm_divergence import GaussianMixture, MonteCarlo, kl_divergence

p = GaussianMixture.from_arrays(  # (1)!
    weights=[0.5, 0.5], means=[[0.0], [1.0]], covariances=[[[1.0]], [[1.0]]]
)

q = GaussianMixture.from_arrays(
    weights=[0.5, 0.5], means=[[0.5], [2.5]], covariances=[[[1.0]], [[0.5]]]
)


kl_estimate = kl_divergence(p, q, method=MonteCarlo(rng=9126))
assert abs(kl_estimate.value - 0.32286) < 1e-5
```

1.  This defines $p$ as a one dimensional ($d=1$) Gaussian mixture with two equally weighted components. The density of $p$ is given by
 
    $$
    p(x) = 0.5 \mathcal{N}(x;0.0,1.0) + 0.5 \mathcal{N}(x;1.0,1.0).
    $$

    !!! note "GaussianMixture.from_arrays"
        The `GaussianMixture.from_arrays` constructor is a more flexible way to construct Gaussian mixtures from arrays of weights, means, and covariances.
    
!!! note "Other methods"
    The `kl_divergence` function also supports other estimation methods. See the [`kl_divergence`](../reference/kl_estimators.md#gmm_divergence.kl_divergence) for details.

!!! note "Single gaussian special case"
    If $p$ and $q$ are single Gaussians (i.e. $K_p = K_q = 1$), then the KL divergence has a closed-form expression and can be computed exactly.

    When using the `kl_divergence` function, if both $p$ and $q$ are single Gaussians, then the KL divergence can be computed exactly using the closed-form expression, which is less computationally expensive and more accurate than estimation methods. This is achieved by setting the `method` argument to `"closed_form"`, i.e.
    
    ``` python
    kl_exact = kl_divergence(p, q, method="closed_form")
    ```

[^hershey2007approximating]:
    Hershey, John R., and Peder A. Olsen. "Approximating the Kullback Leibler divergence between Gaussian mixture models." 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol. 4. IEEE, 2007.
