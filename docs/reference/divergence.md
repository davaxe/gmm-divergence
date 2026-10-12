---
icon: lucide/equal-approximately
---

# Divergence API

Top-level divergence helpers are documented under [Top-level API](root.md). This
page documents estimator configuration classes and categorical KL divergence
from `gmm_divergence.divergence`.

Divergence calls accept `p` and `q` by keyword. Estimators are explicit and
keyword-only. Compared Gaussian-family distributions must have the same feature
dimension. Mathematical expressions follow the
[notation conventions](index.md#notation-and-terminology): $p$ is the reference
and $q$ the comparison distribution.

`MomentMatchedGaussian()` always matches aggregate moments, including mixture
weights. For nearest component analysis, use `component_kl_matrix(p, q).min()`;
this is a component diagnostic, not mixture KL.

Results separate `divergence` from `estimator`. `n_evaluations` counts integration
points, including deterministic sigma points; `n_samples` counts Monte
Carlo observations only. Composite results retain directional `terms` and omit
aggregate uncertainty because directional dependence is not inferred.

`monte_carlo.standard_error` is `None` when the integration design cannot
estimate uncertainty, including singleton iid draws and singleton positive
strata. Adaptive `Draw` sampling reports `monte_carlo.target_reached`; `False`
means the sample budget was exhausted. Monte Carlo estimates can be negative
from sampling error and are not clipped.

Symmetric KL averages both directions. Jensen-Shannon returns **divergence**
in nats; its square root is a distance and is not returned by this API.

$$
D_{\mathrm{SKL}}(p,q)=\tfrac12\left[D_{\mathrm{KL}}(p\|q)
+D_{\mathrm{KL}}(q\|p)\right],
$$

$$
D_{\mathrm{JS}}(p,q)=\tfrac12D_{\mathrm{KL}}(p\|m)
+\tfrac12D_{\mathrm{KL}}(q\|m),\qquad m=\tfrac12p+\tfrac12q.
$$

The exact Jensen-Shannon divergence is bounded by $\log 2$ nats. Sampling and
deterministic approximations need not preserve the bounds of exact divergences.

::: gmm_divergence.divergence
    options:
        members: false
        show_root_full_path: true

::: gmm_divergence.divergence.MonteCarlo

::: gmm_divergence.divergence.Unscented

::: gmm_divergence.divergence.MomentMatchedGaussian

::: gmm_divergence.divergence.ClosedForm

::: gmm_divergence.divergence.Variational

## Categorical KL Divergence

`categorical_kl_divergence` compares probability vectors over shared categories.
Corresponding indices must refer to the same category. Mode occupancy is one
use case: independently fitted GMM components must be aligned first, and
components need not represent distinct modes. This function computes categorical
KL, rather than divergence between mixture densities.

For $K$ categories with probabilities $\boldsymbol{\pi}_p$ and
$\boldsymbol{\pi}_q$, the exact quantity is

$$
D_{\mathrm{KL}}(\boldsymbol{\pi}_p\|\boldsymbol{\pi}_q)
=\sum_{k:\pi_{p,k}>0}\pi_{p,k}\log\frac{\pi_{p,k}}{\pi_{q,k}}.
$$

Zero reference probabilities contribute zero. A positive reference probability
with zero comparison probability gives infinity. Smoothing with $\varepsilon$
(`epsilon`) replaces each vector's entries by
$(\pi_k+\varepsilon)/(1+K\varepsilon)$; it changes the distributions compared.

```python
import gmm_divergence as gd

result = gd.divergence.categorical_kl_divergence([0.8, 0.2], [0.5, 0.5])
assert result.value > 0

# Smoothing changes the probability vectors and makes this comparison finite.
smoothed = gd.divergence.categorical_kl_divergence([1.0, 0.0], [0.0, 1.0], epsilon=1e-8)
assert smoothed.value > 0
```

::: gmm_divergence.divergence.categorical_kl_divergence

## Aligned Joint KL

`aligned_joint_kl_divergence` computes exact KL between the joint distributions of
component label and observation. Both mixtures must have the same number of
components and feature dimensions, and corresponding indices must represent
the same component. No component matching is performed.

$$
D_{\mathrm{KL}}(p(k,x) \| q(k,x)) =
D_{\mathrm{KL}}(\boldsymbol{\pi}_p\|\boldsymbol{\pi}_q)
+\sum_{k=1}^{K}\pi_{p,k}D_{\mathrm{KL}}(p_k\|q_k).
$$

Here $k\in\{1,\ldots,K\}$ is the common component label, $x\in\mathbb{R}^d$
an observation, and $p(k,x)=\pi_{p,k}p_k(x)$ with
$p_k(x)=\mathcal{N}(x;\mu_{p,k},\Sigma_{p,k})$, and likewise for $q$.
The vectors $\boldsymbol{\pi}_p$, $\boldsymbol{\pi}_q$ contain component probabilities.
The result separates differences in component weights from differences within
components. This is a joint KL, rather than the marginal KL between mixture
densities; reordering just one mixture changes the assumed correspondence.
A positive reference weight paired with a zero comparison weight gives
infinite joint KL.

Marginalizing over the component label cannot increase KL, so this joint value
is an upper bound on the marginal mixture KL. Combining candidates flattens
their components; its provenance alone does not establish the correspondence
required for an aligned comparison.

```python
import gmm_divergence as gd

components = [gd.Gaussian.univariate(-1), gd.Gaussian.univariate(1)]
p = gd.GaussianMixture.from_components(components, weights=[0.8, 0.2])
q = gd.GaussianMixture.from_components(components, weights=[0.5, 0.5])
result = gd.aligned_joint_kl_divergence(p, q)
assert result.value == result.weight_kl
assert result.weighted_component_kl == 0
```

:::gmm_divergence.divergence.aligned_joint_kl_divergence


::: gmm_divergence.divergence.AlignedJointKLResult

::: gmm_divergence.divergence.MonteCarloStatistics

::: gmm_divergence.divergence.KLEstimator
