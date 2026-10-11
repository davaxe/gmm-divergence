---
icon: lucide/equal-approximately
---

# Divergence API

Top-level divergence helpers are documented under [Top-level API](root.md). This
page documents estimator configuration classes and categorical KL divergence
from `gmm_divergence.divergence`.

Divergence calls accept `p` and `q` by keyword. Estimators are explicit and
keyword-only. Unsupported configurations raise `TypeError` immediately;
invalid shapes or values raise `ValueError`, and invalid component positions
raise `IndexError`.

`MomentMatchedGaussian()` always matches aggregate moments, including mixture
weights. For nearest component analysis, use `component_kl_matrix(p, q).min()`;
this is a component diagnostic, not mixture KL.

Results separate `divergence` from `estimator`. `n_evaluations` counts integration
observations, including deterministic sigma points; `n_samples` counts Monte
Carlo observations only. Composite results retain directional `terms` and omit
aggregate uncertainty because directional dependence is not inferred.

`monte_carlo.standard_error` is `None` when the integration design cannot
estimate uncertainty, including singleton iid draws and singleton positive
strata. Adaptive `Draw` sampling reports `monte_carlo.target_reached`; `False`
means the sample budget was exhausted. Monte Carlo estimates can be negative
from sampling error and are not clipped.

Symmetric KL averages both directions. Jensen-Shannon returns **divergence**
in nats; its square root is a distance and is not returned by this API.

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

```python
import gmm_divergence as gd

result = gd.divergence.categorical_kl_divergence([0.8, 0.2], [0.5, 0.5])
assert result.value > 0

# Smoothing changes the probability vectors and makes this comparison finite.
smoothed = gd.divergence.categorical_kl_divergence([1.0, 0.0], [0.0, 1.0], epsilon=1e-8)
assert smoothed.value > 0
```

::: gmm_divergence.divergence.categorical_kl_divergence

## Aligned Component KL

`aligned_joint_kl_divergence` computes exact KL between the joint distributions of
component label and observation. Both mixtures must have the same number of
components and feature dimensions, and corresponding indices must represent
the same component. No component matching is performed.

$$
D_{\mathrm{KL}}(p(k,x) \| q(k,x)) =
D_{\mathrm{KL}}(\pi_p \| \pi_q) +
\sum_k \pi_{p,k} D_{\mathrm{KL}}(p_k \| q_k).
$$

The result separates differences in component weights from differences within
components. This is a joint KL, rather than the marginal KL between mixture
densities; reordering just one mixture changes the assumed correspondence.
A positive reference weight paired with a zero comparison weight gives
infinite joint KL.

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
