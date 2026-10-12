---
icon: lucide/book-marked
---

# API reference

The API reference is organized by import namespace. The top-level
`gmm_divergence` module is a curated convenience layer for common objects and
primary functions. Configuration classes live in the namespace pages.

| Page | Import namespace | Contents |
| --- | --- | --- |
| [Top-level API](root.md) | `gmm_divergence` | Root exports: common classes, helper functions, and results |
| [Distributions](distributions.md) | `gmm_divergence.distributions` | Gaussian-family metadata not exported at the root |
| [Divergence](divergence.md) | `gmm_divergence.divergence` | Estimator configuration |
| [Fitting](fitting.md) | `gmm_divergence.fitting` | Objectives, optimizers, and selectors |
| [Sampling](sampling.md) | `gmm_divergence.sampling` | Sample specifications used by Monte Carlo estimators |
| [Covariance](covariance.md) | `gmm_divergence.covariance` | Covariance regularization and epsilon heuristics |

For day-to-day use, import the package once:

```python
import gmm_divergence as gd
```

Then access configuration objects through namespaces such as
`gd.divergence.MonteCarlo`, `gd.sampling.Draw`, and `gd.fitting.ForwardKL`.

## Notation and terminology

Docstrings and mathematical explanations use the following conventions.
Python argument and attribute names retain their API spelling; for example,
the mathematical dimension $d$ corresponds to `n_features`.

| Symbol | Meaning |
| --- | --- |
| $d$, $x \in \mathbb{R}^d$ | Feature dimension and one observation |
| $X$, $x_n$ | Random observation and its $n$-th observed value |
| $p$, $q$ | Reference and comparison distributions; also their densities when applicable |
| $K_p$, $K_q$, $k$, $j$ | Component counts and component indices; $K$ also counts explicitly defined categorical outcomes |
| $p_k$, $q_j$ | Gaussian component densities |
| $\pi_{p,k}$, $\pi_{q,j}$ | Nonnegative component probabilities, each distribution's weights summing to one |
| $\boldsymbol{\pi}_p$, $\boldsymbol{\pi}_q$ | Component or categorical probability vectors |
| $\mu_p$, $\Sigma_p$ | Aggregate mean and covariance of $p$ |
| $\mu_{p,k}$, $\Sigma_{p,k}$ | Mean vector and covariance matrix of component $k$ of $p$ |
| $L$, $i$, $q_i$ | Number of candidate distributions, candidate index, and candidate density |
| $K_i$, $\pi_{i,k}$, $\mu_{i,k}$, $\Sigma_{i,k}$ | Component count and parameters within candidate $q_i$ |
| $\mathbf{w}=(w_1,\ldots,w_L)^\top$ | Candidate weights; distinct from internal component probabilities |
| $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$ | Combined candidate distribution |
| $\Delta_L=\{\mathbf{w}\in\mathbb{R}^L:w_i\geq0,\ \sum_{i=1}^{L}w_i=1\}$ | Probability simplex with $L$ coordinates and dimension $L-1$ |
| $N$, $n$ | Observation count and observation index |
| $N_p$, $N_q$ | Target observation count and count per candidate batch; these may differ |
| $a_n$, $a_{i,n}$ | Normalized integration weights for target and candidate observations |
| $N_k$, $n_k$, $r_{nk}$ | Soft component count, allocated draw count, and responsibility of component $k$ for observation $n$ |
| $\beta\in[0,1]$ | Forward contribution in bidirectional KL; `forward_weight` |
| $\eta\in[0,1]$ | Covariance shrinkage strength; `strength` |
| $\varepsilon\geq0$, $s>0$, $r$ | Additive covariance loading or categorical smoothing (defined locally), heuristic scale, and retained covariance rank |
| $\lambda_1\leq\cdots\leq\lambda_d$ | Ordered covariance eigenvalues |
| $\lambda_{\mathrm{floor}}>0$, $\kappa_{\max}>1$ | Eigenvalue floor and condition-number target |
| $I_d$, $A^\top$, $\mathrm{tr}(A)$, $\mathrm{diag}(A)$ | Identity matrix, transpose, trace, and diagonal part of a matrix; diag applied to a list constructs a diagonal matrix |

A component is a Gaussian term within a mixture. A candidate is an entire
distribution whose contribution is being fitted; it may contain several
components. Categorical categories may be aligned component labels, but Gaussian
components are not necessarily distinct modes. Mathematical indices start at
one; Python indices start at zero.

All logarithms are natural. KL and Jensen-Shannon **divergences** are in nats;
the square root of Jensen-Shannon divergence is its **distance**, which the API
does not return. Symmetric KL is the average of the two directed KL values.
Moment matching is a squared raw-moment residual, not a divergence in nats.

Exact quantities use $D_{\mathrm{KL}}$ and $D_{\mathrm{JS}}$; numerical approximations use
$\widehat D_{\mathrm{KL}}$ and $\widehat D_{\mathrm{JS}}$. Hats on means or
covariances denote empirical estimates. A hat alone does not imply Monte Carlo
sampling. In fitting, the reference distribution $p$ is called the target.
Feature vectors such as $x$ and $\mu_p$
are column vectors in equations; arrays store observations in rows. The shapes
of the implementation's arrays are documented separately.
