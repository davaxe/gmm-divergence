---
icon: lucide/chart-no-axes-combined
---

# Fitting mixture weights

Following the [notation conventions](reference/index.md#notation-and-terminology),
let $p$ be the fixed target density and $q_1,\ldots,q_L$ the fixed candidate
densities. Each candidate may be a Gaussian or a Gaussian mixture. Fitting
chooses nonnegative candidate weights $\mathbf{w}=(w_1,\ldots,w_L)^\top$ summing
to one, so that $q_{\mathbf{w}}=\sum_{i=1}^{L}w_iq_i$ approximates $p$ under the
chosen objective. Candidate weights $w_i$ are distinct from the component
probabilities inside each candidate.

## Forward KL divergence formulation

For Gaussian mixtures in $d$ features, write

$$
p(x)=\sum_{k=1}^{K_p}\pi_{p,k}\mathcal{N}(x;\mu_{p,k},\Sigma_{p,k}),
$$

and, for candidate $i\in\{1,\ldots,L\}$,

$$
q_i(x)=\sum_{k=1}^{K_i}\pi_{i,k}\mathcal{N}(x;\mu_{i,k},\Sigma_{i,k}).
$$

Here $K_p$ and $K_i$ are component counts, the means are $d$-dimensional vectors,
and covariances are positive-definite $d\times d$ matrices. Each distribution's
component probabilities are nonnegative and sum to one. All these parameters
are fixed; only $\mathbf{w}$ is optimized. A Gaussian is the case with one
component of probability one.

The combined distribution is

$$
q_{\mathbf{w}}(x)
=\sum_{i=1}^{L}w_iq_i(x)
=\sum_{i=1}^{L}\sum_{k=1}^{K_i}
 w_i\pi_{i,k}\mathcal{N}(x;\mu_{i,k},\Sigma_{i,k}).
$$

The represented mixture contains $\sum_{i=1}^{L}K_i$ components, including any
zero-weight components retained by the implementation. The forward objective is

\begin{equation}
\label{eq:mixture-weight-optimization}
\mathbf{w}^{\star}
=\arg\min_{\mathbf{w}\in\Delta_L}D_{\mathrm{KL}}(p\|q_{\mathbf{w}}),
\end{equation}

where the $L$-coordinate probability simplex is

$$
\Delta_L=\left\{\mathbf{w}\in\mathbb{R}^{L}:
 w_i\geq0,\ \sum_{i=1}^{L}w_i=1\right\}.
$$

### Practical objective

The exact forward objective separates into a fixed target term and cross-entropy:

$$
F(\mathbf{w})=D_{\mathrm{KL}}(p\|q_{\mathbf{w}})
=C_p+H(p,q_{\mathbf{w}}),
$$

$$
C_p=\mathbb{E}_{X\sim p}[\log p(X)],
\qquad H(p,q_{\mathbf{w}})=-\mathbb{E}_{X\sim p}[\log q_{\mathbf{w}}(X)].
$$

$C_p$ does not depend on $\mathbf{w}$, so minimizing cross-entropy gives the same
weights as minimizing forward KL. The implementation nevertheless includes the
cached target term in `objective_value`, so the reported quantity estimates KL,
not cross-entropy.

With fixed target observations $x_n$ and normalized integration weights
$a_n\geq0$, $\sum_{n=1}^{N_p}a_n=1$, preparation builds

$$
\widehat F(\mathbf{w})
=\sum_{n=1}^{N_p}a_n
 \left[\log p(x_n)-\log q_{\mathbf{w}}(x_n)\right].
$$

For independent observations $x_n\sim p$, use $a_n=1/N_p$. Stratified batches
instead use the component-mass integration weights described under
[KL estimation](kl_estimation.md#monte-carlo-estimate-and-uncertainty). The
resulting numerical optimization problem is

\begin{equation}
\label{eq:mixture-weight-optimization-kl-simplified-monte-carlo}
\widehat{\mathbf{w}}^{\star}
=\arg\min_{\mathbf{w}\in\Delta_L}\widehat F(\mathbf{w}).
\end{equation}

The hat distinguishes the sampled problem from the exact objective in
$\eqref{eq:mixture-weight-optimization}$. Optimizer convergence concerns this
fixed sampled problem, not the unknown exact optimum. Samples and densities
are cached throughout a fit; the numerical kernels floor weights at $10^{-300}$
when taking logarithms or dividing in forward gradients.

## Alternative objective: reverse KL divergence

The exact reverse objective is

$$
R(\mathbf{w})=D_{\mathrm{KL}}(q_{\mathbf{w}}\|p)
=\mathbb{E}_{X\sim q_{\mathbf{w}}}
 \left[\log q_{\mathbf{w}}(X)-\log p(X)\right].
$$

There is no need to redraw from the changing mixture during optimization.
Linearity of integration gives

\begin{equation}
\label{eq:mixture-weight-optimization-reverse-kl}
R(\mathbf{w})=\sum_{i=1}^{L}w_i\,
\mathbb{E}_{X\sim q_i}
 \left[\log q_{\mathbf{w}}(X)-\log p(X)\right].
\end{equation}

Draw fixed observations $x_{i,n}$ from each candidate $q_i$, with $N_q$
observations per candidate and normalized within-batch integration weights
$a_{i,n}$. The implementation minimizes

$$
\widehat R(\mathbf{w})
=\sum_{i=1}^{L}w_i\sum_{n=1}^{N_q}a_{i,n}
 \left[\log q_{\mathbf{w}}(x_{i,n})-\log p(x_{i,n})\right].
$$

For independent draws, $a_{i,n}=1/N_q$. For stratified batches, the weights
preserve each candidate's component mass. Both log-density terms are inside the
sample average. The target density is evaluated at candidate observations;
reverse fitting neither requires nor consumes target observations. All candidate
batches must have equal counts; $N_q$ need not equal the target count $N_p$ used
by objectives that require both sources.

## Alternative objective: bidirectional KL divergence

Let $\beta\in[0,1]$ be `forward_weight`. The objective is

$$
B_{\beta}(\mathbf{w})
=\beta D_{\mathrm{KL}}(p\|q_{\mathbf{w}})
 +(1-\beta)D_{\mathrm{KL}}(q_{\mathbf{w}}\|p).
$$

Its sampled implementation is $\widehat B_{\beta}
=\beta\widehat F+(1-\beta)\widehat R$, using separately configured target and
candidate batches. At either endpoint, the unused source is not sampled.
$\beta=1/2$ is the averaged symmetric KL, also returned by
`symmetric_kl_divergence` when comparing two fixed distributions. The forward
term retains its cached target-density constant.

## Alternative objective: Jensen-Shannon divergence

Define the midpoint density

$$
m_{\mathbf{w}}(x)=\tfrac12p(x)+\tfrac12q_{\mathbf{w}}(x).
$$

The exact objective is

$$
D_{\mathrm{JS}}(p,q_{\mathbf{w}})
=\tfrac12D_{\mathrm{KL}}(p\|m_{\mathbf{w}})
 +\tfrac12D_{\mathrm{KL}}(q_{\mathbf{w}}\|m_{\mathbf{w}}).
$$

Using the same source notation as above, its fixed-sample estimate is

$$
\widehat D_{\mathrm{JS}}(p,q_{\mathbf{w}})
=\frac12\sum_{n=1}^{N_p}a_n
 \log\frac{p(x_n)}{m_{\mathbf{w}}(x_n)}
+\frac12\sum_{i=1}^{L}w_i\sum_{n=1}^{N_q}a_{i,n}
 \log\frac{q_{\mathbf{w}}(x_{i,n})}{m_{\mathbf{w}}(x_{i,n})}.
$$

The exact divergence lies in $[0,\log 2]$ nats. A finite-sample estimate need not
preserve this bound. The objective and divergence API do not take its square root.

This gives a symmetric alternative to the directional KL objectives:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    [
        gd.Gaussian.univariate(mean=0.0, variance=0.5),
        gd.Gaussian.univariate(mean=2.0, variance=0.5),
    ],
    weights=[0.6, 0.4],
)
q1 = gd.Gaussian.univariate(mean=0.0, variance=0.5)
q2 = gd.Gaussian.univariate(mean=2.0, variance=0.5)

fit = gd.fit_gaussian_weights(
    p,
    [q1, q2],
    objective=gd.fitting.JensenShannon(
        target_sampling=gd.sampling.Draw(n_samples=10_000, rng=102),
        candidate_sampling=gd.sampling.Draw(n_samples=10_000, rng=102),
    ),
    optimizer=gd.fitting.SoftmaxLBFGSB(),
)
```

For objectives that expose both options, `target_sampling` controls samples from the
reference distribution and `candidate_sampling` controls one fixed batch per candidate
distribution. Use `gd.sampling.Samples(...)` for precomputed reference samples and
`gd.sampling.SampleBatches(...)` for precomputed candidate batches.
`gd.sampling.Stratified(...)` can be used for either side; a single Gaussian is
treated as a one-component mixture.

`ReverseKL` exposes only `candidate_sampling`: its expectation is under the fitted
candidate mixture, and `p` is evaluated at the candidate samples. It therefore
does not require or consume samples drawn from `p`.


## Categorical forward KL

Categorical fitting uses the same $L$ candidate weights and simplex. For $K$
shared categories, let $\boldsymbol{\pi}_p$ be the target probability vector and
$\pi_{i,k}$ candidate $i$'s probability of category $k$. The fitted probabilities
and exact objective are

$$
\pi_{\mathbf{w},k}=\sum_{i=1}^{L}w_i\pi_{i,k},
\qquad F_{\mathrm{cat}}(\mathbf{w})
=\sum_{k:\pi_{p,k}>0}\pi_{p,k}
 \log\frac{\pi_{p,k}}{\pi_{\mathbf{w},k}}.
$$

No observations are drawn; `fit_categorical_weights` sums directly over
categories. Candidate rows and the target must already be normalized. A
positive target category must have support in at least one candidate. Category
indices must represent the same outcomes in every row.

## Moment matching

Let $\mu_i$ and $\Sigma_i$ denote the aggregate mean and covariance of candidate
$q_i$, and $\mu_p$, $\Sigma_p$ those of the target. Define the raw second moment
$M_{2,i}=\Sigma_i+\mu_i\mu_i^\top$ and similarly $M_{2,p}$. By linearity,

$$
\mu_{q_{\mathbf{w}}}=\sum_{i=1}^{L}w_i\mu_i,
\qquad M_{2,q_{\mathbf{w}}}=\sum_{i=1}^{L}w_iM_{2,i}.
$$

The default `MomentMatching` objective is
$\|\mu_{q_{\mathbf{w}}}-\mu_p\|_2^2$. With `fit_second_moments=True`, it is

$$
J_{\mathrm{mom}}(\mathbf{w})
=\|\mu_{q_{\mathbf{w}}}-\mu_p\|_2^2
 +\|M_{2,q_{\mathbf{w}}}-M_{2,p}\|_{\mathrm{F}}^2,
$$

where $\|\cdot\|_{\mathrm{F}}$ is the Frobenius norm, matching the implementation's
sum of squared flattened entries. Covariance itself is not linear under mixing;
raw second moments are. This exact moment objective uses no sampling, depends
on feature scale, and is not a divergence in nats.

## Example

The [`fit_gaussian_weights`](reference/root.md#gmm_divergence.fit_gaussian_weights) function fits the weights of a mixture of candidate
distributions $q_i$ to a fixed reference mixture $p$. For example:

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    components=[
        gd.Gaussian.univariate(mean=0.0, variance=0.5),
        gd.Gaussian.univariate(mean=2.0, variance=0.5),
    ],
    weights=[0.6, 0.4],
)

q1 = gd.Gaussian.univariate(mean=0.0, variance=0.5)
q2 = gd.Gaussian.univariate(mean=2.0, variance=0.5)
result = gd.fit_gaussian_weights(
    p,
    [q1, q2],
    optimizer=gd.fitting.SimplexSLSQP(),
    objective=gd.fitting.ForwardKL(target_sampling=gd.sampling.Draw(n_samples=10_000, rng=102)),
)

assert result.converged
assert abs(result.weights[0] - 0.6) < 1e-2
assert abs(result.weights[1] - 0.4) < 1e-2
```

Here, the optimizer recovers the mixture weights of the reference distribution by
combining the two candidate distributions `q1` and `q2`. The result contains the final
scalar objective value, the directly usable `result.mixture`, and optimizer
termination metadata. `result.mixture.mapping` refers to original candidate indices.

## Reusing a prepared fit

Use [`prepare_gaussian_weight_fit`](reference/fitting.md#gmm_divergence.fitting.prepare_gaussian_weight_fit)
when fitting the same objective more than once. Preparation performs candidate
selection, sampling, and density or moment calculations once. Each call to
`prepared.fit(optimizer=...)` runs the optimizer and returns a complete result,
mapping weights back to the original candidate sequence.

```python
import gmm_divergence as gd

p = gd.GaussianMixture.from_components(
    [gd.Gaussian.univariate(-1.0), gd.Gaussian.univariate(1.0)], weights=[0.3, 0.7]
)
candidates = [gd.Gaussian.univariate(-1.0), gd.Gaussian.univariate(1.0)]
prepared = gd.fitting.prepare_gaussian_weight_fit(
    p,
    candidates,
    objective=gd.fitting.ForwardKL(target_sampling=gd.sampling.Draw(n_samples=10_000, rng=102)),
)

result = prepared.fit(optimizer=gd.fitting.SimplexSLSQP())

# Reuse the cached samples and density matrices with a warm start.
warm_result = prepared.fit(
    optimizer=gd.fitting.SimplexSLSQP(
        initial_weights=result.weights[list(prepared.active_candidate_indices)]
    )
)

# Inspect the cached objective and its gradient without invoking an optimizer.
value, gradient = prepared.evaluate([0.5, 0.5])
assert result.converged
assert warm_result.converged
assert gradient.shape == (2,)
```

!!! info "Alternative objectives for `fit_gaussian_weights`"
    The `fit_gaussian_weights` function also supports reverse KL, bidirectional
    KL, Jensen-Shannon, and moment-matching objectives through the `objective`
    parameter. See the [API
    reference](reference/root.md#gmm_divergence.fit_gaussian_weights) for
    details.
