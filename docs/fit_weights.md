---
icon: lucide/chart-no-axes-combined
---

# Fitting mixture weights

This section describes how to fit mixture weights for a convex combination of
fixed candidate Gaussian mixtures. Given a reference mixture $p$ and candidate
mixtures $q_1,\dots,q_N$, the goal is to choose non-negative weights $w_i$ that
sum to one so that the combined mixture $q_{\mathbf{w}}=\sum_i w_i q_i$
approximates $p$ as closely as possible. The primary formulation minimizes the
forward KL divergence $D_{\mathrm{KL}}(p\|q_{\mathbf{w}})$, with alternative
reverse and bidirectional KL objectives discussed afterward.

## Forward KL divergence formulation

Similar to the previous sections, let $p$ be a fixed Gaussian mixture and let
$\{q_i\}_{i=1}^N$ denote a collection of fixed Gaussian mixtures. Specifically,

$$
p(x) = \sum_{k=1}^{K_p} \pi_k \mathcal{N}(x;\mu_k,\Sigma_k),
$$

and, for each $i \in \{1,\dots,N\}$,

$$
q_i(x) = \sum_{j=1}^{K_i} \omega_{ij}\mathcal{N}(x;\nu_{ij},\Lambda_{ij}).
$$

Here, $K_p$ is the number of components in $p$, while $K_i$ is the number
of components in the mixture $q_i$, which may vary with $i$. All mixtures are
fixed, meaning that the component weights, means, and covariances,

$$
\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K_p}
\quad\text{and}\quad
\{\omega_{ij},\nu_{ij},\Lambda_{ij}\}_{j=1}^{K_i}, \; i=1,\dots,N,
$$

are treated as fixed quantities. The objective is to find a set of non-negative weights $\{w_i\}_{i=1}^N$ that sum to one, such that the KL divergence from $p$ to the mixture

$$
q_{\mathbf{w}}(x) = \sum_{i=1}^N w_i q_i(x)
$$

is minimized. Formally, this can be expressed as the following optimization problem:

\begin{equation}
\label{eq:mixture-weight-optimization}
\begin{aligned}
    \min_{\mathbf{w} \in \Delta_N}
    \quad & D_{\mathrm{KL}}\!\left(p \,\|\, q_{\mathbf{w}}\right),
\end{aligned}
\end{equation}

where

$$
\Delta_N =
\left\{
\mathbf{w} \in \mathbb{R}^N
\,:\,
w_i \ge 0,\;
\sum_{i=1}^N w_i = 1
\right\}.
$$


Importantly, the resulting mixture $q_{\mathbf{w}}$ is itself a Gaussian mixture, with the number of components equal to the sum of the number of components across all $q_i$:

$$
q_{\mathbf{w}}(x) = \sum_{i=1}^N w_i \sum_{j=1}^{K_i} \omega_{ij}\mathcal{N}(x;\nu_{ij},\Lambda_{ij}) = \sum_{i=1}^N \sum_{j=1}^{K_i} w_i \omega_{ij}\mathcal{N}(x;\nu_{ij},\Lambda_{ij}).
$$

This can further compactly be expressed as
$$
q_{\mathbf{w}}(x) = \sum_{i=1}^N \sum_{j=1}^{K_i} \tilde{\omega}_{ij} \mathcal{N}(x;\nu_{ij},\Lambda_{ij}),
\quad \text{

### Practical objective

Using the defenition of KL (see [Kl estimation](kl_estimation.md#definition)) divergence, the optimization problem in $\eqref{eq:mixture-weight-optimization}$ can be rewritten as

$$
\min_{\mathbf{w} \in \Delta_N}
\quad \mathbb{E}_{X\sim p}
\left\lbrack \log p(X) - \log q_{\mathbf{w}}(X) \right\rbrack.
$$

Since $p$ does not depend on $\mathbf{w}$, the optimization problem is equivalent to

$$
\min_{\mathbf{w} \in \Delta_N}
\quad \underbrace{-\mathbb{E}_{X\sim p}
\left\lbrack \log q_{\mathbf{w}}(X) \right\rbrack}_{\equiv J(\mathbf{w}), \;\text{objective function}}.
$$

The _objective function_ $J(\mathbf{w})$ is the negative expected log-likelihood of the mixture $q_{\mathbf{w}}$ under the distribution $p$ and can generally not be expressed in closed form. However, it
can be estimated using Monte Carlo sampling. Specifically, given $M$ independent and identically distributed (iid) samples $x^{(1)},\dots,x^{(M)}$ drawn from $p$, we can construct the following estimator for $J(\mathbf{w})$:

$$
\hat{J}(\mathbf{w}) = - \frac{1}{M} \sum_{m=1}^M \log q_{\mathbf{w}}(x^{(m)}),
\quad \text{where} \quad
x^{(1)},\dots,x^{(M)} \overset{\mathrm{iid}}{\sim} p,
$$

resulting in the final practical optimization problem

\begin{equation}
\label{eq:mixture-weight-optimization-kl-simplified-monte-carlo}
\mathbf{w}^{\star} = \arg\min_{\mathbf{w} \in \Delta_N} \hat{J}(\mathbf{w}).
\end{equation}

## Alternative objective: _reverse_ KL divergence

Alternatively, one could consider the reverse KL divergence from $q_{\mathbf{w}}$ to $p$:

$$
D_{\mathrm{KL}}(q_{\mathbf{w}} \| p) = \int q_{\mathbf{w}}(x) \log \frac{q_{\mathbf{w}}(x)}{p(x)} dx = \mathbb{E}_{X\sim q_{\mathbf{w}}}
\left\lbrack \log q_{\mathbf{w}}(X) - \log p(X) \right\rbrack.
$$

In this case, the optimization problem would be

\begin{equation}
\label{eq:mixture-weight-optimization-reverse-kl}
\begin{aligned}
    \min_{\mathbf{w} \in \Delta_N}
    \quad & D_{\mathrm{KL}}\!\left(q_{\mathbf{w}} \
|\, p\right) = \mathbb{E}_{X\sim q_{\mathbf{w}}}
\left\lbrack \log q_{\mathbf{w}}(X) - \log p(X) \right\rbrack.
\end{aligned}
\end{equation}

This optimization problem can also be estimated using Monte Carlo sampling, but
it requires sampling from the mixture $q_{\mathbf{w}}$, which itself depends on
the optimization variable $\mathbf{w}$. This can make the optimization more
challenging, as the sampling distribution change as $\mathbf{w}$ is updated.

However, the underlying components are fixed and it is possible to reuse samples from the candidate mixtures $q_i$ to construct an estimator for the reverse KL divergence. For example, given $M$ iid samples $x_i^{(1)},\dots,x_i^{(M)}$ drawn from each candidate mixture $q_i$, the following estimator for the reverse KL divergence can be constructed:

$$
\hat{D}_{\mathrm{KL}}(q_{\mathbf{w}} \| p) = \sum_{i=1}^N w_i \left( \frac{1}{M} \sum_{m=1}^M \log q_{\mathbf{w}}(x_i^{(m)}) - \log p(x_i^{(m)}) \right),
$$

where

$$
x_i^{(1)},\dots,x_i^{(M)} \overset{\mathrm{iid}}{\sim} q_i, \; i=1,\dots,N.
$$

## Alternative objective: _bidirectional_ KL divergence

Another alternative is to consider a bidirectional KL divergence, which combines the forward and reverse KL divergences:

$$
D_{\mathrm{biKL}}^{\alpha}(p, q_{\mathbf{w}}) = \alpha D_{\mathrm{KL}}(p \| q_{\mathbf{w}}) + (1-\alpha) D_{\mathrm{KL}}(q_{\mathbf{w}} \| p), \quad \alpha \in \[0,1\].
$$

The optimization problem would essentially be a weighted combination of the two
previous optimization problems, and can also be estimated using Monte Carlo
sampling.


## Example

The [`fit_mixture_weights`](../reference/index.md#gmm_divergence.fit_mixture_weights) function fits the weights of a mixture of candidate
distributions $q_i$ to a fixed reference mixture $p$. For example:

```python
from gmm_divergence import ForwardKL, GaussianMixture, fit_mixture_weights

p = GaussianMixture.from_arrays(
    weights=[0.6, 0.4], means=[[0.0], [2.0]], covariances=[[[0.5]], [[0.5]]]
)

q1 = GaussianMixture.from_arrays(weights=[1.0], means=[[0.0]], covariances=[[[0.5]]])

q2 = GaussianMixture.from_arrays(weights=[1.0], means=[[2.0]], covariances=[[[0.5]]])

result = fit_mixture_weights(p, [q1, q2], method="softmax-lbfgsb", objective=ForwardKL(rng=9126))

assert result.converged
assert abs(result.weights[0] - 0.6) < 1e-2
assert abs(result.weights[1] - 0.4) < 1e-2
print(result.fit_objective)
print(result.objective_value)
print(result.forward_kl.value)
print(result.reverse_kl.value)
```

Here, the optimizer recovers the mixture weights of the reference distribution by
combining the two candidate mixtures `q1` and `q2`. The result keeps the scalar
optimizer objective separate from the forward and reverse KL diagnostics, since
the optimized objective depends on the selected fit direction.

!!! info "Alternative metrics when using `fit_mixture_weights`"
    The `fit_mixture_weights` function also supports fitting mixture weights
    using the reverse KL divergence and the bidirectional KL divergence by
    setting the `objective` parameter. See the [API
    reference](../reference/index.md#gmm_divergence.fit_mixture_weights) for
    details.
