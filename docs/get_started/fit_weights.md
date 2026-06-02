---
icon: lucide/chart-no-axes-combined
---

# Fitting mixture weights

## Formulation

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
q_w(x) = \sum_{i=1}^N w_i q_i(x)
$$

is minimized. Formally, this can be expressed as the following optimization problem:

\begin{equation}
\label{eq:mixture-weight-optimization}
\begin{aligned}
    \min_{\mathbf{w} \in \Delta_N}
    \quad & D_{\mathrm{KL}}\!\left(p \,\middle\|\, q_{\mathbf{w}}\right),
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


Importantly, the resulting mixture $q_w$ is itself a Gaussian mixture, with the number of components equal to the sum of the number of components across all $q_i$:

$$
q_w(x) = \sum_{i=1}^N w_i \sum_{j=1}^{K_i} \omega_{ij}\mathcal{N}(x;\nu_{ij},\Lambda_{ij}) = \sum_{i=1}^N \sum_{j=1}^{K_i} w_i \omega_{ij}\mathcal{N}(x;\nu_{ij},\Lambda_{ij}).
$$

## Practical objective

## Example
