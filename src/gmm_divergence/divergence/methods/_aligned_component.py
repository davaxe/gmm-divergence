from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gmm_divergence.divergence._api import categorical_kl_divergence, component_kl_matrix
from gmm_divergence.results import AlignedKLResult

if TYPE_CHECKING:
    from gmm_divergence.distributions._mixture import GaussianMixture


def aligned_component_kl(p: GaussianMixture, q: GaussianMixture, /) -> AlignedKLResult:
    r"""Compute KL divergence between Gaussian mixtures with aligned components.

    Given two Gaussian mixtures with corresponding component indices,

    $$
    p(k,x) = \pi_{p,k}\mathcal{N}(x;\mu_{p,k},\Sigma_{p,k}),
    $$

    $$
    q(k,x) = \pi_{q,k}\mathcal{N}(x;\mu_{q,k},\Sigma_{q,k}),
    $$

    compute the exact joint KL divergence:

    $$
    D_{\mathrm{KL}}(p(k,x)\|q(k,x))
    =
    D_{\mathrm{KL}}(\pi_p\|\pi_q)
    +
    \sum_{k=1}^{K}
    \pi_{p,k}
    D_{\mathrm{KL}}(p_k\|q_k).
    $$

    Unlike ordinary GMM KL, this divergence assumes that component indices
    correspond between the two distributions.

    Parameters
    ----------
    p, q : GaussianMixture
        Gaussian mixtures with equal numbers of components and
        equal feature dimensions. Component indices must be aligned.

    Returns
    -------
    AlignedKLResult
        KL decomposition containing:

        - `value`: Total joint KL divergence.
        - `weight_kl`: KL divergence between mixture weights.
        - `component_kls`: Individual Gaussian component KL values.
        - `weighted_component_kl`: Weighted sum of component KL values.
    """
    if p.n_components != q.n_components:
        msg = (
            "Aligned-component KL requires equal numbers of components, "
            f"got {p.n_components} and {q.n_components}."
        )
        raise ValueError(msg)

    if p.dim != q.dim:
        msg = f"Aligned-component KL requires equal feature dimensions, got {p.dim} and {q.dim}."
        raise ValueError(msg)

    weight_kl = categorical_kl_divergence(p.weights, q.weights).value

    # Consider only calculating the diagonal of the component KL matrix.
    component_kls = np.diag(component_kl_matrix(p, q)).copy()
    weighted_component_kl = float(np.dot(p.weights, component_kls))

    return AlignedKLResult(
        value=weight_kl + weighted_component_kl,
        weight_kl=weight_kl,
        component_kls=component_kls,
        weighted_component_kl=weighted_component_kl,
    )
