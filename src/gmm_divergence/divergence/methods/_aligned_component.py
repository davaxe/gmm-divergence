from __future__ import annotations

import numpy as np

from gmm_divergence._core._numeric import gaussian_kl_from_factors
from gmm_divergence.distributions._mixture import GaussianMixture
from gmm_divergence.divergence.methods._categorical import kl_categorical
from gmm_divergence.results import AlignedJointKLResult


def aligned_joint_kl_divergence(p: GaussianMixture, q: GaussianMixture) -> AlignedJointKLResult:
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

    This compares the joint distributions of component label and observation,
    assuming component indices correspond. No component matching is performed.
    It is not the marginal KL between mixture densities. Values are in nats.

    Parameters
    ----------
    p, q : GaussianMixture
        Gaussian mixtures with equal numbers of components and
        equal feature dimensions. Component indices must be aligned.

    Returns
    -------
    AlignedJointKLResult
        KL decomposition containing:

        - `value`: Total joint KL divergence.
        - `weight_kl`: KL divergence between mixture weights.
        - `component_kls`: Individual Gaussian component KL values.
        - `weighted_component_kl`: Weighted sum of component KL values.
    """
    _validate_mixture(p, name="p")
    _validate_mixture(q, name="q")
    if p.n_components != q.n_components:
        msg = (
            "Aligned-component KL requires equal numbers of components, "
            f"got {p.n_components} and {q.n_components}."
        )
        raise ValueError(msg)

    if p.n_features != q.n_features:
        msg = (
            f"Aligned-component KL requires equal feature dimensions, got "
            f"{p.n_features} and {q.n_features}."
        )
        raise ValueError(msg)

    weight_kl = kl_categorical(p.weights, q.weights).value

    component_kls = gaussian_kl_from_factors(
        p.covariances, q.means - p.means, q.cholesky(), p.log_determinants(), q.log_determinants()
    )
    component_kls.setflags(write=False)
    weighted_component_kl = float(np.dot(p.weights, component_kls))

    return AlignedJointKLResult(
        value=weight_kl + weighted_component_kl,
        weight_kl=weight_kl,
        component_kls=component_kls,
        weighted_component_kl=weighted_component_kl,
    )


def _validate_mixture(value: object, *, name: str) -> None:
    if not isinstance(value, GaussianMixture):
        msg = f"{name} must be a GaussianMixture with aligned components."
        raise TypeError(msg)
