from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt

from gmm_divergence.distribution.gaussian import Gaussian
from gmm_divergence.distribution.gmm import GaussianMixture
from gmm_divergence.utils import as_weights

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class MixtureMapping:
    source_index: npt.NDArray[np.intp]
    """Index of the original input distribution for each flattened output component."""

    local_component_index: npt.NDArray[np.intp]
    """Component index within the original input mixture for each flattened output component."""

    def source_of(self, component_index: int) -> tuple[int, int]:
        """Return source input and local component index for flattened component.

        Returns
        -------
        tuple[int, int]
            `(input_index, local_component_index)`.
        """
        return self.source_index[component_index], self.local_component_index[component_index]

    def component_of(self, input_index: int, local_component_index: int) -> int:
        """Return the flattened component of given source input and local component.

        Returns
        -------
        int
            The flattened component index corresponding to the specified source
            input and local component.
        """
        mask = (self.source_index == input_index) & (
            self.local_component_index == local_component_index
        )
        if not np.any(mask):
            msg = (
                f"No component found for input_index={input_index},"
                f" local_component_index={local_component_index}"
            )
            raise ValueError(msg)
        return np.where(mask)[0][0]


@dataclass(frozen=True, slots=True)
class CombinedGaussianMixture:
    """Wrapper dataclass for a combined GaussianMixture."""

    mixture: GaussianMixture
    mapping: MixtureMapping


@overload
def combine_gaussians(
    sources: Sequence[Gaussian | GaussianMixture],
    weights: npt.ArrayLike | None = None,
    *,
    include_mapping: Literal[False] = False,
) -> GaussianMixture: ...


@overload
def combine_gaussians(
    sources: Sequence[Gaussian | GaussianMixture],
    weights: npt.ArrayLike | None = None,
    *,
    include_mapping: Literal[True],
) -> CombinedGaussianMixture: ...


def combine_gaussians(
    sources: Sequence[Gaussian | GaussianMixture],
    weights: npt.ArrayLike | None = None,
    *,
    include_mapping: bool = False,
) -> GaussianMixture | CombinedGaussianMixture:
    """Combine Gaussian mixtures and/or Gaussians into a single GaussianMixture.

    A single Gaussian is internally converted to a GaussianMixture with one
    component. The resulting GaussianMixture has a number of components equal to
    the sum of the number of components in the input.

    Parameters
    ----------
    sources : Sequence[Gaussian | GaussianMixture]
        The input distributions to combine.
    weights : ArrayLike, optional
        Weights for each input distribution. If None, equal weights are used.
    include_mapping : bool, default=False
        Whether to return a CombinedMixture with mapping information. If False,
        only the combined GaussianMixture is returned. The mapping information
        includes the index of the original input distribution and the local
        component index within that distribution for each component in the
        combined mixture.

    Returns
    -------
    GaussianMixture or CombinedMixture
        The combined GaussianMixture. If `return_mapping` is True, a
        CombinedMixture containing the combined mixture and mapping information
        is returned.

    """
    if len(sources) == 0:
        msg = "components must contain at least one distribution"
        raise ValueError(msg)

    weights_: list[float]
    if weights is None:
        weights_ = [1 / len(sources)] * len(sources)
    else:
        weights_arr = as_weights(weights, expected_length=len(sources), name="weights")
        weights_ = weights_arr.tolist()

    mixtures = [as_mixture(component) for component in sources]
    mixture = GaussianMixture(
        weights=np.concatenate([w * m.weights for w, m in zip(weights_, mixtures, strict=True)]),
        means=np.concatenate([m.means for m in mixtures], axis=0),
        covariances=np.concatenate([m.covariances for m in mixtures], axis=0),
    )

    if not include_mapping:
        return mixture

    mapping = MixtureMapping(
        source_index=np.concatenate([
            np.full(len(m.weights), i, dtype=np.intp) for i, m in enumerate(mixtures)
        ]),
        local_component_index=np.concatenate([
            np.arange(len(m.weights), dtype=np.intp) for m in mixtures
        ]),
    )
    return CombinedGaussianMixture(mixture=mixture, mapping=mapping)


def as_mixture(distribution: Gaussian | GaussianMixture) -> GaussianMixture:
    """Convert a Gaussian or GaussianMixture to a GaussianMixture."""
    if isinstance(distribution, Gaussian):
        return GaussianMixture(
            weights=np.ones(1, dtype=np.float64),
            means=distribution.mean[None, :],
            covariances=distribution.covariance[None, :, :],
        )
    return distribution
