from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from gmm_divergence.typing import Covariances, FloatArray, Weights

GaussianComponentArrays: TypeAlias = tuple[Weights, FloatArray, Covariances]


class Distribution(ABC):
    """Base class for probability distributions used in divergence computations."""

    @abstractmethod
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Compute the log probability density at the given points."""
        ...

    @abstractmethod
    def sample(self, n_samples: int, rng: np.random.Generator | int | None = None) -> FloatArray:
        """Generate samples from the distribution."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        ...

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Compute the probability density at the given points."""
        return np.exp(self.logpdf(x))


class GaussianFamily(Distribution, ABC):
    """Marker class for distributions in the Gaussian family.

    This includes `Gaussian` and `GaussianMixture`, which can be represented in
    terms of component arrays (weights, means, covariances).
    """

    @abstractmethod
    def component_arrays(self) -> GaussianComponentArrays:
        """Return the component arrays for the distribution.

        Returns
        -------
        GaussianComponentArrays
            A tuple containing the weights, means, and covariances of the
            Gaussian components. For a single Gaussian, the weights should be
            an array of shape (1,) with value 1.0, and the means and covariances
            should have an extra leading dimension of size 1.
        """

    @property
    @override
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        _, means, _ = self.component_arrays()
        return means.shape[1]
