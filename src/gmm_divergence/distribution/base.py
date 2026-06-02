from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from gmm_divergence.typing import FloatArray

GaussianComponentArrays: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]


class Distribution(ABC):
    """Base class for probability distributions used in divergence computations."""

    @abstractmethod
    def logpdf(self, x: npt.ArrayLike) -> FloatArray:
        """Compute the log probability density at the given points."""
        ...

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> FloatArray:
        """Generate samples from the distribution."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        ...

    @property
    def dtype(self) -> type[np.float64]:
        """Return the data type of the distribution parameters."""
        return np.float64

    def pdf(self, x: npt.ArrayLike) -> FloatArray:
        """Compute the probability density at the given points."""
        return np.exp(self.logpdf(x))
