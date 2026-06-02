from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeAlias

import numpy as np
import numpy.typing as npt

from gmm_divergence.typing import PrecisionT, PrecisionT_co

GaussianComponentArrays: TypeAlias = tuple[
    npt.NDArray[PrecisionT],
    npt.NDArray[PrecisionT],
    npt.NDArray[PrecisionT],
]


class Distribution(ABC, Generic[PrecisionT_co]):
    """Base class for probability distributions used in divergence computations."""

    @abstractmethod
    def logpdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT_co]:
        """Compute the log probability density at the given points."""
        ...

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> npt.NDArray[PrecisionT_co]:
        """Generate samples from the distribution."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> type[PrecisionT_co]:
        """Return the data type of the distribution parameters."""
        ...

    def pdf(self, x: npt.ArrayLike) -> npt.NDArray[PrecisionT_co]:
        """Compute the probability density at the given points."""
        return np.exp(self.logpdf(x))
