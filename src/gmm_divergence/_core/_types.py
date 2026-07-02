from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

# Domain aliases represent validated arrays; produce them via validation helpers
# or by construction paths that preserve the documented invariants.
Weights: TypeAlias = FloatArray
Covariance: TypeAlias = FloatArray
Covariances: TypeAlias = FloatArray
