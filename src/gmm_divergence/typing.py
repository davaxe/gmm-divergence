from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
Weights: TypeAlias = FloatArray
Covariance: TypeAlias = FloatArray
Covariances: TypeAlias = FloatArray
