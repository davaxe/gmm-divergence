from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import TypeVar

PrecisionT = TypeVar("PrecisionT", np.float32, np.float64, default=np.float64)
PrecisionT_co = TypeVar("PrecisionT_co", np.float32, np.float64, covariant=True)
CovarianceType = Literal["full", "diag"]
