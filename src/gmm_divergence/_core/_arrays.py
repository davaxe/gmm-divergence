"""Array ownership helpers shared by immutable containers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_Scalar = TypeVar("_Scalar", bound=np.generic)


def readonly_copy(values: npt.ArrayLike, *, dtype: type[_Scalar]) -> npt.NDArray[_Scalar]:
    """Return an independent array without changing the input's writeability."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array
