"""Component provenance for flattened mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._arrays import readonly_copy


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MixtureMapping:
    """Original source and local component index for every mixture component."""

    source_index: npt.NDArray[np.intp]
    local_component_index: npt.NDArray[np.intp]

    def __init__(self, source_index: npt.ArrayLike, local_component_index: npt.ArrayLike) -> None:
        sources = index_array(source_index, name="source_index")
        components = index_array(local_component_index, name="local_component_index")
        if sources.shape != components.shape:
            msg = "mapping indices must be one-dimensional arrays with matching shapes."
            raise ValueError(msg)
        object.__setattr__(self, "source_index", sources)
        object.__setattr__(self, "local_component_index", components)

    def source_of(self, component_index: int) -> tuple[int, int]:
        """Return `(source_index, local_component_index)` of a flattened component."""
        validate_index(component_index, self.source_index.size)
        return int(self.source_index[component_index]), int(
            self.local_component_index[component_index]
        )

    def component_of(self, input_index: int, local_component_index: int) -> int:
        """Find a flattened component by its original source and local index."""
        for name, index in (
            ("input_index", input_index),
            ("local_component_index", local_component_index),
        ):
            if isinstance(index, bool) or not isinstance(index, Integral):
                msg = f"{name} must be an integer, got {index!r}."
                raise TypeError(msg)
        matches = np.flatnonzero(
            (self.source_index == input_index)
            & (self.local_component_index == local_component_index)
        )
        if matches.size == 0:
            msg = (
                f"No component found for input_index={input_index}, "
                f"local_component_index={local_component_index}."
            )
            raise ValueError(msg)
        return int(matches[0])


def index_array(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.intp]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        msg = f"{name} must be a nonempty 1D array."
        raise ValueError(msg)
    if array.dtype.kind not in {"i", "u"}:
        msg = f"{name} must contain integer indices."
        raise TypeError(msg)
    if np.any(array < 0) or np.any(array > np.iinfo(np.intp).max):
        msg = f"{name} must contain nonnegative indices representable as intp."
        raise IndexError(msg)
    return readonly_copy(array, dtype=np.intp)


def validate_index(index: int, size: int) -> None:
    if isinstance(index, bool) or not isinstance(index, Integral):
        msg = f"component index must be an integer, got {index!r}."
        raise TypeError(msg)
    if index < 0 or index >= size:
        msg = f"Component index {index} is out of bounds for {size} components."
        raise IndexError(msg)
