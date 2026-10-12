"""Component provenance for flattened mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt

from gmm_divergence._core._arrays import readonly_copy


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MixtureMapping:
    """Immediate-source provenance for every flattened mixture component.

    Parameters
    ----------
    source_index : array-like, shape (n_components,)
        Nonempty vector of nonnegative integer source positions.
    local_component_index : array-like, shape (n_components,)
        Nonnegative integer positions within each source, matching source_index.

    Notes
    -----
    Both arrays are copied into read-only intp arrays. Combination refers to
    immediate inputs; fitting refers to the original candidate sequence. Nested
    source mappings are not recursively flattened.
    """

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
        """Look up the source of a flattened component.

        Parameters
        ----------
        component_index : int
            Position in `[0, n_components)`; booleans are not accepted.

        Returns
        -------
        tuple of int
            Source index and local component index, respectively.

        Notes
        -----
        Index validation raises TypeError for noninteger positions and
        IndexError for positions outside the mapping.
        """
        validate_index(component_index, self.source_index.size)
        return int(self.source_index[component_index]), int(
            self.local_component_index[component_index]
        )

    def component_of(self, input_index: int, local_component_index: int) -> int:
        """Find a flattened component by its source and local position.

        Parameters
        ----------
        input_index : int
            Source position recorded in source_index.
        local_component_index : int
            Component position within that source.

        Returns
        -------
        int
            First matching flattened component position if rows are duplicated.

        Raises
        ------
        TypeError
            If either argument is not an integer or is a boolean.
        ValueError
            If the requested source and component pair is absent.
        """
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
    """Copy a nonempty 1D integer vector into read-only, nonnegative intp indices.

    name identifies invalid inputs in shape, dtype, and bounds error messages.
    """
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
    """Validate an integer position in [0, size), rejecting booleans and negatives."""
    if isinstance(index, bool) or not isinstance(index, Integral):
        msg = f"component index must be an integer, got {index!r}."
        raise TypeError(msg)
    if index < 0 or index >= size:
        msg = f"Component index {index} is out of bounds for {size} components."
        raise IndexError(msg)
