from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray


def check_covariance_shape(covariance: FloatArray) -> Literal["single", "batched"]:
    """Classify square covariance shape (d, d) as single or (n, d, d) as batched.

    Checks dimensionality and square trailing axes only. Does not check values,
    symmetry, positive definiteness, or whether dimensions are nonempty.
    """
    if covariance.ndim == 2:
        if covariance.shape[0] != covariance.shape[1]:
            msg = f"Expected covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "single"

    if covariance.ndim == 3:
        if covariance.shape[1] != covariance.shape[2]:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "batched"

    msg = f"covariance must have shape (d, d) or (n, d, d), got {covariance.shape}."
    raise ValueError(msg)


def validate_covariance_input(covariance: FloatArray) -> Literal["single", "batched"]:
    """Classify nonempty finite covariance input as single or batched.

    covariance must have shape (d, d) or (n, d, d) and be symmetric within
    NumPy's default allclose tolerance. Positive definiteness is not required;
    this validation is used before regularization and epsilon estimation.
    """
    shape_kind = check_covariance_shape(covariance)
    if covariance.shape[-1] == 0 or (shape_kind == "batched" and covariance.shape[0] == 0):
        msg = "covariance must contain at least one nonempty matrix."
        raise ValueError(msg)
    if not np.all(np.isfinite(covariance)):
        msg = "covariance must contain only finite values."
        raise ValueError(msg)
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2)):
        msg = "covariance must be symmetric."
        raise ValueError(msg)
    return shape_kind
