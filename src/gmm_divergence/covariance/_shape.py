from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from gmm_divergence._core._types import FloatArray


def check_covariance_shape(
    covariance: FloatArray, *, batched: bool | None = None
) -> Literal["single", "batched"]:
    """Return whether covariance input is single or batched after shape validation."""
    if covariance.ndim == 2:
        if batched is True:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)

        if covariance.shape[0] != covariance.shape[1]:
            msg = f"Expected covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "single"

    if covariance.ndim == 3:
        if batched is False:
            msg = f"Expected single covariance with shape (d, d), got {covariance.shape}."
            raise ValueError(msg)

        if covariance.shape[1] != covariance.shape[2]:
            msg = f"Expected batched covariance with shape (n, d, d), got {covariance.shape}."
            raise ValueError(msg)

        return "batched"

    msg = f"covariance must have shape (d, d) or (n, d, d), got {covariance.shape}."
    raise ValueError(msg)


def validate_covariance_input(
    covariance: FloatArray, *, batched: bool | None = None
) -> Literal["single", "batched"]:
    """Validate the shape and values of covariance regularizer input."""
    shape_kind = check_covariance_shape(covariance, batched=batched)
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
