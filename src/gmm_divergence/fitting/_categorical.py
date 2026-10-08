"""Categorical-mixture weight fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from gmm_divergence.fitting._options import FitMethod
    from gmm_divergence.results import CategoricalMixtureFitResult


def fit_categorical_mixture_weights(
    p_w: npt.ArrayLike,  # pyright: ignore[reportUnusedParameter]
    q_i_w: npt.ArrayLike,  # pyright: ignore[reportUnusedParameter]
    /,
    method: FitMethod,  # pyright: ignore[reportUnusedParameter]
) -> CategoricalMixtureFitResult: ...
