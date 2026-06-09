from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

    from gmm_divergence._core._types import Weights
    from gmm_divergence.distributions._combine import CombinedGaussianMixture
    from gmm_divergence.fitting._options import FitObjective
    from gmm_divergence.fitting._selector import CandidateSelection


@dataclass(frozen=True, slots=True)
class DivergenceResult:
    """Result of a divergence estimation."""

    value: float
    """The estimated divergence."""
    method: str | None = None
    """The method used for estimation, if applicable."""
    num_samples: int | None = None
    """The number of samples used for estimation, if applicable."""


@dataclass(frozen=True, slots=True, repr=False)
class KLFitResult:
    """Result of fitting a Gaussian mixture to minimize KL divergence.

    Primary result when using
    [`fit_mixture_weights`][gmm_divergence.fitting.fit_mixture_weights]
    and related functions.
    """

    weights: Weights
    """The fitted mixture weights as a 1D array."""
    fit_objective: FitObjective
    """The KL objective used to fit the mixture weights."""
    objective_value: float
    """The final scalar objective value minimized by the optimizer."""
    forward_kl: DivergenceResult
    """Estimated forward KL divergence, ``KL(p || q_w)``."""
    reverse_kl: DivergenceResult
    """Estimated reverse KL divergence, ``KL(q_w || p)``."""
    scipy_result: OptimizeResult | None
    """The full result object returned by the optimization routine, if used."""
    fitted_mixture: CombinedGaussianMixture
    """The full combined Gaussian mixture corresponding to the fitted weights."""
    alpha: float | None = None
    """Forward-objective weight used by bidirectional fitting, if applicable."""
    iterations: int | None = None
    """The number of iterations taken by the optimization routine, if applicable."""
    converged: bool | None = None
    """Whether the optimization routine reported convergence, if applicable."""
    candidate_selection: CandidateSelection | None = None
    """The candidate selection results used during fitting, if applicable."""

    @override
    def __str__(self) -> str:
        return format_kl_fit_result(self)

    def display(
        self,
        *,
        max_components: int | None = None,
        min_weight: float | None = 1e-4,
        sort_by_weight: bool = True,
        precision: int = 4,
        source_labels: IndexLabels | None = None,
        component_labels: IndexLabels | None = None,
        local_component_labels: LocalIndexLabels | None = None,
    ) -> str:
        """Format the KL fit result as a compact human-readable summary."""
        return format_kl_fit_result(
            self,
            max_components=max_components,
            min_weight=min_weight,
            sort_by_weight=sort_by_weight,
            precision=precision,
            source_labels=source_labels,
            component_labels=component_labels,
            local_component_labels=local_component_labels,
        )


class SupportsStr(Protocol):
    @override
    def __str__(self) -> str: ...


IndexLabels: TypeAlias = Mapping[int, SupportsStr] | Sequence[SupportsStr]
LocalIndexLabels: TypeAlias = Mapping[tuple[int, int], SupportsStr]


def format_kl_fit_result(
    result: KLFitResult,
    *,
    max_components: int | None = None,
    min_weight: float | None = 1e-4,
    sort_by_weight: bool = True,
    precision: int = 4,
    source_labels: IndexLabels | None = None,
    component_labels: IndexLabels | None = None,
    local_component_labels: LocalIndexLabels | None = None,
) -> str:
    """Format a KL fit result as a compact human-readable summary."""
    prec: int = precision
    max_components = result.weights.size if max_components is None else max_components
    _validate_input(max_components, min_weight, prec)

    weights = np.asarray(result.weights, dtype=np.float64)
    component_count = len(weights)
    weight_sum = float(np.sum(weights))

    active_threshold = 0.0 if min_weight is None else min_weight
    active_count = np.count_nonzero(weights > active_threshold)

    order = np.argsort(-weights) if sort_by_weight else np.arange(component_count)
    if min_weight is not None:
        order = order[weights[order] >= min_weight]

    shown = order[:max_components]
    shown_mask = np.zeros(component_count, dtype=bool)
    shown_mask[shown] = True

    omitted_count = component_count - len(shown)
    omitted_mass = float(np.sum(weights[~shown_mask]))
    max_weight = float(np.max(weights)) if component_count else 0.0
    min_display = "-" if min_weight is None else f"{min_weight:.{prec}g}"
    forward_kl = result.forward_kl.value
    forward_kl_per_dim = forward_kl / result.fitted_mixture.mixture.dim
    reverse_kl = result.reverse_kl.value
    reverse_kl_per_dim = reverse_kl / result.fitted_mixture.mixture.dim
    lines = [
        "KL fit result",
        f"  fit objective:   {result.fit_objective}",
        f"  objective value: {result.objective_value:.{prec}g}",
        f"  forward KL:      {forward_kl:.{prec}g} ({forward_kl_per_dim:.{prec}g} per dim)",
        f"  reverse KL:      {reverse_kl:.{prec}g} ({reverse_kl_per_dim:.{prec}g} per dim)",
        f"  KL method:       {result.forward_kl.method or '-'}",
        f"  forward samples: {result.forward_kl.num_samples or '-'}",
        f"  reverse samples: {result.reverse_kl.num_samples or '-'}",
        f"  components:   {component_count}",
        f"  active:       {active_count} (weight > {active_threshold:.{prec}g})",
        f"  shown:        {len(shown)} (min weight {min_display})",
        f"  weight sum:   {weight_sum:.{prec}g}",
        f"  max weight:   {max_weight:.{prec}g}",
    ]
    if result.alpha is not None:
        lines.append(f"  alpha:        {result.alpha:.{prec}g}")
    if result.iterations is not None:
        lines.append(f"  iterations:   {result.iterations}")
    if result.converged is not None:
        lines.append(f"  converged:    {result.converged}")

    lines.extend(("", "  weights:"))

    if len(shown) == 0:
        lines.append("    <none>")
    else:
        rows: list[tuple[str, str, str, str, str]] = []
        for rank, component_index_ in enumerate(shown, start=1):
            component_index = int(component_index_)
            source_index, local_index = result.fitted_mixture.mapping.source_of(component_index)
            rows.append((
                str(rank),
                _index_label(component_index, component_labels),
                _index_label(source_index, source_labels),
                _local_index_label(source_index, local_index, local_component_labels),
                f"{weights[component_index]:.{prec}g}",
            ))

        lines.extend(
            _format_table(
                headers=("rank", "component", "source", "local", "weight"),
                rows=rows,
                right_aligned={0, 4},
            )
        )

    if omitted_count > 0:
        ommited_line: str = (
            f"  ... omitted {omitted_count} component(s), total weight {omitted_mass:.{prec}g}"
        )
        lines.append(ommited_line)

    if result.scipy_result is not None and not bool(result.scipy_result.success):
        lines.extend(("", f"  optimizer message: {result.scipy_result.message}"))
    return "\n".join(lines)


def _index_label(index: int, labels: IndexLabels | None) -> str:
    if labels is None:
        return str(index)
    try:
        return str(labels[index])
    except (IndexError, KeyError):
        return str(index)


def _local_index_label(source_index: int, local_index: int, labels: LocalIndexLabels | None) -> str:
    if labels is None:
        return str(local_index)
    return str(labels.get((source_index, local_index), local_index))


def _format_table(
    *, headers: tuple[str, ...], rows: Sequence[tuple[str, ...]], right_aligned: set[int]
) -> list[str]:
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]

    def format_cell(index: int, value: str) -> str:
        if index in right_aligned:
            return value.rjust(widths[index])
        return value.ljust(widths[index])

    lines = ["    " + "  ".join(starmap(format_cell, enumerate(headers)))]
    lines.extend("    " + "  ".join(starmap(format_cell, enumerate(row))) for row in rows)
    return lines


def _validate_input(max_components: int | None, min_weight: float | None, precision: int) -> None:
    if max_components is not None and max_components < 0:
        msg = "max_components must be non-negative."
        raise ValueError(msg)
    if min_weight is not None and min_weight < 0:
        msg = "min_weight must be non-negative or None."
        raise ValueError(msg)
    if precision < 0:
        msg = "precision must be non-negative."
        raise ValueError(msg)
