from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import starmap
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from gmm_divergence.results import KLFitResult


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
    est_kl: float = result.estimated_kl.value
    est_kl_per_dim = est_kl / result.fitted_mixture.mixture.dim
    lines = [
        "KL fit result",
        f"  objective:    {result.objective:.{prec}g}",
        f"  estimated KL: {result.estimated_kl.value:.{prec}g} ({est_kl_per_dim:.{prec}g} per dim)",
        f"  KL method:    {result.estimated_kl.method or '-'}",
        f"  samples:      {result.estimated_kl.num_samples or '-'}",
        f"  components:   {component_count}",
        f"  active:       {active_count} (weight > {active_threshold:.{prec}g})",
        f"  shown:        {len(shown)} (min weight {min_display})",
        f"  weight sum:   {weight_sum:.{prec}g}",
        f"  max weight:   {max_weight:.{prec}g}",
    ]
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
