from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal

from gmm_divergence import (
    BidirectionalKL,
    ForwardKL,
    Gaussian,
    GaussianMixture,
    KLFitResult,
    MomentMatching,
    ReverseKL,
    fit_mixture_weights,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh

    from gmm_divergence.fitting import FitObjective
    from gmm_divergence.typing import FloatArray


FIT_OBJECTIVES: tuple[FitObjective, ...] = (
    "forward",
    "reverse",
    "bidirectional",
    "moment_matching",
)

BIDIRECTIONAL_ALPHA = 0.5

GRID_MIN = -2.75
GRID_MAX = 2.75
GRID_SIZE = 350

FIGSIZE = (18, 8)

TARGET_COLOR = "black"
MIXTURE_COLOR = "tab:blue"
REFERENCE_COLOR = "gray"
COMPONENT_COLORS = (
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)
RESIDUAL_COLORMAP = "seismic"
TARGET_CONTOUR_LINEWIDTH = 1.6
MIXTURE_CONTOUR_LINEWIDTH = 1.3
COMPONENT_CONTOUR_LINEWIDTH = 1.1
WEIGHTED_COMPONENT_CONTOUR_LINEWIDTH = 0.8
RESIDUAL_CONTOUR_LINEWIDTH = 0.9

TARGET_CONTOUR_LINESTYLE = "solid"
MIXTURE_CONTOUR_LINESTYLE = "dashed"
COMPONENT_CONTOUR_LINESTYLE = "dashed"

WEIGHTED_COMPONENT_ALPHA = 0.75
GRID_ALPHA = 0.25

TARGET_MEAN_MARKER = "x"
TARGET_MEAN_SIZE = 80
TARGET_MEAN_LINEWIDTH = 2

MIXTURE_MEAN_MARKER = "o"
MIXTURE_MEAN_SIZE = 52

COMPONENT_MEAN_MARKER = "o"
COMPONENT_MEAN_SIZE = 45

LEGEND_TARGET_MEAN_MARKERSIZE = 8
LEGEND_MIXTURE_MEAN_MARKERSIZE = 6


@dataclass(frozen=True)
class ObjectiveFit:
    objective: FitObjective
    result: KLFitResult

    @property
    def mixture(self) -> GaussianMixture:
        return self.result.fitted_mixture.mixture


def target() -> Gaussian:
    return Gaussian.from_arrays(mean=[0.0, 0.0], covariance=[[1.25, 0.75], [0.75, 0.65]])


def components() -> list[Gaussian]:
    return [
        # Reverse-KL-friendly:
        # tight component at the high-density core.
        Gaussian.from_arrays(mean=[-0.05, -0.05], covariance=[[0.22, 0.10], [0.10, 0.18]]),
        # Forward-KL-friendly:
        # broad component covering much of the target support,
        # but with extra mass outside the target.
        Gaussian.from_arrays(mean=[0.05, 0.05], covariance=[[2.30, 0.55], [0.55, 1.15]]),
        # Tail/shoulder component:
        # helps cover one end of the elongated target,
        # but is unattractive to reverse KL.
        Gaussian.from_arrays(mean=[1.45, 0.95], covariance=[[0.28, 0.12], [0.12, 0.22]]),
    ]


def component_color(index: int) -> str:
    return COMPONENT_COLORS[index % len(COMPONENT_COLORS)]


def gaussian_pdf_grid(gaussian: Gaussian, x: FloatArray, y: FloatArray) -> FloatArray:
    pos = np.dstack((x, y))

    return np.asarray(multivariate_normal(mean=gaussian.mean, cov=gaussian.covariance).pdf(pos))


def mixture_pdf_grid(mixture: GaussianMixture, x: FloatArray, y: FloatArray) -> FloatArray:
    z = np.zeros_like(x)

    for i in range(mixture.n_components):
        weight = mixture.weights[i]
        gaussian = mixture.get_component(i)
        z += weight * gaussian_pdf_grid(gaussian, x, y)

    return z


def component_pdf_grids(q_i: list[Gaussian], x: FloatArray, y: FloatArray) -> list[FloatArray]:
    return [gaussian_pdf_grid(q, x, y) for q in q_i]


def symmetric_plot_limit(values: list[FloatArray]) -> float:
    limit = 0.0

    for value in values:
        finite_values = value[np.isfinite(value)]
        if finite_values.size:
            limit = max(limit, float(np.max(np.abs(finite_values))))

    return limit if limit > 0.0 else 1.0


def density_contour(
    ax: Axes, x: FloatArray, y: FloatArray, z: FloatArray, levels: FloatArray, **kwargs: object
) -> None:
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    valid_levels = levels[(levels > z_min) & (levels < z_max)]
    if valid_levels.size:
        _ = ax.contour(x, y, z, levels=valid_levels, **kwargs)


def mixture_and_residual_grids(
    p_density: FloatArray, x: FloatArray, y: FloatArray, fits: list[ObjectiveFit]
) -> tuple[dict[FitObjective, FloatArray], dict[FitObjective, FloatArray], float]:
    mixture_grids: dict[FitObjective, FloatArray] = {
        fit.objective: mixture_pdf_grid(fit.mixture, x, y) for fit in fits
    }

    residual_grids: dict[FitObjective, FloatArray] = {
        objective: p_density - mixture_density
        for objective, mixture_density in mixture_grids.items()
    }

    residual_abs_limit = symmetric_plot_limit(list(residual_grids.values()))

    return mixture_grids, residual_grids, residual_abs_limit


def fit_weight_objectives(
    p: Gaussian, q_i: list[Gaussian], *, rng: int = 1024, samples: int = 20_000
) -> list[ObjectiveFit]:
    fits: list[ObjectiveFit] = []

    for objective in FIT_OBJECTIVES:
        result = fit_mixture_weights(
            p, q_i, objective=fit_objective_config(objective, samples, rng)
        )

        fits.append(ObjectiveFit(objective=objective, result=result))

    return fits


def fit_objective_config(
    objective: FitObjective, samples: int, rng: int
) -> ForwardKL | ReverseKL | BidirectionalKL | MomentMatching:
    match objective:
        case "forward":
            return ForwardKL(sampling=samples, rng=rng)
        case "reverse":
            return ReverseKL(p_sampling=samples, q_sampling=samples, rng=rng)
        case "bidirectional":
            return BidirectionalKL(
                p_sampling=samples, q_sampling=samples, alpha=BIDIRECTIONAL_ALPHA, rng=rng
            )
        case "moment_matching":
            return MomentMatching(fit_second_moments=True)


def objective_display_name(objective: FitObjective) -> str:
    return objective.replace("_", " ").title()


def objective_axis_title(fit: ObjectiveFit) -> str:
    return objective_display_name(fit.result.fit_objective)


def objective_metrics_text(fit: ObjectiveFit) -> str:
    weights = ", ".join(f"{weight:.2f}" for weight in fit.result.weights)
    forward_kl = fit.result.forward_kl.value
    reverse_kl = fit.result.reverse_kl.value
    return "\n".join((
        rf"$D_{{\mathrm{{KL}}}}(p, q_w) = {forward_kl:.3f}$",
        rf"$D_{{\mathrm{{KL}}}}(q_w, p) = {reverse_kl:.3f}$",
        rf"$w = [{weights}]$",
    ))


def plot_setup_panel(
    ax: Axes,
    p: Gaussian,
    q_i: list[Gaussian],
    component_grids: list[FloatArray],
    x: FloatArray,
    y: FloatArray,
    p_density: FloatArray,
    p_levels: FloatArray,
) -> None:
    density_contour(
        ax,
        x,
        y,
        p_density,
        p_levels,
        colors=TARGET_COLOR,
        linewidths=TARGET_CONTOUR_LINEWIDTH,
        linestyles=TARGET_CONTOUR_LINESTYLE,
    )

    _ = ax.scatter(
        p.mean[0],
        p.mean[1],
        marker=TARGET_MEAN_MARKER,
        s=TARGET_MEAN_SIZE,
        linewidths=TARGET_MEAN_LINEWIDTH,
        color=TARGET_COLOR,
    )

    for i, (gaussian, z_q) in enumerate(zip(q_i, component_grids, strict=True), start=1):
        color = component_color(i - 1)

        density_contour(
            ax,
            x,
            y,
            z_q,
            p_levels,
            colors=[color],
            linewidths=COMPONENT_CONTOUR_LINEWIDTH,
            linestyles=COMPONENT_CONTOUR_LINESTYLE,
        )

        _ = ax.scatter(
            gaussian.mean[0],
            gaussian.mean[1],
            marker=COMPONENT_MEAN_MARKER,
            s=COMPONENT_MEAN_SIZE,
            color=color,
        )

    _ = ax.set_title("Target $p$ and candidate components $q_i$")
    _ = ax.set_xlabel("$x$")
    _ = ax.set_ylabel("$y$")


def plot_residual_background(
    ax: Axes, x: FloatArray, y: FloatArray, residual: FloatArray, *, residual_abs_limit: float
) -> QuadMesh:
    return ax.pcolormesh(
        x,
        y,
        residual,
        cmap=RESIDUAL_COLORMAP,
        vmin=-residual_abs_limit,
        vmax=residual_abs_limit,
        shading="auto",
    )


def plot_weight_fits(p: Gaussian, q_i: list[Gaussian], fits: list[ObjectiveFit]) -> None:
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_SIZE)
    x, y = np.meshgrid(grid, grid)

    fig, axes = plt.subplot_mosaic(
        [
            ["setup", "forward", "reverse", "bidirectional", "moment_matching"],
            [
                "setup",
                "forward_residual",
                "reverse_residual",
                "bidirectional_residual",
                "moment_matching_residual",
            ],
        ],
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    fit_by_objective = {fit.objective: fit for fit in fits}

    p_density = gaussian_pdf_grid(p, x, y)
    component_grids = component_pdf_grids(q_i, x, y)

    p_levels = np.linspace(
        0.08 * float(np.max(p_density)), 0.9 * float(np.max(p_density)), 4
    ).astype(np.float64)

    plot_setup_panel(axes["setup"], p, q_i, component_grids, x, y, p_density, p_levels)

    mixture_grids, residual_grids, residual_abs_limit = mixture_and_residual_grids(
        p_density, x, y, fits
    )

    residual_images: list[QuadMesh] = []

    for objective in FIT_OBJECTIVES:
        fit = fit_by_objective[objective]
        z_mix = mixture_grids[objective]
        residual = residual_grids[objective]
        mixture_mean = fit.mixture.as_gaussian().mean

        ax_contour = axes[objective]

        density_contour(
            ax_contour,
            x,
            y,
            p_density,
            p_levels,
            colors=TARGET_COLOR,
            linewidths=TARGET_CONTOUR_LINEWIDTH,
            linestyles=TARGET_CONTOUR_LINESTYLE,
        )

        density_contour(
            ax_contour,
            x,
            y,
            z_mix,
            p_levels,
            colors=MIXTURE_COLOR,
            linewidths=MIXTURE_CONTOUR_LINEWIDTH,
            linestyles=MIXTURE_CONTOUR_LINESTYLE,
        )

        _ = ax_contour.scatter(
            p.mean[0],
            p.mean[1],
            marker=TARGET_MEAN_MARKER,
            s=TARGET_MEAN_SIZE,
            linewidths=TARGET_MEAN_LINEWIDTH,
            color=TARGET_COLOR,
        )

        _ = ax_contour.scatter(
            mixture_mean[0],
            mixture_mean[1],
            marker=MIXTURE_MEAN_MARKER,
            s=MIXTURE_MEAN_SIZE,
            color=MIXTURE_COLOR,
        )

        _ = ax_contour.set_title(objective_axis_title(fit))
        _ = ax_contour.text(
            0.03,
            0.97,
            objective_metrics_text(fit),
            transform=ax_contour.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 3.0},
        )
        ax_residual = axes[f"{objective}_residual"]
        residual_image = plot_residual_background(
            ax_residual, x, y, residual, residual_abs_limit=residual_abs_limit
        )

        residual_images.append(residual_image)

        density_contour(
            ax_residual,
            x,
            y,
            p_density,
            p_levels,
            colors=TARGET_COLOR,
            linewidths=RESIDUAL_CONTOUR_LINEWIDTH,
            linestyles=TARGET_CONTOUR_LINESTYLE,
        )

        density_contour(
            ax_residual,
            x,
            y,
            z_mix,
            p_levels,
            colors=MIXTURE_COLOR,
            linewidths=RESIDUAL_CONTOUR_LINEWIDTH,
            linestyles=MIXTURE_CONTOUR_LINESTYLE,
        )

        _ = ax_residual.set_title(r"$p(x)-q_w(x)$")
        _ = ax_residual.set_xlabel("$x$")

    for ax in axes.values():
        ax.set_aspect("equal")
        _ = ax.set_xlim(GRID_MIN, GRID_MAX)
        _ = ax.set_ylim(GRID_MIN, GRID_MAX)
        ax.grid(alpha=GRID_ALPHA)

    for objective in FIT_OBJECTIVES:
        axes[f"{objective}_residual"].grid(visible=False)

    _ = axes["forward_residual"].set_ylabel("$y$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_COLOR,
            linestyle=TARGET_CONTOUR_LINESTYLE,
            linewidth=TARGET_CONTOUR_LINEWIDTH,
            label="Target $p$",
        ),
        Line2D(
            [0],
            [0],
            color=MIXTURE_COLOR,
            linestyle=MIXTURE_CONTOUR_LINESTYLE,
            linewidth=TARGET_CONTOUR_LINEWIDTH,
            label="Fitted mixture $q_w$",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            linestyle=COMPONENT_CONTOUR_LINESTYLE,
            linewidth=MIXTURE_CONTOUR_LINEWIDTH,
            label="Candidate components $q_i$",
        ),
        Line2D(
            [0],
            [0],
            color=TARGET_COLOR,
            marker=TARGET_MEAN_MARKER,
            linestyle="None",
            markersize=LEGEND_TARGET_MEAN_MARKERSIZE,
            label="Target mean",
        ),
        Line2D(
            [0],
            [0],
            color=MIXTURE_COLOR,
            marker=MIXTURE_MEAN_MARKER,
            linestyle="None",
            markersize=LEGEND_MIXTURE_MEAN_MARKERSIZE,
            label="Mixture mean",
        ),
    ]
    _ = fig.legend(handles=legend_handles, loc="outside lower center", ncols=6, frameon=False)

    _ = fig.colorbar(
        residual_images[-1],
        ax=[axes[f"{objective}_residual"] for objective in FIT_OBJECTIVES],
        label=r"$p(x)-q_w(x)$",
        shrink=0.86,
    )
    plt.show()


def main() -> None:
    p = target()
    q_i = components()
    fits = fit_weight_objectives(p, q_i, rng=124, samples=20_000)
    plot_weight_fits(p, q_i, fits)


if __name__ == "__main__":
    main()
