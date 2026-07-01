from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import gmm_divergence as gd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    from gmm_divergence._core._types import FloatArray


@dataclass(frozen=True)
class ObjectiveFit:
    objective: gd.FitObjective
    result: gd.KLFitResult

    @property
    def mixture(self) -> gd.GaussianMixture:
        return self.result.fitted_mixture.mixture


@dataclass(frozen=True)
class FitMetrics:
    forward_kl: float
    reverse_kl: float
    tv_distance: float
    overlap: float
    hellinger: float
    mean_error: float
    covariance_error: float


def target() -> gd.Gaussian:
    return gd.Gaussian.from_arrays(mean=[0.0, 0.0], covariance=[[1.25, 0.75], [0.75, 0.65]])


def components() -> list[gd.Gaussian]:
    return [
        gd.Gaussian.from_arrays(mean=[-0.05, -0.05], covariance=[[0.12, 0.05], [0.05, 0.10]]),
        gd.Gaussian.from_arrays(mean=[0.05, 0.05], covariance=[[3.20, 0.75], [0.75, 1.60]]),
        gd.Gaussian.from_arrays(mean=[1.45, 0.95], covariance=[[0.28, 0.12], [0.12, 0.22]]),
    ]


def gaussian_pdf_grid(gaussian: gd.Gaussian, x: FloatArray, y: FloatArray) -> FloatArray:
    pos = np.dstack((x, y))
    return np.asarray(multivariate_normal(mean=gaussian.mean, cov=gaussian.covariance).pdf(pos))


def mixture_pdf_grid(mixture: gd.GaussianMixture, x: FloatArray, y: FloatArray) -> FloatArray:
    z = np.zeros_like(x)

    for i in range(mixture.n_components):
        z += mixture.weights[i] * gaussian_pdf_grid(mixture.get_component(i), x, y)

    return z


def density_contour(
    ax: Axes, x: FloatArray, y: FloatArray, z: FloatArray, levels: FloatArray, **kwargs: object
) -> None:
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))

    valid_levels = levels[(levels > z_min) & (levels < z_max)]

    if valid_levels.size:
        _ = ax.contour(x, y, z, levels=valid_levels, **kwargs)


def make_grid(
    grid_min: float = -2.75, grid_max: float = 2.75, grid_size: int = 500
) -> tuple[FloatArray, FloatArray]:
    grid = np.linspace(grid_min, grid_max, grid_size)
    x, y = np.meshgrid(grid, grid)

    return x.astype(np.float64), y.astype(np.float64)


def grid_spacing(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    dx = float(abs(x[0, 1] - x[0, 0]))
    dy = float(abs(y[1, 0] - y[0, 0]))

    return dx, dy


def fit_weight_objectives(
    p: gd.Gaussian, q_i: list[gd.Gaussian], *, rng: int = 124, samples: int = 20_000
) -> list[ObjectiveFit]:
    objectives: tuple[gd.FitObjective, ...] = (
        "forward",
        "reverse",
        "bidirectional",
        "jensen_shannon",
    )

    fits: list[ObjectiveFit] = []
    for objective in objectives:
        if objective == "forward":
            fit_objective = gd.ForwardKL(sampling=gd.DrawSamples(samples, rng=rng))

        elif objective == "reverse":
            fit_objective = gd.ReverseKL(
                p_sampling=gd.DrawSamples(samples, rng=rng),
                q_sampling=gd.DrawSamples(samples, rng=rng),
            )

        elif objective == "bidirectional":
            fit_objective = gd.BidirectionalKL(
                p_sampling=gd.DrawSamples(samples, rng=rng),
                q_sampling=gd.DrawSamples(samples, rng=rng),
                alpha=0.5,
            )

        elif objective == "jensen_shannon":
            fit_objective = gd.JensenShannon(
                p_sampling=gd.DrawSamples(samples, rng=rng),
                q_sampling=gd.DrawSamples(samples, rng=rng),
            )

        else:
            msg = f"Unsupported objective: {objective}"
            raise ValueError(msg)

        result = gd.fit_mixture_weights(p, q_i, objective=fit_objective)

        fits.append(ObjectiveFit(objective=objective, result=result))

    return fits


def objective_display_name(objective: gd.FitObjective) -> str:
    return objective.replace("_", " ").title()


def compute_metrics(
    p: gd.Gaussian,
    fit: ObjectiveFit,
    p_density: FloatArray,
    mixture_density: FloatArray,
    x: FloatArray,
    y: FloatArray,
) -> FitMetrics:
    dx, dy = grid_spacing(x, y)
    cell_area = dx * dy

    l1_error = float(np.sum(np.abs(p_density - mixture_density)) * cell_area)
    tv_distance = 0.5 * l1_error

    overlap = float(np.sum(np.minimum(p_density, mixture_density)) * cell_area)

    bhattacharyya_coefficient = float(np.sum(np.sqrt(p_density * mixture_density)) * cell_area)
    bhattacharyya_coefficient = float(np.clip(bhattacharyya_coefficient, 0.0, 1.0))

    hellinger = float(np.sqrt(max(0.0, 1.0 - bhattacharyya_coefficient)))

    fitted_gaussian = fit.mixture.as_gaussian()

    mean_error = float(np.linalg.norm(p.mean - fitted_gaussian.mean))

    covariance_error = float(np.linalg.norm(p.covariance - fitted_gaussian.covariance, ord="fro"))

    return FitMetrics(
        forward_kl=float(fit.result.forward_kl.value),
        reverse_kl=float(fit.result.reverse_kl.value),
        tv_distance=tv_distance,
        overlap=overlap,
        hellinger=hellinger,
        mean_error=mean_error,
        covariance_error=covariance_error,
    )


def metrics_text(fit: ObjectiveFit, metrics: FitMetrics) -> str:
    weights = ", ".join(f"{weight:.2f}" for weight in fit.result.weights)

    return "\n".join((
        rf"$D_{{KL}}(p, q_w) = {metrics.forward_kl:.3f}$",
        rf"$D_{{KL}}(q_w, p) = {metrics.reverse_kl:.3f}$",
        "",
        rf"$TV = {metrics.tv_distance:.3f}$",
        rf"Overlap $= {metrics.overlap:.3f}$",
        rf"Hellinger $= {metrics.hellinger:.3f}$",
        "",
        rf"$\Vert \mu_p - \mu_q \Vert = {metrics.mean_error:.3f}$",
        rf"$\Vert \Sigma_p - \Sigma_q \Vert_F = {metrics.covariance_error:.3f}$",
        "",
        rf"$w = [{weights}]$",
    ))


def set_common_axis_style(
    ax: Axes, *, grid_min: float = -2.75, grid_max: float = 2.75, show_grid: bool = True
) -> None:
    ax.set_aspect("equal")

    _ = ax.set_xlim(grid_min, grid_max)
    _ = ax.set_ylim(grid_min, grid_max)

    _ = ax.set_xlabel("$x$")
    _ = ax.set_ylabel("$y$")

    if show_grid:
        ax.grid(alpha=0.25)


def add_density_overlays(
    ax: Axes,
    x: FloatArray,
    y: FloatArray,
    p_density: FloatArray,
    mixture_density: FloatArray,
    levels: FloatArray,
    *,
    target_linewidth: float = 1.0,
    mixture_linewidth: float = 1.0,
) -> None:
    density_contour(
        ax, x, y, p_density, levels, colors="black", linewidths=target_linewidth, linestyles="solid"
    )

    density_contour(
        ax,
        x,
        y,
        mixture_density,
        levels,
        colors="tab:blue",
        linewidths=mixture_linewidth,
        linestyles="dashed",
    )


def add_colorbar(fig: Figure, ax: Axes, image: ScalarMappable, label: str) -> None:
    _ = fig.colorbar(image, ax=ax, label=label, fraction=0.046, pad=0.04)


def plot_target_and_candidates_panel(
    ax: Axes,
    p: gd.Gaussian,
    q_i: list[gd.Gaussian],
    x: FloatArray,
    y: FloatArray,
    p_density: FloatArray,
    levels: FloatArray,
) -> None:
    density_contour(ax, x, y, p_density, levels, colors="black", linewidths=1.7, linestyles="solid")

    _ = ax.scatter(
        p.mean[0], p.mean[1], marker="x", s=85, linewidths=2.2, color="black", label="Target mean"
    )

    component_colors = ["tab:orange", "tab:green", "tab:red"]

    for i, gaussian in enumerate(q_i):
        q_density = gaussian_pdf_grid(gaussian, x, y)
        color = component_colors[i % len(component_colors)]

        density_contour(
            ax, x, y, q_density, levels, colors=[color], linewidths=1.2, linestyles="dashed"
        )

        _ = ax.scatter(
            gaussian.mean[0],
            gaussian.mean[1],
            marker="o",
            s=45,
            color=color,
            label=rf"Candidate $q_{i + 1}$ mean",
        )

    _ = ax.set_title("Target and candidate components")
    set_common_axis_style(ax)

    _ = ax.legend(frameon=False, loc="lower right", fontsize=9)


def plot_fit_panel(
    ax: Axes,
    p: gd.Gaussian,
    fit: ObjectiveFit,
    x: FloatArray,
    y: FloatArray,
    p_density: FloatArray,
    mixture_density: FloatArray,
    levels: FloatArray,
) -> None:
    mixture_mean = fit.mixture.as_gaussian().mean

    add_density_overlays(
        ax, x, y, p_density, mixture_density, levels, target_linewidth=1.7, mixture_linewidth=1.5
    )

    _ = ax.scatter(
        p.mean[0], p.mean[1], marker="x", s=85, linewidths=2.2, color="black", label="Target mean"
    )

    _ = ax.scatter(
        mixture_mean[0], mixture_mean[1], marker="o", s=60, color="tab:blue", label="Mixture mean"
    )

    _ = ax.set_title("Target and fitted mixture")
    set_common_axis_style(ax)

    _ = ax.legend(frameon=False, loc="lower right", fontsize=9)


def plot_metrics_panel(ax: Axes, fit: ObjectiveFit, metrics: FitMetrics) -> None:
    _ = ax.axis("off")
    _ = ax.text(
        0.02,
        0.98,
        metrics_text(fit, metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.30,
        bbox={
            "facecolor": "white",
            "alpha": 0.96,
            "edgecolor": "0.82",
            "boxstyle": "round,pad=0.45",
        },
    )


def plot_diagnostic_panel(
    fig: Figure,
    ax: Axes,
    x: FloatArray,
    y: FloatArray,
    values: FloatArray,
    p_density: FloatArray,
    mixture_density: FloatArray,
    levels: FloatArray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    image = ax.pcolormesh(x, y, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    add_density_overlays(
        ax, x, y, p_density, mixture_density, levels, target_linewidth=0.85, mixture_linewidth=0.85
    )

    _ = ax.set_title(title)
    set_common_axis_style(ax, show_grid=False)

    add_colorbar(fig, ax, image, colorbar_label)


def plot_objective_fit(
    p: gd.Gaussian,
    q_i: list[gd.Gaussian],
    fit: ObjectiveFit,
    x: FloatArray,
    y: FloatArray,
    p_density: FloatArray,
    levels: FloatArray,
) -> None:
    mixture_density = mixture_pdf_grid(fit.mixture, x, y)

    signed_error = p_density - mixture_density
    absolute_error = np.abs(signed_error)

    eps = 1e-12
    log_density_ratio = np.log(mixture_density + eps) - np.log(p_density + eps)
    log_density_ratio = np.clip(log_density_ratio, -4.0, 4.0)

    signed_error_limit = float(np.max(np.abs(signed_error)))
    absolute_error_limit = float(np.max(absolute_error))

    metrics = compute_metrics(p, fit, p_density, mixture_density, x, y)

    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)

    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.0])

    ax_problem = fig.add_subplot(gs[0, 0])
    ax_fit = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])

    ax_signed = fig.add_subplot(gs[1, 0])
    ax_absolute = fig.add_subplot(gs[1, 1], sharex=ax_signed, sharey=ax_signed)
    ax_log_ratio = fig.add_subplot(gs[1, 2], sharex=ax_signed, sharey=ax_signed)

    _ = fig.suptitle(
        f"{objective_display_name(fit.objective)} objective", fontsize=16, fontweight="bold"
    )

    plot_target_and_candidates_panel(ax_problem, p, q_i, x, y, p_density, levels)

    plot_fit_panel(ax_fit, p, fit, x, y, p_density, mixture_density, levels)

    plot_metrics_panel(ax_metrics, fit, metrics)

    plot_diagnostic_panel(
        fig,
        ax_signed,
        x,
        y,
        signed_error,
        p_density,
        mixture_density,
        levels,
        title=r"Signed error: $p(x) - q_w(x)$",
        colorbar_label=r"$p(x) - q_w(x)$",
        cmap="seismic",
        vmin=-signed_error_limit,
        vmax=signed_error_limit,
    )

    plot_diagnostic_panel(
        fig,
        ax_absolute,
        x,
        y,
        absolute_error,
        p_density,
        mixture_density,
        levels,
        title=r"Absolute error: $|p(x) - q_w(x)|$",
        colorbar_label=r"$|p(x) - q_w(x)|$",
        cmap="viridis",
        vmin=0.0,
        vmax=absolute_error_limit,
    )

    plot_diagnostic_panel(
        fig,
        ax_log_ratio,
        x,
        y,
        log_density_ratio,
        p_density,
        mixture_density,
        levels,
        title=r"Log-density ratio: $\log(q_w(x) / p(x))$",
        colorbar_label=r"$\log(q_w(x) / p(x))$",
        cmap="seismic",
        vmin=-5.0,
        vmax=5.0,
    )


def main() -> None:
    p = target()
    q_i = components()

    x, y = make_grid()
    p_density = gaussian_pdf_grid(p, x, y)

    levels = np.linspace(0.08 * float(np.max(p_density)), 0.9 * float(np.max(p_density)), 4).astype(
        np.float64
    )

    fits = fit_weight_objectives(p, q_i, rng=124, samples=20_000)

    for fit in fits:
        plot_objective_fit(p, q_i, fit, x, y, p_density, levels)

    plt.show()


if __name__ == "__main__":
    main()
