"""Static figures for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .heuristics import Route, route_length

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3de", "#a3a29b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
})


def _draw_routes(ax, df: pd.DataFrame, routes: list[Route], dist: np.ndarray, title: str) -> None:
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    for k, r in enumerate(routes):
        color = SERIES[k % len(SERIES)]
        path = [0, *r, 0]
        ax.plot(lon[path], lat[path], color=color, lw=2, zorder=2,
                label=f"Van {k + 1}: {len(r)} stops, {route_length(r, dist):.1f} mi")
        ax.scatter(lon[r], lat[r], s=46, color=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter(lon[0], lat[0], s=180, marker="s", color=INK, edgecolor="white", linewidth=2, zorder=4)
    ax.annotate("Depot (Edison)", (lon[0], lat[0]), xytext=(8, -14), textcoords="offset points",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def route_maps(df, baseline, optimized, dist, base_miles, opt_miles, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
    _draw_routes(axes[0], df, baseline, dist, f"Baseline: nearest-neighbor dispatch ({base_miles:.1f} mi)")
    _draw_routes(axes[1], df, optimized, dist, f"Optimized: exact MILP ({opt_miles:.1f} mi)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def comparison_bars(summary: pd.DataFrame, baseline: str, optimized: str, path) -> None:
    metrics = [("miles", "Miles driven", "{:.0f}"), ("driver_hours", "Driver hours", "{:.1f}"),
               ("daily_cost_usd", "Daily cost ($)", "${:,.0f}"), ("co2_kg", "CO₂ (kg)", "{:.0f}")]
    b = summary.set_index("plan").loc[baseline]
    o = summary.set_index("plan").loc[optimized]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (col, label, fmt) in zip(axes, metrics):
        vals = [b[col], o[col]]
        bars = ax.bar([0, 1], vals, width=0.6, color=[BASE, SERIES[0]])
        for x, v in zip([0, 1], vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
        pct = 100 * (o[col] - b[col]) / b[col]
        ax.set_title(f"{label}\n{pct:+.1f}%", loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xticks([0, 1], ["Baseline", "Optimized"])
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
