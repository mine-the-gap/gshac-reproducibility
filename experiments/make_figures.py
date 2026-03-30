"""
make_figures.py

Generate all paper figures from benchmark CSV results.
Run after benchmark.py completes:

    python3 experiments/make_figures.py

Outputs (in paper/):
    fig_scaling.pdf          — 2×3 grid: time (top) + memory (bottom) × 3 h_max
    fig_scenarios.pdf        — illustration of tight / moderate / loose scenarios
    fig_limits.pdf           — scalability projection with dense infeasibility region
"""

from __future__ import annotations

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "paper")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

SCENARIO_STYLES = {
    "tight":    dict(color="#2CA02C", marker="^"),
    "moderate": dict(color="#FF7F0E", marker="s"),
    "loose":    dict(color="#1F77B4", marker="o"),
}

SCENARIO_LABELS = {
    "tight":    "Tight",
    "moderate": "Moderate",
    "loose":    "Loose",
}


# ── Helper: read benchmark CSV ────────────────────────────────────────

def read_benchmark_csv(path):
    """Read CSV, auto-detect old (no mean/sd) vs new column format."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for key in r:
                try:
                    r[key] = float(r[key])
                except (ValueError, TypeError):
                    pass
            # Normalise old column names → new names
            if "dense_time_s" in r and "dense_time_mean" not in r:
                r["dense_time_mean"] = r.get("dense_time_s", "")
                r["dense_mem_mean"]  = r.get("dense_mem_mib", "")
                r["sparse_time_mean"] = r.get("sparse_time_s", "")
                r["sparse_mem_mean"]  = r.get("sparse_mem_mib", "")
            rows.append(r)
    return rows


# ── Helper: synthetic point generator (same as benchmark.py) ──────────

def generate_clustered_points(n_total, n_centres, spread_m,
                              domain_m=500_000, seed=42):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, domain_m, size=(n_centres, 2))
    assignments = rng.integers(0, n_centres, size=n_total)
    noise = rng.normal(0, spread_m, size=(n_total, 2))
    return centres[assignments] + noise


# =====================================================================
# Figure 1: Scenario illustration
# =====================================================================

def make_scenario_figure(out_path):
    """Three-panel conceptual illustration of density scenarios.

    Uses a compact domain (20 km) with h_max = 5 km so that the cluster
    structure, edge connectivity, and component boundaries are all
    clearly visible at print resolution.  Parameters are proportional
    to the benchmark scenarios (same spread_factor × h_max ratios).
    """
    from matplotlib.collections import LineCollection

    SCENARIOS = [
        ("Tight",    dict(n_centres=15, spread_factor=0.03)),
        ("Moderate", dict(n_centres=8,  spread_factor=0.10)),
        ("Loose",    dict(n_centres=4,  spread_factor=0.30)),
    ]

    n = 400
    h_max = 5_000        # metres
    domain_m = 20_000    # 20 km — compact enough to show structure

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    for ax, (label, cfg) in zip(axes, SCENARIOS):
        spread_m = cfg["spread_factor"] * h_max
        pts = generate_clustered_points(
            n, cfg["n_centres"], spread_m, domain_m=domain_m, seed=42)

        # Build distance graph
        tree = cKDTree(pts)
        pairs = tree.query_pairs(h_max, output_type="ndarray")

        mean_k = 2 * len(pairs) / n
        graph_pct = 2 * len(pairs) / (n * (n - 1)) * 100

        # Convert to km
        pts_km = pts / 1000

        # Draw edges
        if len(pairs) > 0:
            segs = np.stack([pts_km[pairs[:, 0]],
                             pts_km[pairs[:, 1]]], axis=1)
            lc = LineCollection(segs, colors="#b0b0b0", linewidths=0.15,
                                alpha=0.3, zorder=1)
            ax.add_collection(lc)

        # Draw points
        ax.scatter(pts_km[:, 0], pts_km[:, 1], s=4, color="black",
                   alpha=0.8, zorder=2, linewidths=0)

        # Draw h_max scale bar in upper-right corner
        bar_x1 = domain_m / 1000 - 0.5
        bar_x0 = bar_x1 - h_max / 1000
        bar_y = domain_m / 1000 + 0.8
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#d62728",
                linewidth=2.0, solid_capstyle="butt", clip_on=False)
        ax.text((bar_x0 + bar_x1) / 2, bar_y + 0.5,
                r"$h_{max}$", fontsize=7, ha="center", color="#d62728")

        # Title + subtitle combined, well spaced
        title_str = (f"{label}  "
                     f"(C={cfg['n_centres']}, "
                     r"$\sigma$" + f"={cfg['spread_factor']}" +
                     r"$\,h_{max}$)")
        ax.set_title(title_str, fontsize=9, pad=18)

        # Stats below plot
        ax.text(0.5, -0.15,
                r"$\bar{k}$" + f" = {mean_k:.0f},  "
                f"density = {graph_pct:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=7.5,
                color="#555555")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, domain_m / 1000 + 0.5)
        ax.set_ylim(-0.5, domain_m / 1000 + 0.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("km", fontsize=8)

    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.10, top=0.82)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# =====================================================================
# Figure 2: Consolidated scaling (2×3 grid) — Option A
# =====================================================================

def make_scaling_figure(rows, out_path):
    """2×3 grid: top row = wall time, bottom row = peak memory."""

    h_maxes = sorted(set(r["h_max_km"] for r in rows))
    fig, axes = plt.subplots(2, len(h_maxes), figsize=(7.16, 4.4),
                             sharex="col")

    metrics = [
        ("dense_time_mean", "sparse_time_mean", "Wall time (s)"),
        ("dense_mem_mean",  "sparse_mem_mean",  "Peak memory (MiB)"),
    ]

    for row_idx, (dense_key, sparse_key, ylabel) in enumerate(metrics):
        for col_idx, h in enumerate(h_maxes):
            ax = axes[row_idx, col_idx]

            for scen, sty in SCENARIO_STYLES.items():
                sub = sorted(
                    [r for r in rows
                     if r["h_max_km"] == h and r["scenario"] == scen],
                    key=lambda r: r["n"],
                )
                xs = [r["n"] for r in sub]

                # Sparse (always available)
                ys = [r[sparse_key] for r in sub]
                ax.plot(xs, ys, color=sty["color"], linestyle="--",
                        marker=sty["marker"], markersize=4, linewidth=1.2,
                        alpha=0.7)

                # Dense (only where feasible)
                xs_d = [r["n"] for r in sub if r[dense_key] != ""]
                yd   = [r[dense_key] for r in sub if r[dense_key] != ""]
                if xs_d:
                    ax.plot(xs_d, yd, color=sty["color"], linestyle="-",
                            marker=sty["marker"], markersize=4,
                            linewidth=1.2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle=":", linewidth=0.4,
                    alpha=0.6)

            if row_idx == 0:
                ax.set_title(f"$h_{{max}}$ = {h:.0f} km", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("$n$")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    # Legend: scenario colours + dense/sparse line styles
    legend_elements = []
    for scen, sty in SCENARIO_STYLES.items():
        legend_elements.append(
            Line2D([0], [0], color=sty["color"], marker=sty["marker"],
                   markersize=4, linewidth=1.2, linestyle="-",
                   label=f"{SCENARIO_LABELS[scen]}"))
    legend_elements.append(
        Line2D([0], [0], color="grey", linestyle="-", linewidth=1.2,
               label="Dense (fastcluster)"))
    legend_elements.append(
        Line2D([0], [0], color="grey", linestyle="--", linewidth=1.2,
               label="GSHAC"))

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(h_pad=0.8)
    fig.subplots_adjust(bottom=0.13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# =====================================================================
# Figure 3: Scalability limits — measured + extrapolated
# =====================================================================

def make_limits_figure(rows, out_path):
    """
    Two-panel figure:
      Left:  time vs n  (measured sparse + extrapolated, dense infeasible zone)
      Right: memory vs n (same structure)

    Shows where dense becomes infeasible and where sparse remains practical.
    Adds annotation about parallel processing potential.
    """

    # Use constant-k benchmark data if available (real-world scaling regime)
    constk_path = os.path.join(RESULTS_DIR, "benchmark_constk.csv")
    if os.path.exists(constk_path):
        ck_rows = read_benchmark_csv(constk_path)
        ck_rows.sort(key=lambda r: r["n"])
    else:
        # Fallback to moderate h=10km from main benchmark
        ck_rows = sorted(
            [r for r in rows
             if r["h_max_km"] == 10.0 and r["scenario"] == "moderate"],
            key=lambda r: r["n"],
        )

    ns        = np.array([r["n"] for r in ck_rows])
    sp_time   = np.array([r["sparse_time_mean"] for r in ck_rows])
    sp_mem    = np.array([r["sparse_mem_mean"]  for r in ck_rows])
    dn_time   = np.array([r["dense_time_mean"] if r["dense_time_mean"] != "" else np.nan
                          for r in ck_rows])
    dn_mem    = np.array([r["dense_mem_mean"] if r["dense_mem_mean"] != "" else np.nan
                          for r in ck_rows])

    # Extrapolate sparse to larger n (O(n log n) for graph + O(n k) for HAC)
    # Fit log-log slope from measured data
    valid = sp_time > 0
    log_n = np.log(ns[valid])
    log_t = np.log(sp_time[valid])
    slope_t, intercept_t = np.polyfit(log_n, log_t, 1)

    log_m = np.log(sp_mem[valid])
    slope_m, intercept_m = np.polyfit(log_n, log_m, 1)

    n_max = 7.5  # log10 upper limit for extrapolation lines
    n_extrap = np.logspace(np.log10(ns[0]), n_max, 100)

    sp_time_extrap = np.exp(intercept_t) * n_extrap ** slope_t
    sp_mem_extrap  = np.exp(intercept_m) * n_extrap ** slope_m

    # Dense O(n^2) extrapolation from last measured point
    dn_valid = ~np.isnan(dn_time)
    if dn_valid.any():
        n_ref = ns[dn_valid][-1]
        t_ref = dn_time[dn_valid][-1]
        m_ref = dn_mem[dn_valid][-1]
    else:
        n_ref = ns[0]
        t_ref = dn_time[0] if not np.isnan(dn_time[0]) else 1
        m_ref = dn_mem[0]  if not np.isnan(dn_mem[0])  else 1

    dn_time_extrap = t_ref * (n_extrap / n_ref) ** 2
    dn_mem_extrap  = m_ref * (n_extrap / n_ref) ** 2

    # RAM limit
    RAM_MIB = 30 * 1024  # 30 GiB

    # Real-world reference points (from benchmark CSVs)
    rw_points = [
        {"label": "$n$=261K", "n": 261_073, "time": 11.8,
         "mem": 1855 + 109, "t_ofs": (1, 6), "m_ofs": (1, 6)},
        {"label": "$n$=2M", "n": 2_000_000, "time": 165.3,
         "mem": 919 + 441, "t_ofs": (1, 6), "m_ofs": (1, 6)},
    ]

    # Find n where dense memory crosses 30 GiB RAM
    n_infeasible = n_extrap[np.searchsorted(dn_mem_extrap, RAM_MIB)]

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    for ax in (ax_t, ax_m):
        ax.axvspan(n_infeasible, 10**7.5, alpha=0.08,
                   color="red", zorder=0, label="_nolegend_")
        ax.axvline(n_infeasible, color="#d62728", linestyle="-.",
                   linewidth=0.7, alpha=0.5, zorder=0)

    # Helper: place inline label along an extrapolation line
    def inline_label(ax, x_arr, y_arr, text, color, idx=-3):
        """Place text along the slope of a line at position idx."""
        x0, x1 = np.log10(x_arr[idx - 1]), np.log10(x_arr[idx])
        y0, y1 = np.log10(y_arr[idx - 1]), np.log10(y_arr[idx])
        # Get slope in display coords for rotation
        p0 = ax.transData.transform((x_arr[idx - 1], y_arr[idx - 1]))
        p1 = ax.transData.transform((x_arr[idx], y_arr[idx]))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(x_arr[idx], y_arr[idx], text, fontsize=7, color=color,
                rotation=angle, rotation_mode="anchor",
                ha="right", va="bottom")

    # ── Left panel: time ──
    ax_t.plot(n_extrap, dn_time_extrap, color="#d62728", linestyle=":",
              linewidth=1.0, alpha=0.7)
    ax_t.plot(ns[dn_valid], dn_time[dn_valid], color="#d62728",
              linestyle="-", marker="s", markersize=4, linewidth=1.2,
              label="Dense (fastcluster)")
    ax_t.plot(n_extrap, sp_time_extrap, color="#1f77b4", linestyle=":",
              linewidth=1.0, alpha=0.7)
    ax_t.plot(ns, sp_time, color="#1f77b4", linestyle="-", marker="o",
              markersize=4, linewidth=1.2, label="GSHAC (constant $\\bar{k}$)")

    for rw in rw_points:
        ax_t.plot(rw["n"], rw["time"], marker="*", markersize=12,
                  color="#2ca02c", zorder=5)
        ax_t.annotate(rw["label"], (rw["n"], rw["time"]),
                      textcoords="offset points", xytext=rw["t_ofs"],
                      fontsize=7, color="#2ca02c")

    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("$n$")
    ax_t.set_ylabel("Wall time (s)")
    # ax_t.set_title("Time scalability")
    ax_t.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    # ── Right panel: memory ──
    ax_m.plot(n_extrap, dn_mem_extrap, color="#d62728", linestyle=":",
              linewidth=1.0, alpha=0.7)
    ax_m.plot(ns[dn_valid], dn_mem[dn_valid], color="#d62728",
              linestyle="-", marker="s", markersize=4, linewidth=1.2,
              label="Dense (fastcluster)")
    ax_m.plot(n_extrap, sp_mem_extrap, color="#1f77b4", linestyle=":",
              linewidth=1.0, alpha=0.7)
    ax_m.plot(ns, sp_mem, color="#1f77b4", linestyle="-", marker="o",
              markersize=4, linewidth=1.2, label="GSHAC (constant $\\bar{k}$)")

    for rw in rw_points:
        ax_m.plot(rw["n"], rw["mem"], marker="*", markersize=12,
                  color="#2ca02c", zorder=5)
        ax_m.annotate(rw["label"], (rw["n"], rw["mem"]),
                      textcoords="offset points", xytext=rw["m_ofs"],
                      fontsize=7, color="#2ca02c")

    # RAM ceiling
    ax_m.axhline(RAM_MIB, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_m.text(n_extrap[0] * 1, RAM_MIB * 1.4, "30 GiB RAM",
              fontsize=7, color="grey", ha="left")

    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlabel("$n$")
    ax_m.set_ylabel("Peak memory (MiB)")
    # ax_m.set_title("Memory scalability")
    ax_m.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    # Force layout so transData gives correct angles for inline labels
    fig.tight_layout()
    fig.canvas.draw()

    # Inline labels on extrapolation lines (time)
    inline_label(ax_t, n_extrap, dn_time_extrap, " $O(n^2)$", "#d62728")
    inline_label(ax_t, n_extrap, sp_time_extrap,
                 f" $n^{{{slope_t:.1f}}}$", "#1f77b4")

    # Inline labels on extrapolation lines (memory)
    inline_label(ax_m, n_extrap, dn_mem_extrap, " $O(n^2)$", "#d62728")
    inline_label(ax_m, n_extrap, sp_mem_extrap,
                 f" $n^{{{slope_m:.1f}}}$", "#1f77b4")

    # Label the infeasible zone on both panels
    for ax in (ax_t, ax_m):
        ax.text(n_infeasible * 1.15, 0.02, "Dense exceeds RAM",
                fontsize=7, color="#d62728", ha="left", va="bottom",
                transform=ax.get_xaxis_transform())

    # Single shared legend below both panels
    handles, labels = ax_t.get_legend_handles_labels()
    # Add real-world marker to legend
    handles.append(Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="#2ca02c", markersize=10,
                          label="Real-world benchmarks"))
    labels.append("Real-world benchmarks")
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    # Scenario illustration (no CSV needed)
    make_scenario_figure(f"{FIGURES_DIR}/fig_scenarios.pdf")

    # Read benchmark results
    csv_path = f"{RESULTS_DIR}/benchmark_python.csv"
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found — skipping scaling figures")
        sys.exit(0)

    rows = read_benchmark_csv(csv_path)
    print(f"Read {len(rows)} rows from {csv_path}")

    # Consolidated scaling figure
    make_scaling_figure(rows, f"{FIGURES_DIR}/fig_scaling.pdf")

    # Limits figure
    make_limits_figure(rows, f"{FIGURES_DIR}/fig_limits.pdf")
