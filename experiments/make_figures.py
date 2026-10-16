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
    """Three-panel illustration of the three benchmark density scenarios.

    Uses the *same* scenario-defining parameters as ``benchmark.py``
    (``n_centres`` = 100/50/20, ``spread_factor`` = 0.03/0.10/0.30 × h_max)
    on the benchmark's 500 km domain, rendered at a single representative
    (n, h_max) drawn from the benchmark sweep so the panels are a genuine
    small instance of the benchmarked scenarios. The scenario labels carry
    no printed k-bar/density values: the mean neighbourhood size varies with
    n and h_max across the sweep, so a single illustrative value would
    misrepresent the benchmark (the benchmarked k-bar range is reported in
    the text and tables). The point of the figure is the qualitative
    density progression tight -> moderate -> loose.
    """
    from matplotlib.collections import LineCollection

    # Scenario-defining parameters identical to benchmark.py.
    SCENARIOS = [
        ("Tight",    dict(n_centres=100, spread_factor=0.03)),
        ("Moderate", dict(n_centres=50,  spread_factor=0.10)),
        ("Loose",    dict(n_centres=20,  spread_factor=0.30)),
    ]

    # A single representative (n, h_max) from the benchmark sweep, on the
    # benchmark's 500 km domain.
    n = 5_000
    h_max = 50_000       # metres (one of the benchmark's h_max values)
    domain_m = 500_000   # 500 km — the benchmark domain

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    scenario_stats = []
    for ax, (label, cfg) in zip(axes, SCENARIOS):
        spread_m = cfg["spread_factor"] * h_max
        pts = generate_clustered_points(
            n, cfg["n_centres"], spread_m, domain_m=domain_m, seed=42)

        # Build distance graph
        tree = cKDTree(pts)
        pairs = tree.query_pairs(h_max, output_type="ndarray")

        mean_k = 2 * len(pairs) / n
        graph_pct = 2 * len(pairs) / (n * (n - 1)) * 100
        scenario_stats.append({
            "scenario":      label,
            "n":             n,
            "h_max_km":      h_max / 1000,
            "domain_km":     domain_m / 1000,
            "n_centres":     cfg["n_centres"],
            "spread_factor": cfg["spread_factor"],
            "n_edges":       int(len(pairs)),
            "mean_k":        round(mean_k, 1),
            "density_pct":   round(graph_pct, 1),
        })

        # Convert to km
        pts_km = pts / 1000

        # Draw edges
        if len(pairs) > 0:
            segs = np.stack([pts_km[pairs[:, 0]],
                             pts_km[pairs[:, 1]]], axis=1)
            lc = LineCollection(segs, colors="#b0b0b0", linewidths=0.15,
                                alpha=0.3, zorder=1, rasterized=True)
            ax.add_collection(lc)

        # Draw points (rasterized: dense scatter + edges would bloat the
        # vector PDF; 300 dpi keeps them crisp in print while text/axes
        # stay vector).
        ax.scatter(pts_km[:, 0], pts_km[:, 1], s=4, color="black",
                   alpha=0.8, zorder=2, linewidths=0, rasterized=True)

        # Draw h_max scale bar in upper-right corner
        bar_x1 = domain_m / 1000 - 0.5
        bar_x0 = bar_x1 - h_max / 1000
        bar_y = domain_m / 1000 + 0.8
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#d62728",
                linewidth=2.0, solid_capstyle="butt", clip_on=False)
        ax.text((bar_x0 + bar_x1) / 2, bar_y + 0.5,
                r"$h_{max}$", fontsize=7, ha="center", color="#d62728")

        # Scenario label only — no printed k-bar/density (see docstring).
        ax.set_title(label, fontsize=10, pad=12)

        ax.set_aspect("equal")
        ax.set_xlim(-0.5, domain_m / 1000 + 0.5)
        ax.set_ylim(-0.5, domain_m / 1000 + 0.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("km", fontsize=8)

    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.10, top=0.82)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Record the illustration's k-bar/density for provenance only (NOT shown
    # on the figure or quoted in the caption; benchmarked values are in the
    # results tables/prose).
    _stats_csv = os.path.join(RESULTS_DIR, "scenario_illustration.csv")
    with open(_stats_csv, "w", newline="") as _f:
        _w = csv.DictWriter(_f, fieldnames=scenario_stats[0].keys())
        _w.writeheader()
        _w.writerows(scenario_stats)
    print(f"Saved {_stats_csv}")


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

# h_max threshold (metres) used for the GeoNames green-star reference point
# in fig_limits.pdf. Change here to switch to a different threshold from
# benchmark_geonames.csv without touching the figure code.
GEONAMES_STAR_HMAX_M = 20000

# Primary k_target for the constant-k blue line in fig_limits. The CSV from
# benchmark_constk.py contains a sweep over multiple k_target values; this
# selects which one becomes the headline curve. k=126 matches the empirical
# mean k of GeoNames-2M at h_max = 20 km (the dominant green star).
CONSTK_PRIMARY_K = 126


def _load_rw_points():
    """Read the GeoNames real-world reference point from benchmark_geonames.csv.

    Returns a list of dicts compatible with the green-star plotting loops in
    make_limits_figure (label, n, time, mem, and offset tuples for the inline
    annotations). The row matching GEONAMES_STAR_HMAX_M is used; a missing CSV
    or threshold is silently skipped so the figure can still render.
    """
    points = []

    def _label(n):
        if n >= 1_000_000:
            return f"$n$={n // 1_000_000}M"
        if n >= 1000:
            return f"$n$={n // 1000}K"
        return f"$n$={n}"

    # GeoNames provides the sole real-world reference star.
    geo_path = os.path.join(RESULTS_DIR, "benchmark_geonames.csv")
    if os.path.exists(geo_path):
        for r in read_benchmark_csv(geo_path):
            if int(float(r["h_max_m"])) == GEONAMES_STAR_HMAX_M:
                n = int(r["n_sample"])
                points.append({
                    "label": _label(n),
                    "n": n,
                    "time": float(r["total_time_s"]),
                    "mem": (float(r["pipeline_mem_mib"]) if "pipeline_mem_mib" in r
                            else max(float(r["graph_mem_mib"]), float(r["hac_mem_mib"]))),
                    "t_ofs": (1, 6),
                    "m_ofs": (1, 6),
                })
                break

    return points


def make_limits_figure(rows, out_path):
    """
    Two-panel figure:
      Left:  time vs n  (measured sparse + extrapolated, dense infeasible zone)
      Right: memory vs n (same structure)

    Shows where dense becomes infeasible and where sparse remains practical.
    Adds annotation about parallel processing potential.
    """

    # Use constant-k benchmark data if available (real-world scaling regime).
    # The CSV may contain multiple k_target values from the slope-invariance
    # sweep; pick the primary one for the figure's blue line.
    constk_path = os.path.join(RESULTS_DIR, "benchmark_constk.csv")
    if os.path.exists(constk_path):
        ck_all = read_benchmark_csv(constk_path)
        ck_primary = [r for r in ck_all
                      if int(float(r.get("k_target", 0) or 0)) == CONSTK_PRIMARY_K]
        ck_rows = sorted(ck_primary if ck_primary else ck_all,
                         key=lambda r: r["n"])
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

    # Commit the fitted scaling exponents shown as the n^slope inline labels.
    _slopes_csv = os.path.join(RESULTS_DIR, "scaling_slopes.csv")
    with open(_slopes_csv, "w", newline="") as _f:
        _w = csv.writer(_f)
        _w.writerow(["source", "k_target", "slope_time", "slope_mem",
                     "n_min", "n_max"])
        _w.writerow(["constk_primary", CONSTK_PRIMARY_K,
                     round(float(slope_t), 3), round(float(slope_m), 3),
                     int(ns.min()), int(ns.max())])
    print(f"Saved {_slopes_csv}")

    n_max = 7.5  # log10 upper limit for extrapolation lines
    n_extrap = np.logspace(np.log10(ns[0]), n_max, 100)

    sp_time_extrap = np.exp(intercept_t) * n_extrap ** slope_t
    sp_mem_extrap  = np.exp(intercept_m) * n_extrap ** slope_m

    # Dense O(n^2) PROJECTION (not measured): the dense baseline is never run past ~50K,
    # so anchor to the last measured dense point (n_ref, t_ref/m_ref) and scale as n^2.
    # These *_extrap arrays feed only the DOTTED, O(n^2)-labeled projection lines below;
    # they are never plotted as or mixed with measured markers.
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

    # Real-world reference points (loaded from benchmark CSVs)
    rw_points = _load_rw_points()

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


def make_geoscaling_figure(out_path):
    """Two GeoNames scaling regimes on the same 2M sample at h_max = 10 km:
    geographic windows at constant density (near-linear, the realistic regime)
    vs fixed-extent subsampling at growing density (~quadratic memory). The two
    curves converge at n = 2M, where both equal the full sample."""
    import csv as _csv

    def _read(name):
        p = os.path.join(RESULTS_DIR, name)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return list(_csv.DictReader(f))

    geo = _read("benchmark_geonames_geographic.csv")
    fix = _read("benchmark_geonames_scaling.csv")
    if not geo or not fix:
        print("WARNING: geoscaling CSVs missing -- skipping fig_geoscaling")
        return

    def col(rows, k, scale=1.0):
        return np.array([float(r[k]) for r in rows]) * scale

    gn, gt, gm = col(geo, "n"), col(geo, "total_time_s"), col(geo, "pipeline_mem_mib", 1 / 1024)
    fn, ft, fm = col(fix, "n"), col(fix, "total_time_s"), col(fix, "pipeline_mem_mib", 1 / 1024)

    def slope(x, y):
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.16, 3.2))
    GEO_C, FIX_C = "#1f77b4", "#d62728"
    for ax, gy, fy, ylab in ((ax_t, gt, ft, "Wall time (s)"),
                             (ax_m, gm, fm, "Peak memory (GiB)")):
        ax.plot(gn, gy, "-o", color=GEO_C, lw=2.2, ms=5, zorder=3,
                label=f"Constant density ($n^{{{slope(gn, gy):.2f}}}$)")
        ax.plot(fn, fy, "--s", color=FIX_C, lw=1.3, ms=4, alpha=0.85, zorder=2,
                label=f"Growing density ($n^{{{slope(fn, fy):.2f}}}$)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$n$ (points)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=7.5, loc="upper left")
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

    # GeoNames scaling regimes (constant vs growing density)
    make_geoscaling_figure(f"{FIGURES_DIR}/fig_geoscaling.pdf")
