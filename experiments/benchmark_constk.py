"""
benchmark_constk.py

Constant-k scaling experiment: domain grows with sqrt(n) so that the
average neighbourhood size k stays approximately constant as n increases.
This simulates real-world scaling where larger datasets cover larger
geographic areas (e.g., national → continental → global inventories),
rather than packing more points into a fixed area.

To validate that the scaling *slopes* are determined by the algorithm
structure rather than by the chosen target neighbourhood size, we sweep
k_target across multiple values. The resulting curves should be near-
parallel on a log-log plot --- same slope, different intercepts ---
confirming that GSHAC scales as O(n k log(n k)) in time and O(n k) in
memory with slopes that are independent of k itself.

Under constant k, GSHAC scales as O(n k log n) ≈ O(n log n) in time and
O(n k) ≈ O(n) in memory, whereas the dense baseline remains O(n²).

Outputs:
  results/benchmark_constk.csv   — one row per (k_target, n) pair
  figures/fig_constk.pdf           — multi-curve figure (slope-invariance)
"""

from __future__ import annotations

import _env  # noqa: F401  (pins thread pools to 1 and the paging state; must precede numpy/scipy/sklearn)

import os
import sys
import time
import tracemalloc
import csv
import gc
import math
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# Shared serif style — matches make_figures.py so all paper figures use one
# font family.
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

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust

_ROOT = os.path.join(os.path.dirname(__file__), "..")
# GSHAC_FAST writes to a SEPARATE dir so a reduced-scale run never overwrites the
# expensive full-scale results in results/.
RESULTS_DIR = os.path.join(_ROOT, "results_fast" if os.environ.get("GSHAC_FAST") == "1" else "results")
FIGURES_DIR = os.path.join(_ROOT, "figures")  # fig_constk.pdf is consumed by the manuscript (maintained separately)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

H_MAX       = 20_000        # 20 km — matches the real-world stars in fig_limits
# Sweep target neighbourhood size to demonstrate slope-invariance.
# k=126 matches the empirical mean k of GeoNames-2M at h_max = 20 km, which is
# the dominant green-star reference in fig_limits; the other values bracket
# it to show that slopes do not depend on the chosen density.
# Set GSHAC_FAST=1 for a reduced-scale reproducibility check (see make reproduce-fast).
FAST = os.environ.get("GSHAC_FAST") == "1"
K_TARGETS   = [50, 126] if FAST else [20, 50, 126, 200]
PRIMARY_K   = 126           # primary k for tab_constk and the fig_limits blue line
METHOD      = "single"
N_REP       = 1 if FAST else 5   # fast track runs once; validate via the full CI (make compare)
N_SIZES     = ([1_000, 5_000, 25_000, 100_000] if FAST
               else [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000])


def domain_for_k(n, h_max, k_target):
    """Domain side length (m) yielding approximately k_target neighbours.

    For uniform random points in a square domain of side L, the expected
    number of neighbours within distance r is k = n * pi * r^2 / L^2.
    Solving for L: L = sqrt(n * pi * r^2 / k).
    """
    return math.sqrt(n * math.pi * h_max**2 / k_target)


def dense_feasible(n, max_gib=6.0):
    return n * (n - 1) / 2 * 8 / 1024 ** 3 <= max_gib


def generate_uniform_points(n_total, domain_m, seed=42):
    """Uniform random points — k depends only on global density."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, domain_m, size=(n_total, 2))


def measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak / (1024 ** 2)


def measure_pipeline(build_fn, run_fn):
    """Graph build + HAC in ONE tracemalloc scope (single build): phase times
    plus the whole-pipeline peak = max(graph-phase, hac-phase incl. resident
    graph). Halves the work vs. building the graph twice per rep."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    graph = build_fn()
    t_graph = time.perf_counter() - t0
    _, peak_g = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    result = run_fn(graph)
    t_hac = time.perf_counter() - t1
    _, peak_h = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return graph, result, t_graph, t_hac, max(peak_g, peak_h) / (1024 ** 2)


# ------------------------------------------------------------------
# Cold-process measurement (one-shot user cost)
# ------------------------------------------------------------------
# Each rep is measured in a fresh subprocess so the dense baseline pays its full
# O(n^2)-matrix allocation/first-touch cost every time --- the cost a user incurs
# running the method ONCE --- rather than the amortised cost of a long-lived process.
# Workers regenerate the identical seeded points, so only scalar results cross the
# process boundary. Mirrors the subprocess isolation used by the GeoNames benchmarks.

def _worker(params_json):
    # ONE measurement per fresh process: either the dense baseline OR the sparse
    # pipeline (never both), so each side pays the full cold-process cost symmetrically.
    p = json.loads(params_json)
    pts = generate_uniform_points(p["n"], p["domain_m"])
    c0 = time.process_time()  # CPU-time evidence: single-threaded => cpu_s ~= wall
    if p["kind"] == "dense":
        _, dt, dm = measure(lambda: dense_hclust(pts, p["cuts"], p["method"]))
        out = {"dense_time": dt, "dense_mem": dm}
    else:  # sparse pipeline (graph + HAC)
        graph, _r, gt, ht, sm = measure_pipeline(
            lambda: spatial_dist_graph(pts, p["h_max"]),
            lambda g: sparse_hclust(g, p["cuts"], p["method"]))
        out = dict(graph_time=gt, hac_time=ht, sparse_mem=sm, n_edges=int(graph["n_edges"]))
    out["cpu_s"] = round(time.process_time() - c0, 4)
    import resource
    out["rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # peak RSS cross-check for tracemalloc
    print("RESULT " + json.dumps(out), flush=True)


def plot_constk(rows, primary_k, h_max_km, fig_path):
    """Render fig_constk from constk result rows. Pure plotting: reads no files
    and runs no benchmark, so it can regenerate the figure from the committed
    CSV. ``rows`` is a list of dicts with numeric ``n``, ``k_target``,
    ``sparse_time_mean``, ``sparse_mem_mean`` and ``dense_time_mean`` /
    ``dense_mem_mean`` ('' for the sparse-only rows beyond dense feasibility)."""
    ks_unique = sorted(set(r["k_target"] for r in rows))
    n_sizes = sorted(set(r["n"] for r in rows))
    colors = cm.viridis(np.linspace(0.15, 0.85, len(ks_unique)))

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # Sparse curves, one per k
    for i, k in enumerate(ks_unique):
        rows_k = sorted([r for r in rows if r["k_target"] == k],
                        key=lambda r: r["n"])
        ns_k = [r["n"] for r in rows_k]
        ts_k = [r["sparse_time_mean"] for r in rows_k]
        ms_k = [r["sparse_mem_mean"]  for r in rows_k]
        lw   = 2.0 if k == primary_k else 1.2
        marker_size = 5 if k == primary_k else 3
        label = f"GSHAC, $\\bar{{k}}$={k}" + (" (primary)" if k == primary_k else "")
        ax_time.plot(ns_k, ts_k, "o-", color=colors[i], linewidth=lw,
                     markersize=marker_size, label=label)
        ax_mem.plot(ns_k, ms_k, "o-", color=colors[i], linewidth=lw,
                    markersize=marker_size, label=label)

    # Dense curve (dense is k-independent; use the first k that carries it)
    dn_rows = sorted([r for r in rows
                       if r["k_target"] == ks_unique[0]
                       and r["dense_time_mean"] != ""],
                      key=lambda r: r["n"])
    if dn_rows:
        dn_ns    = [r["n"] for r in dn_rows]
        dn_times = [r["dense_time_mean"] for r in dn_rows]
        dn_mems  = [r["dense_mem_mean"]  for r in dn_rows]
        ax_time.plot(dn_ns, dn_times, "s--", color="#d62728",
                     linewidth=1.5, markersize=4, label="Dense (fastcluster)")
        ax_mem.plot(dn_ns, dn_mems, "s--", color="#d62728",
                    linewidth=1.5, markersize=4, label="Dense (fastcluster)")

    # Reference slopes anchored to the primary curve
    n_ref = np.array([n_sizes[0], n_sizes[-1]], dtype=float)
    ref_rows = sorted([r for r in rows if r["k_target"] == primary_k],
                      key=lambda r: r["n"])
    t_ref0 = ref_rows[0]["sparse_time_mean"]
    m_ref0 = ref_rows[0]["sparse_mem_mean"]

    ax_time.plot(n_ref, t_ref0 * (n_ref / n_ref[0]) ** 1.1, ":",
                 color="grey", alpha=0.5, label=r"$O(n^{1.1})$ ref.")
    if dn_rows:
        ax_time.plot(n_ref, dn_times[0] * (n_ref / n_ref[0]) ** 2, ":",
                     color="#d62728", alpha=0.3, label=r"$O(n^2)$ ref.")

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("$n$ (dataset size)", fontsize=9)
    ax_time.set_ylabel("Wall time (s)", fontsize=9)
    ax_time.set_title(f"Wall time (constant $\\bar{{k}}$, $h_{{\\max}} = {h_max_km:.0f}$ km)",
                      fontsize=9)
    ax_time.legend(fontsize=6, loc="best")
    ax_time.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    ax_mem.plot(n_ref, m_ref0 * (n_ref / n_ref[0]), ":",
                color="grey", alpha=0.5, label=r"$O(n)$ ref.")
    if dn_rows:
        ax_mem.plot(n_ref, dn_mems[0] * (n_ref / n_ref[0]) ** 2, ":",
                    color="#d62728", alpha=0.3, label=r"$O(n^2)$ ref.")

    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("$n$ (dataset size)", fontsize=9)
    ax_mem.set_ylabel("Peak memory (MiB)", fontsize=9)
    ax_mem.set_title(f"Peak memory (constant $\\bar{{k}}$, $h_{{\\max}} = {h_max_km:.0f}$ km)",
                     fontsize=9)
    ax_mem.legend(fontsize=6, loc="best")
    ax_mem.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def _load_constk_rows(csv_path):
    """Load constk result rows from CSV, coercing only the fields plot_constk
    needs. Dense fields are '' for the sparse-only rows and are kept as ''."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "n":                int(float(r["n"])),
                "k_target":         int(float(r["k_target"])),
                "h_max_km":         float(r["h_max_km"]),
                "sparse_time_mean": float(r["sparse_time_mean"]),
                "sparse_mem_mean":  float(r["sparse_mem_mean"]),
                "dense_time_mean":  float(r["dense_time_mean"]) if r["dense_time_mean"] not in ("", None) else "",
                "dense_mem_mean":   float(r["dense_mem_mean"])  if r["dense_mem_mean"]  not in ("", None) else "",
            })
    return rows


if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    import warnings as _w
    _w.filterwarnings("ignore")
    _worker(sys.argv[2])
    sys.exit(0)


if len(sys.argv) > 1 and sys.argv[1] == "--replot":
    # Regenerate figures/fig_constk.pdf from the committed CSV WITHOUT
    # re-running the benchmark: reads results only, writes the figure only.
    _csv_path = os.path.join(RESULTS_DIR, "benchmark_constk.csv")
    _rows = _load_constk_rows(_csv_path)
    _fig_path = os.path.join(FIGURES_DIR, "fig_constk.pdf")
    plot_constk(_rows, PRIMARY_K, _rows[0]["h_max_km"], _fig_path)
    print(f"Replotted from {_csv_path} (no benchmark run)")
    sys.exit(0)


def measure_cold(params):
    """One dense+sparse measurement in a FRESH subprocess (cold allocator)."""
    cp = subprocess.run([sys.executable, __file__, "--worker", json.dumps(params)],
                        capture_output=True, text=True)
    for line in cp.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"cold worker failed (rc={cp.returncode}): "
                       f"{cp.stderr.strip()[-800:]}")


# ------------------------------------------------------------------
# Run benchmark
# ------------------------------------------------------------------

if __name__ == "__main__":
    out_path = os.path.join(RESULTS_DIR, "benchmark_constk.csv")
    f_out = open(out_path, "w", newline="")
    writer = None
    rows = []

    h_max_km = H_MAX / 1000
    h_cuts = [H_MAX // 2, H_MAX]

    print(f"Constant-k scaling experiment (slope-invariance sweep)")
    print(f"  h_max          = {h_max_km:.0f} km")
    print(f"  k_target sweep = {K_TARGETS}  (primary: k={PRIMARY_K})")
    print(f"  n sizes        = {N_SIZES}")
    print(f"  Dense feasible up to n = {max(n for n in N_SIZES if dense_feasible(n)):,}")
    print()

    for k_target in K_TARGETS:
        print(f"=== k_target = {k_target} =================================")
        for n in N_SIZES:
            domain_m = domain_for_k(n, H_MAX, k_target)
            run_dense = dense_feasible(n)
            tag = "" if run_dense else " [sparse only]"
            print(f"  n = {n:>7,}  domain = {domain_m/1000:>8.1f} km{tag}",
                  end="", flush=True)

            dense_times, dense_mems = [], []
            sparse_times, sparse_mems = [], []
            graph_times, hac_times = [], []
            n_edges_val = mean_k_val = None
            wp = dict(n=n, domain_m=domain_m, h_max=H_MAX, cuts=h_cuts,
                      method=METHOD, run_dense=run_dense)

            # dense and sparse EACH in their own fresh COLD subprocess (symmetric).
            dense_cpus, dense_rsss = [], []
            sparse_cpus, sparse_rsss = [], []
            for rep in range(N_REP):
                if run_dense:
                    d = measure_cold({**wp, "kind": "dense"})
                    dense_times.append(d["dense_time"])
                    dense_mems.append(d["dense_mem"])
                    dense_cpus.append(d["cpu_s"])
                    dense_rsss.append(d["rss_mib"])
                s = measure_cold({**wp, "kind": "sparse"})
                sparse_cpus.append(s["cpu_s"])
                sparse_rsss.append(s["rss_mib"])
                gt, ht, sm = s["graph_time"], s["hac_time"], s["sparse_mem"]
                graph_times.append(gt)
                hac_times.append(ht)
                sparse_times.append(gt + ht)
                sparse_mems.append(sm)
                if rep == 0:
                    n_edges_val = s["n_edges"]
                    mean_k_val = round(2 * n_edges_val / n, 1)

            row = {
                "n":                n,
                "domain_km":        round(domain_m / 1000, 1),
                "h_max_km":         h_max_km,
                "k_target":         k_target,
                "mean_k":           mean_k_val,
                "n_edges":          n_edges_val,
                "dense_time_mean":  round(np.mean(dense_times), 4) if run_dense else "",
                "dense_time_sd":    round(np.std(dense_times, ddof=1), 4) if run_dense else "",
                "dense_mem_mean":   round(np.mean(dense_mems), 2) if run_dense else "",
                "dense_mem_sd":     round(np.std(dense_mems, ddof=1), 2) if run_dense else "",
                "sparse_time_mean": round(np.mean(sparse_times), 4),
                "sparse_time_sd":   round(np.std(sparse_times, ddof=1), 4),
                "sparse_mem_mean":  round(np.mean(sparse_mems), 2),
                "sparse_mem_sd":    round(np.std(sparse_mems, ddof=1), 2),
                "graph_time_mean":  round(np.mean(graph_times), 4),
                "hac_time_mean":    round(np.mean(hac_times), 4),
                "speedup_time":     round(np.mean(dense_times) / np.mean(sparse_times), 2) if run_dense else "",
                "speedup_mem":      round(np.mean(dense_mems) / np.mean(sparse_mems), 2) if run_dense else "",
                # Single-threadedness evidence and peak-RSS cross-check.
                "dense_cpu_mean":   round(np.mean(dense_cpus), 4) if run_dense else "",
                "dense_rss_mean":   round(np.mean(dense_rsss), 1) if run_dense else "",
                "sparse_cpu_mean":  round(np.mean(sparse_cpus), 4),
                "sparse_rss_mean":  round(np.mean(sparse_rsss), 1),
            }
            rows.append(row)

            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f_out.flush()

            if run_dense:
                print(f"  k={mean_k_val:>5}  dense={row['dense_time_mean']:.3f}s/{row['dense_mem_mean']:.0f}MiB"
                      f"  sparse={row['sparse_time_mean']:.3f}s/{row['sparse_mem_mean']:.0f}MiB"
                      f"  speedup={row['speedup_time']}x time, {row['speedup_mem']}x mem")
            else:
                print(f"  k={mean_k_val:>5}  sparse={row['sparse_time_mean']:.3f}s/{row['sparse_mem_mean']:.0f}MiB")

            gc.collect()
        print()

    f_out.close()
    print(f"\nResults written to {out_path}")

    # ------------------------------------------------------------------
    # Fitted log-log slopes per k_target (committed so the manuscript's
    # slope-invariance numbers are reproducible, not hand-computed)
    # ------------------------------------------------------------------
    slope_rows = []
    for k in K_TARGETS:
        rk = sorted([r for r in rows if r["k_target"] == k], key=lambda r: r["n"])
        nk = np.array([r["n"] for r in rk], dtype=float)
        tk = np.array([r["sparse_time_mean"] for r in rk], dtype=float)
        mk = np.array([r["sparse_mem_mean"] for r in rk], dtype=float)
        ok = tk > 0
        slope_t = float(np.polyfit(np.log(nk[ok]), np.log(tk[ok]), 1)[0])
        slope_m = float(np.polyfit(np.log(nk[ok]), np.log(mk[ok]), 1)[0])
        slope_rows.append({
            "k_target":   k,
            "is_primary": int(k == PRIMARY_K),
            "slope_time": round(slope_t, 3),
            "slope_mem":  round(slope_m, 3),
            "n_min":      int(nk.min()),
            "n_max":      int(nk.max()),
        })
    slopes_out = os.path.join(RESULTS_DIR, "benchmark_constk_slopes.csv")
    with open(slopes_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=slope_rows[0].keys())
        w.writeheader()
        w.writerows(slope_rows)
    print(f"Slopes written to {slopes_out}")
    for s in slope_rows:
        print(f"  k={s['k_target']:>3}: time n^{s['slope_time']}, mem n^{s['slope_mem']}"
              + ("  (primary)" if s["is_primary"] else ""))

    # ------------------------------------------------------------------
    # Figure: multi-k slope invariance
    # ------------------------------------------------------------------

    # FAST mode uses reduced data — never overwrite the tracked paper figure with it;
    # write a clearly-named copy under results/ instead.
    fig_path = (os.path.join(RESULTS_DIR, "fig_constk_fast.pdf") if FAST
                else os.path.join(FIGURES_DIR, "fig_constk.pdf"))
    plot_constk(rows, PRIMARY_K, h_max_km, fig_path)
