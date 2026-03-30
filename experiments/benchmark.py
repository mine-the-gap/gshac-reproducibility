"""
benchmark.py

Comprehensive benchmark comparing:
  (A) Dense baseline: full O(n^2) distance matrix -> fastcluster linkage
  (B) Sparse graph:   geographic pruning -> sparse matrix -> component-wise HAC

Three density scenarios vary the spread of synthetic clusters relative to
h_max, controlling the mean neighbourhood size k and component size:

  tight    – spread_m = 0.03 * h_max, n_centres = 100
             Very compact, well-separated clusters (best case for sparse).
  moderate – spread_m = 0.10 * h_max, n_centres = 50
             Intermediate clustering.
  loose    – spread_m = 0.30 * h_max, n_centres = 20
             Broad, overlapping clusters (worst case for sparse).

Measures wall time and peak memory across varying n and h_max.
Reports mean ± sd over N_REP repetitions.

Outputs (relative to repo root, i.e. one level up from this script):
  results/benchmark_python.csv          — summary (mean ± sd per config)
  results/benchmark_python_reps.csv     — per-repetition raw data
  results/benchmark_sklearn_comparison.csv
  results/benchmark_sklearn_timing.csv
  results/benchmark_pysal.csv
  figures/fig_time_python.pdf
  figures/fig_memory_python.pdf
  figures/fig_speedup_python.pdf
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import csv
import gc
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust

# Paths relative to repo root (one level up from experiments/)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Density scenarios
# ------------------------------------------------------------------

SCENARIOS = {
    "tight":    dict(n_centres=100, spread_factor=0.03),
    "moderate": dict(n_centres=50,  spread_factor=0.10),
    "loose":    dict(n_centres=20,  spread_factor=0.30),
}

# ------------------------------------------------------------------
# Synthetic data generator
# ------------------------------------------------------------------

def generate_clustered_points(
    n_total: int,
    n_centres: int,
    spread_m: float,
    domain_m: float = 500_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns (n_total, 2) array of (x, y) coordinates in metres.
    Points are distributed around n_centres Gaussian clusters with
    standard deviation spread_m within a domain of domain_m x domain_m.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, domain_m, size=(n_centres, 2))
    assignments = rng.integers(0, n_centres, size=n_total)
    noise = rng.normal(0, spread_m, size=(n_total, 2))
    return centres[assignments] + noise


# ------------------------------------------------------------------
# Peak memory measurement helper
# ------------------------------------------------------------------

def measure(fn):
    """Run fn(), return (result, elapsed_seconds, peak_memory_mib)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak / (1024 ** 2)


def dense_feasible(n, max_gb=6.0):
    """Check if the condensed distance matrix fits in memory.

    max_gb covers the condensed vector only; fastcluster needs additional
    working memory, so keep well below physical RAM (30 GiB).
    """
    condensed_gb = n * (n - 1) / 2 * 8 / 1e9
    return condensed_gb <= max_gb


# ------------------------------------------------------------------
# Benchmark parameters
# ------------------------------------------------------------------

N_SIZES    = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
H_MAX_VALS = [10_000, 20_000, 50_000]   # metres
N_REP      = 5
METHOD     = "single"
H_CUTS     = list(range(5_000, 55_000, 5_000))  # 5 km to 50 km in 5 km steps

n_configs = len(SCENARIOS) * len(N_SIZES) * len(H_MAX_VALS)
print("Starting Python benchmarks...")
print(f"Scenarios: {list(SCENARIOS.keys())}")
print(f"Configurations: {n_configs}  (x{N_REP} reps each)")
print(f"Dense feasible up to n={max(n for n in N_SIZES if dense_feasible(n)):,}\n")

rows = []       # summary (mean ± sd)
rep_rows = []   # per-repetition raw data

# Incremental CSV writers — flush after each config so OOM doesn't lose data
_summary_path = f"{RESULTS_DIR}/benchmark_python.csv"
_reps_path    = f"{RESULTS_DIR}/benchmark_python_reps.csv"
_summary_file = open(_summary_path, "w", newline="")
_reps_file    = open(_reps_path, "w", newline="")
_summary_writer = None
_reps_writer    = None
_reps_written   = 0

for scenario_name, scenario_cfg in SCENARIOS.items():
    n_centres    = scenario_cfg["n_centres"]
    spread_factor = scenario_cfg["spread_factor"]

    print(f"\n=== Scenario: {scenario_name} "
          f"(n_centres={n_centres}, spread={spread_factor}*h_max) ===")

    for h_max in H_MAX_VALS:
        h_max_km  = h_max / 1000
        spread_m  = spread_factor * h_max

        for n in N_SIZES:
            run_dense = dense_feasible(n)
            tag = "" if run_dense else " [sparse only]"
            print(f"  n = {n:,}, h_max = {h_max_km:.0f} km{tag}")

            pts = generate_clustered_points(
                n_total=n,
                n_centres=n_centres,
                spread_m=spread_m,
            )

            dense_times, dense_mems = [], []
            graph_times, hac_times  = [], []
            sparse_times, sparse_mems = [], []
            n_edges_val = density_val = None
            dt = dm = None

            for rep in range(N_REP):
                # Dense baseline (skip if infeasible)
                if run_dense:
                    _, dt, dm = measure(lambda: dense_hclust(pts, H_CUTS, METHOD))
                    dense_times.append(dt)
                    dense_mems.append(dm)
                    gc.collect()

                # Sparse — time each phase separately
                graph, gt, _ = measure(lambda: spatial_dist_graph(pts, h_max))
                _,     ht, _ = measure(lambda: sparse_hclust(graph, H_CUTS, METHOD))
                graph_times.append(gt)
                hac_times.append(ht)
                sparse_times.append(gt + ht)

                if rep == 0:
                    n_edges_val = graph["n_edges"]
                    density_val = graph["density"]

                # Free graph before measuring memory (avoid two graphs in RAM)
                del graph
                gc.collect()

                # Peak memory measured over the complete pipeline
                def run_sparse_mem():
                    g2 = spatial_dist_graph(pts, h_max)
                    res = sparse_hclust(g2, H_CUTS, METHOD)
                    return res  # don't return g2 — let it be freed

                _, _, sm = measure(run_sparse_mem)
                sparse_mems.append(sm)
                gc.collect()

                # Save per-rep data
                rep_rows.append({
                    "scenario":       scenario_name,
                    "n":              n,
                    "h_max_km":       h_max_km,
                    "rep":            rep,
                    "dense_time_s":   round(dt, 4) if (run_dense and dt is not None) else "",
                    "dense_mem_mib":  round(dm, 2) if (run_dense and dm is not None) else "",
                    "sparse_time_s":  round(gt + ht, 4),
                    "sparse_mem_mib": round(sm, 2),
                    "graph_time_s":   round(gt, 4),
                    "hac_time_s":     round(ht, 4),
                })

            mean_k = round(2 * n_edges_val / n, 1) if n_edges_val else None

            mean_graph  = np.mean(graph_times)
            sd_graph    = np.std(graph_times, ddof=1)
            mean_hac    = np.mean(hac_times)
            sd_hac      = np.std(hac_times, ddof=1)
            mean_sparse = np.mean(sparse_times)
            sd_sparse   = np.std(sparse_times, ddof=1)
            mean_sparse_mem = np.mean(sparse_mems)
            sd_sparse_mem   = np.std(sparse_mems, ddof=1)

            row = {
                "scenario":        scenario_name,
                "n":               n,
                "h_max_km":        h_max_km,
                "method":          METHOD,
                "n_centres":       n_centres,
                "spread_factor":   spread_factor,
                "n_edges":         n_edges_val,
                "mean_k":          mean_k,
                "graph_density":   round(density_val, 6) if density_val else None,
                # Dense — mean ± sd
                "dense_time_mean": round(np.mean(dense_times), 4) if run_dense else "",
                "dense_time_sd":   round(np.std(dense_times, ddof=1), 4) if run_dense else "",
                "dense_mem_mean":  round(np.mean(dense_mems), 2) if run_dense else "",
                "dense_mem_sd":    round(np.std(dense_mems, ddof=1), 2) if run_dense else "",
                # Sparse — total (mean ± sd)
                "sparse_time_mean": round(mean_sparse, 4),
                "sparse_time_sd":   round(sd_sparse, 4),
                "sparse_mem_mean":  round(mean_sparse_mem, 2),
                "sparse_mem_sd":    round(sd_sparse_mem, 2),
                # Sparse — phases (mean ± sd)
                "graph_time_mean":  round(mean_graph, 4),
                "graph_time_sd":    round(sd_graph, 4),
                "hac_time_mean":    round(mean_hac, 4),
                "hac_time_sd":      round(sd_hac, 4),
                # Derived
                "graph_time_frac":  round(mean_graph / mean_sparse, 3) if mean_sparse > 0 else None,
                "speedup_time":     round(np.mean(dense_times) / mean_sparse, 2) if run_dense else "",
                "speedup_mem":      round(np.mean(dense_mems) / mean_sparse_mem, 2) if run_dense else "",
            }
            rows.append(row)

            # Incremental CSV write — survives OOM
            if _summary_writer is None:
                _summary_writer = csv.DictWriter(_summary_file, fieldnames=row.keys())
                _summary_writer.writeheader()
            _summary_writer.writerow(row)
            _summary_file.flush()

            if _reps_writer is None and rep_rows:
                _reps_writer = csv.DictWriter(_reps_file, fieldnames=rep_rows[0].keys())
                _reps_writer.writeheader()
            if _reps_writer:
                for rr in rep_rows[_reps_written:]:
                    _reps_writer.writerow(rr)
                _reps_written = len(rep_rows)
                _reps_file.flush()

            # Free memory before next config
            del pts
            gc.collect()

            if run_dense:
                print(f"    dense: {row['dense_time_mean']:.3f} ± {row['dense_time_sd']:.3f} s, "
                      f"{row['dense_mem_mean']:.0f} MiB")
            print(f"    sparse: {row['sparse_time_mean']:.3f} ± {row['sparse_time_sd']:.3f} s, "
                  f"{row['sparse_mem_mean']:.0f} MiB  "
                  f"(graph {row['graph_time_mean']:.3f} + HAC {row['hac_time_mean']:.3f})")
            if run_dense:
                print(f"    speedup: {row['speedup_time']}× time, "
                      f"{row['speedup_mem']}× memory")

# ------------------------------------------------------------------
# Close incremental CSV files
# ------------------------------------------------------------------

_summary_file.close()
_reps_file.close()

print(f"\nSummary results written to {_summary_path}")
print(f"Per-rep results written to {_reps_path}")

# ------------------------------------------------------------------
# scikit-learn connectivity comparison
# ------------------------------------------------------------------

print("\n--- scikit-learn connectivity comparison ---")

from sklearn.cluster import AgglomerativeClustering
from gshac.spatial_dist_graph import geographic_connectivity

sklearn_rows = []
SKLEARN_LINKAGES = ["single", "complete", "average"]
SK_SCENARIO   = "moderate"
SK_N_SIZES    = [500, 1_000, 2_500]
SK_H_MAX      = 10_000
SK_H_CUT      = 5_000
sk_cfg = SCENARIOS[SK_SCENARIO]
sk_spread = sk_cfg["spread_factor"] * SK_H_MAX

for n in SK_N_SIZES:
    pts = generate_clustered_points(n, sk_cfg["n_centres"], sk_spread)

    # Ground truth from dense baseline
    dn = dense_hclust(pts, [SK_H_CUT], METHOD)
    n_dense = int(np.unique(dn["labels"][SK_H_CUT]).size)

    # Our sparse implementation
    g_sk   = spatial_dist_graph(pts, SK_H_MAX)
    sp_res = sparse_hclust(g_sk, [SK_H_CUT], method="single")
    n_sparse = int(np.unique(sp_res["labels"][SK_H_CUT]).size)

    # Build binary connectivity matrix once
    conn = geographic_connectivity(pts, SK_H_MAX)

    for lnk in SKLEARN_LINKAGES:
        try:
            sk_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=SK_H_CUT,
                connectivity=conn,
                linkage=lnk,
            )
            sk_model.fit(pts)
            n_sklearn = int(sk_model.n_clusters_)
            sklearn_correct = (n_sklearn == n_dense)
        except Exception as exc:
            n_sklearn = None
            sklearn_correct = None
            print(f"  sklearn {lnk} n={n}: ERROR {exc}")

        row = {
            "scenario":       SK_SCENARIO,
            "n":              n,
            "h_max_km":       SK_H_MAX / 1000,
            "h_cut_km":       SK_H_CUT / 1000,
            "linkage":        lnk,
            "n_dense":        n_dense,
            "n_sparse_ours":  n_sparse,
            "n_sklearn_conn": n_sklearn,
            "sklearn_correct": sklearn_correct,
            "our_correct":    (n_sparse == n_dense),
        }
        sklearn_rows.append(row)
        print(f"  n={n:,} linkage={lnk:<8} "
              f"dense={n_dense:4d}  sparse_ours={n_sparse:4d}  "
              f"sklearn_conn={str(n_sklearn):>4}  "
              f"sklearn_ok={sklearn_correct}  our_ok={n_sparse==n_dense}")

sk_out = f"{RESULTS_DIR}/benchmark_sklearn_comparison.csv"
with open(sk_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sklearn_rows[0].keys())
    writer.writeheader()
    writer.writerows(sklearn_rows)
print(f"\nsklearn comparison written to {sk_out}")

# ------------------------------------------------------------------
# scikit-learn connectivity timing comparison (single linkage)
# ------------------------------------------------------------------

print("\n--- scikit-learn connectivity timing comparison ---")

SK_TIMING_N = [500, 1_000, 2_500, 5_000, 10_000, 25_000]
sk_timing_rows = []

for n in SK_TIMING_N:
    pts = generate_clustered_points(n, sk_cfg["n_centres"], sk_spread)

    our_times, sk_times = [], []
    n_clusters_ours = n_clusters_sk = None

    for rep in range(N_REP):
        # Our approach
        g_t = spatial_dist_graph(pts, SK_H_MAX)
        _, ht, _ = measure(lambda: sparse_hclust(g_t, [SK_H_CUT], method="single"))
        our_times.append(ht)

        # sklearn connectivity HAC
        conn = geographic_connectivity(pts, SK_H_MAX)
        def run_sklearn():
            m = AgglomerativeClustering(
                n_clusters=None, distance_threshold=SK_H_CUT,
                connectivity=conn, linkage="single",
            )
            m.fit(pts)
            return m
        sk_model, st, _ = measure(run_sklearn)
        sk_times.append(st)

        if rep == 0:
            r_ours = sparse_hclust(g_t, [SK_H_CUT], method="single")
            n_clusters_ours = int(np.unique(r_ours["labels"][SK_H_CUT]).size)
            n_clusters_sk = int(sk_model.n_clusters_)

    our_mean = np.mean(our_times)
    sk_mean  = np.mean(sk_times)
    speedup  = round(sk_mean / our_mean, 2) if our_mean > 0 else None

    sk_timing_rows.append({
        "scenario":           SK_SCENARIO,
        "n":                  n,
        "h_max_km":           SK_H_MAX / 1000,
        "h_cut_km":           SK_H_CUT / 1000,
        "linkage":            "single",
        "our_time_mean":      round(our_mean, 4),
        "our_time_sd":        round(np.std(our_times, ddof=1), 4),
        "sklearn_conn_time_mean": round(sk_mean, 4),
        "sklearn_conn_time_sd":   round(np.std(sk_times, ddof=1), 4),
        "speedup":            speedup,
        "n_clusters_ours":    n_clusters_ours,
        "n_clusters_sklearn": n_clusters_sk,
        "results_agree":      (n_clusters_ours == n_clusters_sk),
    })
    print(f"  n={n:>6,}  ours={our_mean:.4f}±{np.std(our_times,ddof=1):.4f}s  "
          f"sklearn={sk_mean:.4f}±{np.std(sk_times,ddof=1):.4f}s  "
          f"speedup={speedup}x  agree={n_clusters_ours == n_clusters_sk}")

skt_out = f"{RESULTS_DIR}/benchmark_sklearn_timing.csv"
with open(skt_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sk_timing_rows[0].keys())
    writer.writeheader()
    writer.writerows(sk_timing_rows)
print(f"sklearn timing written to {skt_out}")


# ------------------------------------------------------------------
# Print speedup summary
# ------------------------------------------------------------------

print("\nSpeedup summary (dense_time / sparse_time):")
print(f"{'Scenario':<12} {'n':>7} {'h_max':>8} {'k':>6} "
      f"{'Time':>12} {'Mem':>12} {'graph%':>7}")
print("-" * 72)
for r in rows:
    spd = f"{r['speedup_time']}x" if r['speedup_time'] != "" else "n/a"
    mem = f"{r['speedup_mem']}x" if r['speedup_mem'] != "" else "n/a"
    print(f"{r['scenario']:<12} {r['n']:>7,} {r['h_max_km']:>7.0f}km "
          f"{r['mean_k']:>6.1f} {spd:>12} "
          f"{mem:>12} "
          f"{r['graph_time_frac']*100:>6.0f}%")

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

SCENARIO_STYLES = {
    "tight":    dict(color="#2CA02C", linestyle="-",  marker="^"),
    "moderate": dict(color="#FF7F0E", linestyle="--", marker="s"),
    "loose":    dict(color="#1F77B4", linestyle=":",  marker="o"),
}

h_maxes_km = sorted(set(r["h_max_km"] for r in rows))


def make_time_memory_figure(metric_dense, metric_sparse, ylabel, title, out_path):
    """One row of subplots (one per h_max), each showing all scenarios."""
    fig, axes = plt.subplots(1, len(h_maxes_km), figsize=(10, 3.5), sharey=True)
    for ax, h in zip(axes, h_maxes_km):
        for scen, sty in SCENARIO_STYLES.items():
            sub = [r for r in rows if r["h_max_km"] == h and r["scenario"] == scen]
            sub.sort(key=lambda r: r["n"])
            xs = [r["n"] for r in sub]
            # Sparse: always available
            ys = [r[metric_sparse] for r in sub]
            ax.plot(xs, ys, color=sty["color"], linestyle="--",
                    marker=sty["marker"], linewidth=1.2, alpha=0.6,
                    label=f"Sparse ({scen})")
            # Dense: only where available
            xs_d = [r["n"] for r in sub if r[metric_dense] != ""]
            yd   = [r[metric_dense] for r in sub if r[metric_dense] != ""]
            if xs_d:
                ax.plot(xs_d, yd, color=sty["color"], linestyle="-",
                        marker=sty["marker"], linewidth=1.2,
                        label=f"Dense ({scen})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n", fontsize=9)
        ax.set_title(f"h_max = {h:.0f} km", fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    axes[0].set_ylabel(ylabel, fontsize=9)
    handles, labs = axes[0].get_legend_handles_labels()
    seen = {}
    for h_, l_ in zip(handles, labs):
        if l_ not in seen:
            seen[l_] = h_
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_speedup_figure(out_path):
    fig, axes = plt.subplots(1, len(h_maxes_km), figsize=(10, 3.5), sharey=True)
    for ax, h in zip(axes, h_maxes_km):
        for scen, sty in SCENARIO_STYLES.items():
            sub = [r for r in rows
                   if r["h_max_km"] == h and r["scenario"] == scen
                   and r["speedup_time"] != ""]
            sub.sort(key=lambda r: r["n"])
            if not sub:
                continue
            xs = [r["n"] for r in sub]
            ys = [r["speedup_time"] for r in sub]
            ax.plot(xs, ys, label=scen, **{k: v for k, v in sty.items()},
                    linewidth=1.2)
        ax.axhline(1, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("n", fontsize=9)
        ax.set_title(f"h_max = {h:.0f} km", fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    axes[0].set_ylabel("Speedup (dense / sparse)", fontsize=9)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Time speedup of sparse graph approach — Python", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


make_time_memory_figure(
    "dense_time_mean", "sparse_time_mean",
    "Wall time (s)",
    "Wall time: dense vs. sparse geographic graph (Python)",
    f"{FIGURES_DIR}/fig_time_python.pdf",
)
make_time_memory_figure(
    "dense_mem_mean", "sparse_mem_mean",
    "Peak memory (MiB)",
    "Peak memory: dense vs. sparse geographic graph (Python)",
    f"{FIGURES_DIR}/fig_memory_python.pdf",
)
make_speedup_figure(f"{FIGURES_DIR}/fig_speedup_python.pdf")

# ------------------------------------------------------------------
# Real-world extrapolation (global mining dataset, n=261,073)
# ------------------------------------------------------------------

print("\n--- Real-world extrapolation (global mining dataset) ---")

N_RW      = 261_073
K_MEAN_RW = 57.8
H_MAX_RW  = 20_000   # metres

m_rw = int(N_RW * K_MEAN_RW / 2)
sparse_matrix_mb = m_rw * 2 * 8 / 1e6     # upper + lower triangle, float64
dense_matrix_gb  = N_RW ** 2 * 8 / 1e9    # full n×n, float64

print(f"n = {N_RW:,}")
print(f"Estimated edges m ~ {m_rw:,}  (mean k = {K_MEAN_RW})")
print(f"Sparse matrix memory:  ~{sparse_matrix_mb:.0f} MB")
print(f"Dense matrix memory:   ~{dense_matrix_gb:.0f} GB  (infeasible on workstations)")

ref = [r for r in rows
       if r["scenario"] == "tight" and r["h_max_km"] == 20.0 and r["n"] == 10_000]
if ref:
    t_ref = ref[0]["sparse_time_mean"]
    n_ref = 10_000
    cost_ref = n_ref * np.log(n_ref) + n_ref * ref[0]["mean_k"]
    cost_rw  = N_RW  * np.log(N_RW)  + N_RW  * K_MEAN_RW
    t_index_query = t_ref * (cost_rw / cost_ref)
    print(f"\nTime estimate (sparse, lower bound — index+query only):")
    print(f"  ~{t_index_query/60:.1f} min  (extrapolated from n={n_ref:,} tight benchmark)")
    print(f"  HAC on components adds to this; actual time depends on component sizes.")

print(f"\nDense baseline estimate (n={N_RW:,}):")
ref_dense = [r for r in rows
             if r["scenario"] == "tight" and r["h_max_km"] == 20.0 and r["n"] == 10_000]
if ref_dense:
    t_dense_ref = ref_dense[0]["dense_time_mean"]
    t_dense_rw  = t_dense_ref * (N_RW / 10_000) ** 2
    print(f"  ~{t_dense_rw/3600:.1f} hours  (O(n^2) extrapolation from n=10k)")
    print(f"  Memory: ~{dense_matrix_gb:.0f} GB  → not feasible on any workstation")

# ------------------------------------------------------------------
# PySAL DistanceBand graph construction comparison
# ------------------------------------------------------------------

try:
    import libpysal
    _PYSAL = True
    print(f"\n--- PySAL DistanceBand graph construction comparison "
          f"(libpysal {libpysal.__version__}) ---")
except ImportError:
    _PYSAL = False
    print("\n--- PySAL comparison skipped (libpysal not installed) ---")
    print("    Install with: pip install libpysal")

if _PYSAL:
    PYSAL_N_SIZES    = [500, 1_000, 2_500, 5_000, 10_000, 25_000]
    PYSAL_H_MAX_VALS = [10_000, 20_000]
    PYSAL_SCENARIOS  = ["tight", "moderate", "loose"]

    pysal_rows = []

    for scenario_name in PYSAL_SCENARIOS:
        scfg     = SCENARIOS[scenario_name]
        for h_max in PYSAL_H_MAX_VALS:
            h_max_km = h_max / 1000
            spread_m = scfg["spread_factor"] * h_max
            for n in PYSAL_N_SIZES:
                pts = generate_clustered_points(
                    n_total=n,
                    n_centres=scfg["n_centres"],
                    spread_m=spread_m,
                )
                _, t_our,   _ = measure(lambda: spatial_dist_graph(pts, h_max))
                _, t_dband, _ = measure(
                    lambda: libpysal.weights.DistanceBand(
                        pts, threshold=h_max, binary=True,
                        build_sp=True, silence_warnings=True,
                    )
                )
                ratio = round(t_dband / t_our, 2) if t_our > 0 else None
                print(f"  {scenario_name:<10} n={n:>6,}  h={h_max_km:.0f}km  "
                      f"ours={t_our:.4f}s  DistanceBand={t_dband:.4f}s  "
                      f"ratio={ratio}x")
                pysal_rows.append({
                    "scenario":      scenario_name,
                    "n":             n,
                    "h_max_km":      h_max_km,
                    "graph_our_s":   round(t_our, 4),
                    "graph_dband_s": round(t_dband, 4),
                    "dband_our_ratio": ratio,
                })

    pysal_out = f"{RESULTS_DIR}/benchmark_pysal.csv"
    with open(pysal_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pysal_rows[0].keys())
        writer.writeheader()
        writer.writerows(pysal_rows)
    print(f"\nPySAL graph construction comparison written to {pysal_out}")
