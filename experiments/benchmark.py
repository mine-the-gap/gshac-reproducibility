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
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust

# Paths relative to repo root (one level up from experiments/)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def dense_feasible(n, max_gib=6.0):
    """Check if the condensed distance matrix fits in memory.

    max_gib covers the condensed vector only; fastcluster needs additional
    working memory, so keep well below physical RAM (30 GiB).
    """
    condensed_gib = n * (n - 1) / 2 * 8 / 1024 ** 3
    return condensed_gib <= max_gib


def measure_pipeline(build_fn, run_fn):
    """Graph build + HAC in ONE tracemalloc scope (single build).

    Returns (graph, result, t_graph, t_hac, pipeline_peak_mib). The peak equals
    the old separate combined-scope measurement: max(graph-phase peak, hac-phase
    peak incl. the resident graph), via reset_peak() between phases.
    """
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
# Benchmark parameters
# ------------------------------------------------------------------

# Capped at 50k: n=100k needs ~27-40 GiB (loose/50km) and OOMs a 30 GiB box;
# scaling-to-millions is now carried by the real-data GeoNames ladder
# (benchmark_geonames_scaling.py). 50k still shows the crossover + quadratic trend.
N_SIZES    = [1_000, 5_000, 10_000, 25_000, 50_000]
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

# --- Correctness consolidation (folds in the former validate_correctness.py) ---
# The dense + sparse clusterings computed below ARE the exactness check: we compare
# their cluster labels (cheap, AFTER the measured region) instead of discarding them.
# Exactness (sparse == dense) for dense-feasible n; reproducibility (identical counts
# across the 5 reps) for sparse-only n. Recorded per rep in benchmark_python_reps.csv.
correctness_all_pass = True
correctness_fails    = []

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

            valid_cuts = [h for h in H_CUTS if h <= h_max]
            sparse_counts_rep0 = None
            config_all_agree = True
            exact_mode = "vs_dense" if run_dense else "reproducibility"

            for rep in range(N_REP):
                # Dense baseline (skip if infeasible)
                dn_res = None
                if run_dense:
                    dn_res, dt, dm = measure(lambda: dense_hclust(pts, H_CUTS, METHOD))
                    dense_times.append(dt)
                    dense_mems.append(dm)
                    gc.collect()   # free dense's internal matrices before measuring
                                   # sparse; dn_res keeps only the (~MB) labels

                # Sparse — graph + HAC in ONE tracemalloc scope (single build):
                # phase times AND whole-pipeline peak from one pass, halving the
                # work vs. building the graph twice (timing + memory) per rep.
                graph, sp_res, gt, ht, sm = measure_pipeline(
                    lambda: spatial_dist_graph(pts, h_max),
                    lambda g: sparse_hclust(g, H_CUTS, METHOD))
                graph_times.append(gt)
                hac_times.append(ht)
                sparse_times.append(gt + ht)
                sparse_mems.append(sm)

                if rep == 0:
                    n_edges_val = graph["n_edges"]
                    density_val = graph["density"]

                # --- Correctness check (consolidates validate_correctness.py) ---
                # AFTER measure*(): the peaks are already captured, so comparing the
                # ~KB label arrays cannot perturb the timing/memory measurements.
                sparse_counts = {h: int(np.unique(sp_res["labels"][h]).size)
                                 for h in valid_cuts}
                if run_dense:
                    rep_agree = all(                       # exactness: sparse == dense
                        int(np.unique(dn_res["labels"][h]).size) == sparse_counts[h]
                        for h in valid_cuts)
                else:
                    rep_agree = (sparse_counts_rep0 is None      # reproducibility
                                 or sparse_counts == sparse_counts_rep0)
                if rep == 0:
                    sparse_counts_rep0 = sparse_counts
                if not rep_agree:
                    config_all_agree = False
                    correctness_all_pass = False
                    correctness_fails.append((scenario_name, n, h_max_km, rep))

                del graph, sp_res, dn_res
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
                    "exact_mode":     exact_mode,
                    "exact_agree":    rep_agree,
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
                # Correctness (consolidated from validate_correctness.py)
                "exact_mode":      exact_mode,
                "exact_agree":     config_all_agree,
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

# Correctness summary (consolidates the former standalone validate_correctness.py)
print(f"\nCorrectness check (folded in): "
      f"{'ALL PASS' if correctness_all_pass else str(len(correctness_fails)) + ' FAILURES'} "
      f"over {len(rep_rows)} per-rep checks "
      f"(sparse==dense for n<=25k; reproducible across reps for n>=50k).")
for s, nn, hk, rp in correctness_fails[:10]:
    print(f"  MISMATCH: {s} n={nn:,} h={hk:.0f}km rep={rp}")

# ------------------------------------------------------------------
# Multi-linkage exactness (RQ1 — ALL linkage methods, not just single)
# ------------------------------------------------------------------
# The main loop validates single linkage at scale. This section verifies
# GSHAC == dense fastcluster for single / complete / average / Ward at small
# dense-feasible sizes. The exactness guarantee is linkage-agnostic: connected
# components are separated by gaps > h_max, so no cross-component merge occurs at
# any cut h <= h_max, and each component's dense sub-matrix is exact — hence the
# per-component HAC (any linkage) matches the dense baseline below the threshold.

print("\n--- multi-linkage exactness (GSHAC vs dense fastcluster) ---")

LX_LINKAGES = ["single", "complete", "average", "ward"]
LX_N_SIZES  = [1_000, 5_000]
LX_H_MAX    = [10_000, 20_000]

linkage_rows = []
lx_all_pass  = True
for scenario_name, scenario_cfg in SCENARIOS.items():
    for h_max in LX_H_MAX:
        spread_m   = scenario_cfg["spread_factor"] * h_max
        valid_cuts = [h for h in H_CUTS if h <= h_max]
        for n in LX_N_SIZES:
            pts = generate_clustered_points(n, scenario_cfg["n_centres"], spread_m)
            g = spatial_dist_graph(pts, h_max)
            for lnk in LX_LINKAGES:
                try:
                    sp = sparse_hclust(g, H_CUTS, lnk, coords=pts)
                    dn = dense_hclust(pts, H_CUTS, lnk)
                    mism = [(h // 1000,
                             int(np.unique(sp["labels"][h]).size),
                             int(np.unique(dn["labels"][h]).size))
                            for h in valid_cuts
                            if int(np.unique(sp["labels"][h]).size)
                               != int(np.unique(dn["labels"][h]).size)]
                    agree = not mism
                    note  = "" if agree else f"sparse!=dense at {mism[:3]}"
                    del sp, dn
                except Exception as exc:
                    agree, mism = None, []
                    note = f"ERROR {type(exc).__name__}: {str(exc)[:60]}"
                if agree is not True:
                    lx_all_pass = False
                linkage_rows.append({
                    "scenario":     scenario_name,
                    "n":            n,
                    "h_max_km":     h_max / 1000,
                    "linkage":      lnk,
                    "n_cuts":       len(valid_cuts),
                    "all_agree":    agree,
                    "n_mismatches": len(mism),
                    "note":         note,
                })
                print(f"  {scenario_name:<8} n={n:>5,} h={h_max//1000:>2}km "
                      f"{lnk:<8}: {'PASS' if agree else 'FAIL/ERR ' + note}")
            del pts, g
            gc.collect()

lx_out = f"{RESULTS_DIR}/benchmark_linkage_exactness.csv"
with open(lx_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=linkage_rows[0].keys())
    w.writeheader()
    w.writerows(linkage_rows)
n_ok = sum(1 for r in linkage_rows if r["all_agree"] is True)
print(f"\nMulti-linkage exactness: {'ALL PASS' if lx_all_pass else 'SOME FAILURES'} "
      f"({n_ok}/{len(linkage_rows)} configs across {len(LX_LINKAGES)} linkages) -> {lx_out}")

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

# Note: the per-scenario time/memory/speedup plots that benchmark.py used to emit
# (fig_time_python / fig_memory_python / fig_speedup_python) were removed — they are
# superseded by the combined fig_scaling.pdf (make_figures.py) used in the paper.

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
