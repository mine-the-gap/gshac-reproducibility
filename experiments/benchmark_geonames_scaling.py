"""
benchmark_geonames_scaling.py

GeoNames scaling experiment: subsample the 2M spatially-stratified GeoNames
sample to a ladder of n at a fixed, RAM-feasible threshold, and measure wall
time + peak memory at each n. Fit the log-log scaling and use it to extrapolate
to larger scales (the full 13.4M dataset, or memory-infeasible thresholds)
without ever running a config that exceeds RAM.

KEY: each (n, rep) measurement runs in a FRESH SUBPROCESS. When the worker
exits, the OS reclaims 100% of its pages, so there is zero cross-rep memory
accumulation — the cause of the in-process 5-rep OOM at the heaviest configs.
The orchestrator parses + stratifies GeoNames once, saves the 2M coords to a
temp .npy, and each worker loads it, draws its n-subset, measures, and exits.

Modes:
  worker: python benchmark_geonames_scaling.py --worker <npy> <n> <h_max> <seed>
  main:   python benchmark_geonames_scaling.py

Output: results/benchmark_geonames_scaling.csv
"""

from __future__ import annotations

import _env  # noqa: F401  (pins thread pools to 1 and the paging state; must precede numpy/scipy/sklearn)

import sys, os, json, time, gc, tracemalloc, csv, subprocess
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / ("results_fast" if os.environ.get("GSHAC_FAST") == "1" else "results")
RESULTS_DIR.mkdir(exist_ok=True)

_MIB = 1024 ** 2
_GIB = 1024 ** 3

# Set GSHAC_FAST=1 for a reduced-scale reproducibility check (see make reproduce-fast).
FAST     = os.environ.get("GSHAC_FAST") == "1"
H_MAX    = 10_000   # metres; 10 km is feasible to 2M (~8.6 GiB graph)
N_LADDER = [50_000, 100_000, 200_000] if FAST else [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
N_REP    = 1 if FAST else 5
SEED     = 42


# ------------------------------------------------------------------
# Worker: one isolated measurement (graph build + single-linkage HAC)
# ------------------------------------------------------------------
def _worker(npy_path, n, h_max, seed):
    from gshac.spatial_dist_graph import spatial_dist_graph
    from gshac.sparse_hclust import sparse_hclust

    coords_all = np.load(npy_path)                      # (2M, 2) float32
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords_all), size=n, replace=False)
    coords = np.ascontiguousarray(coords_all[idx])
    del coords_all
    gc.collect()

    gc.collect()
    tracemalloc.start()
    c0 = time.process_time()  # CPU-time evidence: single-threaded => cpu_s ~= wall
    t0 = time.perf_counter()
    g = spatial_dist_graph(coords, h_max, metric="haversine")
    t_graph = time.perf_counter() - t0
    cur, peak_g = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    r = sparse_hclust(g, [h_max], method="single")
    t_hac = time.perf_counter() - t1
    _, peak_h = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_s = time.process_time() - c0
    import resource
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # peak RSS cross-check

    out = dict(
        n=n, h_max_m=h_max, n_edges=int(g["n_edges"]),
        mean_k=round(2 * g["n_edges"] / n, 2),
        n_components=int(g["n_components"]),
        n_clusters=int(np.unique(r["labels"][h_max]).size),
        t_graph=t_graph, t_hac=t_hac, t_total=t_graph + t_hac,
        cpu_s=round(cpu_s, 4),
        rss_mib=round(rss_mib, 1),
        graph_mem=peak_g / _MIB,
        hac_mem=(peak_h - cur) / _MIB,
        pipeline_mem=max(peak_g, peak_h) / _MIB,
    )
    print("RESULT " + json.dumps(out), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    sys.exit(0)


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------
def _agg(reps, key):
    vals = [r[key] for r in reps]
    mean = round(float(np.mean(vals)), 3)
    sd = round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else 0.0
    return mean, sd


if __name__ == "__main__":
    # Reuse the parse + stratified-sample logic from the threshold-sweep script.
    from benchmark_geonames import (download_geonames, load_coords,
                                    stratified_sample, N_SAMPLE, RANDOM_SEED)

    print("=" * 64)
    print(f"GeoNames scaling ladder @ h_max = {H_MAX/1000:.0f} km, "
          f"{N_REP} reps/point (each rep in its own subprocess)")
    print(f"  n ladder = {N_LADDER}")
    print("=" * 64)

    download_geonames()
    df = load_coords()
    coords = stratified_sample(df, N_SAMPLE, seed=RANDOM_SEED).astype(np.float32)
    del df
    gc.collect()

    npy = str(RESULTS_DIR / "_geonames_coords_tmp.npy")
    np.save(npy, coords)
    del coords
    gc.collect()
    print(f"Saved 2M coords to {npy}; spawning isolated workers ...\n")

    rows = []
    for n in N_LADDER:
        reps = []
        for rep in range(N_REP):
            cp = subprocess.run(
                [sys.executable, __file__, "--worker", npy, str(n),
                 str(H_MAX), str(SEED + rep)],
                capture_output=True, text=True,
            )
            line = next((l for l in cp.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if line is None:
                tail = (cp.stderr or cp.stdout)[-400:]
                print(f"  n={n:,} rep={rep+1}: WORKER FAILED (rc={cp.returncode})\n  {tail}")
                continue
            m = json.loads(line[len("RESULT "):])
            reps.append(m)
            print(f"  n={n:>9,} rep {rep+1}/{N_REP}: "
                  f"graph {m['t_graph']:.1f}s/{m['graph_mem']:.0f}MiB  "
                  f"HAC {m['t_hac']:.1f}s/{m['hac_mem']:.0f}MiB  "
                  f"pipeline {m['pipeline_mem']:.0f}MiB")
        if not reps:
            print(f"  n={n:,}: all reps failed — skipping")
            continue

        row = {"n": n, "h_max_m": H_MAX, "n_reps": len(reps),
               "n_edges": reps[0]["n_edges"], "mean_k": reps[0]["mean_k"],
               "n_components": reps[0]["n_components"],
               "n_clusters": reps[0]["n_clusters"],
               "sparse_mib": round(reps[0]["n_edges"] * 2 * 8 / _MIB, 1)}
        for k, name in [("t_graph", "graph_time_s"), ("t_hac", "hac_time_s"),
                        ("t_total", "total_time_s"),
                        ("graph_mem", "graph_mem_mib"),
                        ("hac_mem", "hac_mem_mib"),
                        ("pipeline_mem", "pipeline_mem_mib"),
                        ("cpu_s", "cpu_s"),        # single-thread evidence
                        ("rss_mib", "rss_mib")]:   # peak-RSS cross-check
            mean, sd = _agg(reps, k)
            row[name] = mean
            row[name + "_sd"] = sd
        rows.append(row)

    out = RESULTS_DIR / "benchmark_geonames_scaling.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {out}")

    # Fitted log-log scaling exponents (for extrapolation) -> CSV + console
    ns = np.array([r["n"] for r in rows], dtype=float)
    slope_rows = []
    for key, label in [("total_time_s", "time"), ("pipeline_mem_mib", "memory")]:
        ys = np.array([r[key] for r in rows], dtype=float)
        ok = ys > 0
        slope = float(np.polyfit(np.log(ns[ok]), np.log(ys[ok]), 1)[0])
        slope_rows.append({"metric": label, "h_max_m": H_MAX,
                           "slope": round(slope, 3),
                           "n_min": int(ns.min()), "n_max": int(ns.max())})
        print(f"  {label} scaling: n^{slope:.2f}")
    with open(RESULTS_DIR / "benchmark_geonames_scaling_slopes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slope_rows[0].keys()))
        w.writeheader()
        w.writerows(slope_rows)

    try:
        os.remove(npy)
    except OSError:
        pass
    print("Done.")
