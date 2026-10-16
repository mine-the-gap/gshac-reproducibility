"""
benchmark_hdbscan.py

Open-data resource comparison: sklearn HDBSCAN vs GSHAC sparse single-linkage HAC
on a 100k subset of the 2M GeoNames sample, plus a GeoNames HDBSCAN scaling probe
used to extrapolate HDBSCAN to the full 2M scale.

Both methods are measured in this one script over the identical coordinates with
the same whole-pipeline peak-memory definition; N_REP repetitions (mean +/- sd).
HDBSCAN and GSHAC solve different problems (density-based with noise vs exact
hierarchical at a distance threshold) — this reports resource profiles only.

Why 100k: HDBSCAN is feasible there (~min/fit) and already ~tens-x slower than
GSHAC; the probe shows the gap widening with n, with HDBSCAN extrapolated to
~hours at 2M (infeasible) while GSHAC does 2M in ~minutes. All on OPEN data.

Outputs:
  results/benchmark_hdbscan.csv          — GeoNames-100k HDBSCAN vs GSHAC
  results/benchmark_hdbscan_scaling.csv  — HDBSCAN scaling probe (GeoNames) + extrapolation
"""

from __future__ import annotations

import _env  # noqa: F401  (pins thread pools to 1 and the paging state; must precede numpy/scipy/sklearn)

import sys, os, time, gc, tracemalloc, csv, warnings
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / ("results_fast" if os.environ.get("GSHAC_FAST") == "1" else "results")
RESULTS_DIR.mkdir(exist_ok=True)

# Set GSHAC_FAST=1 for a reduced-scale reproducibility check (see make reproduce-fast).
FAST           = os.environ.get("GSHAC_FAST") == "1"
N_REP          = 1 if FAST else 5   # fast track runs once; validate via the full CI (make compare)
N_REP_HDBSCAN  = 1 if FAST else 5
N_SAMPLE       = 200_000 if FAST else 2_000_000   # GeoNames stratified sample (base)
COMPARISON_N   = 20_000 if FAST else 100_000      # subset for the head-to-head comparison
H_MAX          = 10_000           # metres (matches the GeoNames scaling ladder)
H_CUTS         = [5_000, 10_000]  # GSHAC cut heights for cluster counts
HDB_MCS        = [2, 5]
GEONAMES_N     = 2_000_000        # extrapolation target
PROBE_N        = [5_000, 10_000, 20_000] if FAST else [10_000, 25_000, 50_000, 100_000]
PROBE_MCS      = 5
SEED           = 42

_MIB = 1024 ** 2


def measure(fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / _MIB


def measure_pipeline(build_fn, run_fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    graph = build_fn()
    t_graph = time.perf_counter() - t0
    cur, peak_g = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    result = run_fn(graph)
    t_hac = time.perf_counter() - t1
    _, peak_h = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return graph, result, dict(
        t_graph=t_graph, t_hac=t_hac, t_total=t_graph + t_hac,
        graph_mem=peak_g / _MIB, hac_mem=(peak_h - cur) / _MIB,
        pipeline_mem=max(peak_g, peak_h) / _MIB)


def _mean_sd(xs):
    if len(xs) < 2:
        return round(float(np.mean(xs)), 3), 0.0
    return round(float(np.mean(xs)), 3), round(float(np.std(xs, ddof=1)), 3)


def load_geonames_2M():
    from benchmark_geonames import download_geonames, load_coords, stratified_sample
    download_geonames()
    df = load_coords()
    coords = stratified_sample(df, N_SAMPLE, seed=SEED).astype(np.float64)
    print(f"  GeoNames 2M sample: {len(coords):,} (lon, lat)")
    return coords


def run_hdbscan_once(coords_lonlat, mcs):
    from sklearn.cluster import HDBSCAN
    coords_rad = np.radians(coords_lonlat[:, [1, 0]])

    def _fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = HDBSCAN(min_cluster_size=mcs, metric="haversine", algorithm="ball_tree")
            clf.fit(coords_rad)
        return clf

    clf, elapsed, peak = measure(_fit)
    n_noise = int((clf.labels_ == -1).sum())
    n_clusters = int(clf.labels_.max()) + 1
    return elapsed, peak, n_clusters, n_noise


def run_gshac_once(coords_lonlat, h_max, h_cuts):
    coords = coords_lonlat.astype(np.float64)
    graph, result, m = measure_pipeline(
        lambda: spatial_dist_graph(coords, h_max, metric="haversine"),
        lambda g: sparse_hclust(g, h_cuts, method="single"))
    counts = {h: int(np.unique(result["labels"][h]).size) for h in h_cuts}
    del graph, result
    gc.collect()
    return m, counts


_FIELDS = ["dataset", "method", "min_cluster_size", "h_max_m", "n", "n_reps",
           "time_s", "time_s_sd", "pipeline_mem_mib", "pipeline_mem_mib_sd",
           "graph_time_s", "hac_time_s", "graph_mem_mib", "hac_mem_mib",
           "n_clusters", "n_clusters_h5km", "n_clusters_h10km",
           "n_noise", "noise_pct", "note"]


def _row(**kw):
    return {f: kw.get(f, "") for f in _FIELDS}


if __name__ == "__main__":
    print("=" * 60)
    print(f"HDBSCAN vs GSHAC — GeoNames-{COMPARISON_N // 1000}k (open data)")
    print("=" * 60)
    coords2M = load_geonames_2M()
    rng = np.random.default_rng(SEED)
    coords = np.ascontiguousarray(
        coords2M[rng.choice(len(coords2M), COMPARISON_N, replace=False)])
    n = len(coords)
    rows = []

    for mcs in HDB_MCS:
        print(f"\nHDBSCAN (mcs={mcs}), {N_REP_HDBSCAN} reps ...")
        times, mems = [], []
        nc = nn = None
        for rep in range(N_REP_HDBSCAN):
            t, peak, c, noise = run_hdbscan_once(coords, mcs)
            times.append(t); mems.append(peak)
            if rep == 0:
                nc, nn = c, noise
            print(f"  rep {rep+1}/{N_REP_HDBSCAN}: {t:.1f}s | {peak:.0f} MiB "
                  f"| {c:,} clusters | {noise:,} noise")
            gc.collect()
        tm, ts = _mean_sd(times)
        mm, ms = _mean_sd(mems)
        rows.append(_row(
            dataset="geonames", method="HDBSCAN", min_cluster_size=mcs, n=n,
            n_reps=N_REP_HDBSCAN, time_s=tm, time_s_sd=ts,
            pipeline_mem_mib=round(mm, 1), pipeline_mem_mib_sd=round(ms, 1),
            n_clusters=nc, n_noise=nn, noise_pct=round(nn / n * 100, 1),
            note="sklearn HDBSCAN, haversine, ball_tree; peak = whole-fit"))

    print(f"\nGSHAC sparse single linkage (h_max={H_MAX/1000:.0f} km), {N_REP} reps ...")
    g_t, g_pipe, g_gt, g_ht, g_gm, g_hm = [], [], [], [], [], []
    counts = None
    for rep in range(N_REP):
        m, counts = run_gshac_once(coords, H_MAX, H_CUTS)
        g_t.append(m["t_total"]); g_pipe.append(m["pipeline_mem"])
        g_gt.append(m["t_graph"]); g_ht.append(m["t_hac"])
        g_gm.append(m["graph_mem"]); g_hm.append(m["hac_mem"])
        print(f"  rep {rep+1}/{N_REP}: total {m['t_total']:.1f}s "
              f"| pipeline {m['pipeline_mem']:.0f} MiB")
    tm, ts = _mean_sd(g_t)
    pm, ps = _mean_sd(g_pipe)
    rows.append(_row(
        dataset="geonames", method="GSHAC_sparse_single", h_max_m=H_MAX, n=n,
        n_reps=N_REP, time_s=tm, time_s_sd=ts,
        pipeline_mem_mib=round(pm, 1), pipeline_mem_mib_sd=round(ps, 1),
        graph_time_s=round(float(np.mean(g_gt)), 3),
        hac_time_s=round(float(np.mean(g_ht)), 3),
        graph_mem_mib=round(float(np.mean(g_gm)), 1),
        hac_mem_mib=round(float(np.mean(g_hm)), 1),
        n_clusters=counts[H_MAX], n_clusters_h5km=counts[5000],
        n_clusters_h10km=counts[10000], n_noise=0, noise_pct=0.0,
        note="GSHAC single linkage, haversine; peak = whole pipeline (incl. graph)"))

    out = RESULTS_DIR / "benchmark_hdbscan.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nComparison written to {out}")

    # ---- HDBSCAN scaling probe on GeoNames subsamples + extrapolation ----
    print(f"\nHDBSCAN scaling probe (mcs={PROBE_MCS}) on GeoNames subsamples ...")
    probe = []
    for pn in PROBE_N:
        sub = coords2M[rng.choice(len(coords2M), pn, replace=False)]
        t, _, _, _ = run_hdbscan_once(sub, PROBE_MCS)
        probe.append({"n": pn, "time_s": round(t, 4)})
        print(f"  n={pn:>7,}: {t:.3f} s")
        gc.collect()
    # Project HDBSCAN's cost to the full 2M sample WITHOUT running it there (hours-long).
    # Fit a power law t = e^intercept * n^slope to the MEASURED probe timings above
    # (real run_hdbscan_once calls) and evaluate it at n=2M. The result is written with
    # kind="extrapolated_geonames_2M" (never "measured") and the fitted slope is recorded,
    # so the projection is fully reproducible and clearly flagged as an estimate.
    ns = np.array([p["n"] for p in probe], dtype=float)
    tsv = np.array([p["time_s"] for p in probe], dtype=float)
    slope, intercept = np.polyfit(np.log(ns), np.log(tsv), 1)
    t_2M = float(np.exp(intercept) * GEONAMES_N ** slope)
    print(f"  log-log slope = {slope:.2f}")
    print(f"  extrapolated HDBSCAN @ n={GEONAMES_N:,}: {t_2M:.0f} s (~{t_2M/3600:.1f} h)")
    srows = [{"n": int(p["n"]), "mcs": PROBE_MCS, "time_s": p["time_s"],
              "kind": "measured", "slope": ""} for p in probe]
    srows.append({"n": GEONAMES_N, "mcs": PROBE_MCS, "time_s": round(t_2M, 1),
                  "kind": "extrapolated_geonames_2M", "slope": round(float(slope), 3)})
    out_scaling = RESULTS_DIR / "benchmark_hdbscan_scaling.csv"
    with open(out_scaling, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "mcs", "time_s", "kind", "slope"])
        w.writeheader()
        w.writerows(srows)
    print(f"Scaling probe written to {out_scaling}")
    print("Done.")
