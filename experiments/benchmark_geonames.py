"""
benchmark_geonames.py

Scalability benchmark of sparse single-linkage HAC on a spatially stratified
2M-point sample of GeoNames allCountries (n_full ~ 13.4M), across
h_max = 1 km / 5 km / 10 km / 20 km. The 20 km threshold sits at the memory
limit of a 30 GiB workstation and is the headline scalability result.

Phases:
  1. Download + parse GeoNames allCountries.txt (cached in data/)
  2. Draw a 2M spatially-stratified random sample (1 deg x 1 deg grid, proportional)
  3. For each h_max: build sparse graph + run single-linkage HAC, N_REP times,
     reporting mean +/- sd for timing and for three memory figures (graph-phase
     peak, HAC working memory, and whole-pipeline peak), all in binary MiB.
  4. Write results/benchmark_geonames.csv

This script is self-contained: it downloads allCountries.txt on first run.
"""

from __future__ import annotations

import sys, os, time, gc, tracemalloc, csv, urllib.request, zipfile
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TXT_PATH    = DATA_DIR / "allCountries.txt"
ZIP_PATH    = DATA_DIR / "geonames_allCountries.zip"
ZIP_URL     = "https://download.geonames.org/export/dump/allCountries.zip"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLE     = 2_000_000
N_FULL_REF   = 13_412_837                          # for context only (no extrapolation)
RANDOM_SEED  = 42
H_MAX_VALS   = [1_000, 5_000, 10_000, 20_000]      # metres (2 km dropped)
N_REP        = 5

_MIB = 1024 ** 2
_GIB = 1024 ** 3


# ------------------------------------------------------------------
# Two-phase pipeline measurement (single build; graph + HAC + whole-run peaks)
# ------------------------------------------------------------------

def measure_pipeline(build_fn, run_fn):
    """Measure build-then-run in ONE tracemalloc scope. See benchmark_mining.py
    for the rationale. Returns (graph, result, metrics) in MiB / seconds."""
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    graph = build_fn()
    t_graph = time.perf_counter() - t0
    cur_after_graph, peak_graph = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    result = run_fn(graph)
    t_hac = time.perf_counter() - t1
    _, peak_during_hac = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "t_graph": t_graph,
        "t_hac": t_hac,
        "t_total": t_graph + t_hac,
        "graph_mem": peak_graph / _MIB,
        "hac_mem": (peak_during_hac - cur_after_graph) / _MIB,
        "pipeline_mem": max(peak_graph, peak_during_hac) / _MIB,
    }
    return graph, result, metrics


def _mean_sd(xs):
    return round(float(np.mean(xs)), 4), round(float(np.std(xs, ddof=1)), 4)


# ------------------------------------------------------------------
# 0. Download (self-contained; cached)
# ------------------------------------------------------------------

def download_geonames():
    if TXT_PATH.exists():
        print(f"  Using cached {TXT_PATH}  ({TXT_PATH.stat().st_size/_MIB:.0f} MiB)")
        return
    if not ZIP_PATH.exists():
        print(f"  Downloading {ZIP_URL} ...")
        def progress(count, block, total):
            pct = count * block / total * 100
            print(f"\r  {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=progress)
        print()
    print("  Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract("allCountries.txt", DATA_DIR)
    print(f"  Extracted to {TXT_PATH}  ({TXT_PATH.stat().st_size/_MIB:.0f} MiB)")


# ------------------------------------------------------------------
# 1. Parse
# ------------------------------------------------------------------
def load_coords():
    import pandas as pd
    print(f"Parsing {TXT_PATH} ...")
    t0 = time.perf_counter()
    df = pd.read_csv(
        TXT_PATH, sep="\t", header=None, usecols=[4, 5],
        names=["lat", "lon"], dtype={"lat": "float32", "lon": "float32"},
        on_bad_lines="skip", engine="c",
    )
    print(f"  {len(df):,} features in {time.perf_counter()-t0:.1f} s")
    return df


# ------------------------------------------------------------------
# 2. Spatially stratified sample (1 deg x 1 deg grid, proportional allocation)
# ------------------------------------------------------------------
def stratified_sample(df, n_sample, seed=42):
    """Draw n_sample rows using 1 deg x 1 deg grid cells as strata with
    proportional (largest-remainder) allocation. Returns (n_sample, 2)
    float32 array of (lon, lat)."""
    rng = np.random.default_rng(seed)
    n_full = len(df)

    cell_id = (np.floor(df["lon"].to_numpy()).astype(np.int32) * 200
               + np.floor(df["lat"].to_numpy()).astype(np.int32))

    sort_idx    = np.argsort(cell_id, kind="stable")
    sorted_c    = cell_id[sort_idx]
    _, g_start  = np.unique(sorted_c, return_index=True)
    g_end       = np.append(g_start[1:], n_full)
    g_sizes     = g_end - g_start
    n_cells     = len(g_start)

    alloc_f   = g_sizes / n_full * n_sample
    alloc     = np.floor(alloc_f).astype(np.int64)
    shortfall = int(n_sample - alloc.sum())
    remainder = alloc_f - alloc
    topup     = np.argsort(remainder)[::-1][:shortfall]
    alloc[topup] += 1

    print(f"  Stratified sample: {n_cells:,} cells, "
          f"target {n_sample:,} / {n_full:,} ({n_sample/n_full*100:.1f}%)")

    selected = []
    for start, end, k in zip(g_start, g_end, alloc):
        if k == 0:
            continue
        pool = sort_idx[start:end]
        selected.append(rng.choice(pool, size=min(int(k), len(pool)), replace=False))
    sel_idx = np.concatenate(selected)
    coords  = df[["lon", "lat"]].to_numpy(dtype=np.float32)[sel_idx]
    print(f"  Sampled {len(coords):,} features")
    return coords


# ------------------------------------------------------------------
# 3. Benchmark one threshold (N_REP repetitions)
# ------------------------------------------------------------------
def run_benchmark(coords, h_max_m):
    n = len(coords)
    h_cut = h_max_m  # single cut at h_max

    print(f"\n{'='*55}")
    print(f"h_max = {h_max_m:,} m  (n = {n:,}),  {N_REP} reps")
    print(f"{'='*55}")

    graph_times, hac_times, total_times = [], [], []
    graph_mems, hac_mems, pipeline_mems = [], [], []

    n_edges = n_components = max_c = n_singletons = None
    comp_sizes = None
    n_clusters = None

    for rep in range(N_REP):
        graph, result, m = measure_pipeline(
            lambda: spatial_dist_graph(coords, h_max_m, metric="haversine"),
            lambda g: sparse_hclust(g, [h_cut], method="single"),
        )
        graph_times.append(m["t_graph"]);   hac_times.append(m["t_hac"])
        total_times.append(m["t_total"])
        graph_mems.append(m["graph_mem"]);  hac_mems.append(m["hac_mem"])
        pipeline_mems.append(m["pipeline_mem"])

        if rep == 0:
            n_edges      = graph["n_edges"]
            n_components = graph["n_components"]
            comp_sizes   = np.bincount(graph["components"])
            max_c        = int(comp_sizes.max())
            n_singletons = int((comp_sizes == 1).sum())
            n_clusters   = int(np.unique(result["labels"][h_cut]).size)

        print(f"  rep {rep+1}/{N_REP}:  graph {m['t_graph']:.1f}s / {m['graph_mem']:.0f} MiB"
              f"  |  HAC {m['t_hac']:.1f}s / {m['hac_mem']:.0f} MiB"
              f"  |  pipeline peak {m['pipeline_mem']:.0f} MiB")

        # Critical: free the (large) graph before the next repetition.
        del graph, result
        gc.collect()

    mean_k = 2 * n_edges / n

    gt_mean, gt_sd = _mean_sd(graph_times)
    ht_mean, ht_sd = _mean_sd(hac_times)
    tt_mean, tt_sd = _mean_sd(total_times)
    gm_mean, gm_sd = _mean_sd(graph_mems)
    hm_mean, hm_sd = _mean_sd(hac_mems)
    pm_mean, pm_sd = _mean_sd(pipeline_mems)

    # Analytical sizes (binary; full AND condensed)
    sparse_mib            = n_edges * 2 * 8 / _MIB
    dense_full_gib        = n ** 2 * 8 / _GIB
    dense_condensed_gib   = n * (n - 1) / 2 * 8 / _GIB
    max_sub_full_gib      = max_c ** 2 * 8 / _GIB
    max_sub_condensed_gib = max_c * (max_c - 1) / 2 * 8 / _GIB

    print(f"  Edges: {n_edges:,}  (mean k = {mean_k:.1f})  |  components {n_components:,}"
          f"  |  max comp {max_c:,}")
    print(f"  Graph {gt_mean:.1f}s / {gm_mean:.0f} MiB  |  HAC {ht_mean:.1f}s / {hm_mean:.0f} MiB"
          f"  |  total {tt_mean:.1f}s  |  pipeline peak {pm_mean:.0f} MiB")
    print(f"  Largest-component dense sub-matrix avoided: "
          f"{max_sub_full_gib:.1f} GiB (full) / {max_sub_condensed_gib:.1f} GiB (condensed)")

    return {
        "n_sample":       n,
        "n_full":         N_FULL_REF,
        "h_max_m":        h_max_m,
        "h_cut_m":        h_cut,
        "metric":         "haversine",
        "linkage":        "single",
        "n_reps":         N_REP,
        "n_edges":        n_edges,
        "mean_k":         round(mean_k, 2),
        "sparse_mib":     round(sparse_mib, 1),
        "dense_full_gib_sample":      round(dense_full_gib, 1),
        "dense_condensed_gib_sample": round(dense_condensed_gib, 1),
        "n_components":   n_components,
        "comp_p50":       int(np.percentile(comp_sizes, 50)),
        "comp_p90":       int(np.percentile(comp_sizes, 90)),
        "comp_p99":       int(np.percentile(comp_sizes, 99)),
        "comp_max":       max_c,
        "n_singletons":   n_singletons,
        "max_sub_full_gib":      round(max_sub_full_gib, 2),
        "max_sub_condensed_gib": round(max_sub_condensed_gib, 2),
        # Timings (s) — mean / sd
        "graph_time_s":    gt_mean, "graph_time_s_sd": gt_sd,
        "hac_time_s":      ht_mean, "hac_time_s_sd":   ht_sd,
        "total_time_s":    tt_mean, "total_time_s_sd": tt_sd,
        # Memory (MiB, binary) — mean / sd
        "graph_mem_mib":     round(gm_mean, 1), "graph_mem_mib_sd":     round(gm_sd, 1),
        "hac_mem_mib":       round(hm_mean, 1), "hac_mem_mib_sd":       round(hm_sd, 1),
        "pipeline_mem_mib":  round(pm_mean, 1), "pipeline_mem_mib_sd":  round(pm_sd, 1),
        "n_clusters":     n_clusters,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"GeoNames benchmark  ({N_SAMPLE/1e6:.0f}M-point stratified sample, {N_REP} reps)")
    print("=" * 60)

    download_geonames()
    df = load_coords()
    coords_sample = stratified_sample(df, N_SAMPLE, seed=RANDOM_SEED)
    del df
    gc.collect()

    print(f"\nCoordinate range: lon [{coords_sample[:,0].min():.1f}, {coords_sample[:,0].max():.1f}]"
          f"  lat [{coords_sample[:,1].min():.1f}, {coords_sample[:,1].max():.1f}]")

    results = [run_benchmark(coords_sample, h_max) for h_max in H_MAX_VALS]

    out = RESULTS_DIR / "benchmark_geonames.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out}")
