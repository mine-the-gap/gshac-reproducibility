"""
benchmark_geonames.py

Spatially stratified 2M-point sample from GeoNames allCountries
(n_full ~ 13.4M), benchmarked across h_max = 1 km / 2 km / 5 km / 10 km.

Phases:
  1. Parse GeoNames (allCountries.txt must exist in data/)
  2. Draw a 2M spatially-stratified random sample (1° x 1° grid, proportional)
  3. For each h_max: build sparse graph + run single-linkage HAC
     - HAC is skipped (feasibility note written) if max-component dense
       sub-matrix would exceed MAX_SUBMAT_GB
  4. Write results/benchmark_geonames.csv

Run explore_geonames.py first to download and unzip allCountries.txt.
"""

from __future__ import annotations

import sys, time, tracemalloc, csv
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TXT_PATH    = DATA_DIR / "allCountries.txt"

N_SAMPLE     = 2_000_000
RANDOM_SEED  = 42
H_MAX_VALS   = [1_000, 2_000, 5_000, 10_000]   # metres
H_CUT_FRAC   = 1.0   # h_cut = h_max (single cut per run)
MAX_SUBMAT_GB = 20.0  # skip HAC if largest-component sub-matrix exceeds this


def measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


# ------------------------------------------------------------------
# 1. Parse
# ------------------------------------------------------------------
def load_coords():
    import pandas as pd
    print(f"Parsing {TXT_PATH} ...")
    t0 = time.perf_counter()
    df = pd.read_csv(
        TXT_PATH,
        sep="\t",
        header=None,
        usecols=[4, 5],
        names=["lat", "lon"],
        dtype={"lat": "float32", "lon": "float32"},
        on_bad_lines="skip",
        engine="c",
    )
    elapsed = time.perf_counter() - t0
    print(f"  {len(df):,} features in {elapsed:.1f} s")
    return df


# ------------------------------------------------------------------
# 2. Spatially stratified sample
# ------------------------------------------------------------------
def stratified_sample(df, n_sample, seed=42):
    """
    Draw n_sample rows from df using 1° x 1° grid cells as strata.
    Proportional allocation; O(n log n) via numpy argsort — no Python
    loop over individual rows.
    Returns a (n_sample, 2) float32 array of (lon, lat).
    """
    rng = np.random.default_rng(seed)
    n_full = len(df)

    cell_id = (np.floor(df["lon"].to_numpy()).astype(np.int32) * 200
               + np.floor(df["lat"].to_numpy()).astype(np.int32))

    # Sort by cell; get group boundaries
    sort_idx    = np.argsort(cell_id, kind="stable")
    sorted_c    = cell_id[sort_idx]
    _, g_start  = np.unique(sorted_c, return_index=True)
    g_end       = np.append(g_start[1:], n_full)
    g_sizes     = g_end - g_start
    n_cells     = len(g_start)

    # Proportional allocation with largest-remainder rounding
    alloc_f   = g_sizes / n_full * n_sample
    alloc     = np.floor(alloc_f).astype(np.int64)
    shortfall = int(n_sample - alloc.sum())
    remainder = alloc_f - alloc
    topup     = np.argsort(remainder)[::-1][:shortfall]
    alloc[topup] += 1

    print(f"  Stratified sample: {n_cells:,} cells, "
          f"target {n_sample:,} / {n_full:,} "
          f"({n_sample/n_full*100:.1f}%)")

    # One Python loop over cells (n_cells ~ 16k, fast)
    selected = []
    for start, end, k in zip(g_start, g_end, alloc):
        if k == 0:
            continue
        pool = sort_idx[start:end]
        chosen = rng.choice(pool, size=min(int(k), len(pool)), replace=False)
        selected.append(chosen)

    sel_idx = np.concatenate(selected)
    coords  = df[["lon", "lat"]].to_numpy(dtype=np.float32)[sel_idx]
    print(f"  Sampled {len(coords):,} features")
    return coords


# ------------------------------------------------------------------
# 3. Benchmark loop
# ------------------------------------------------------------------
def run_benchmark(coords, h_max_m, dense_gb_full):
    n = len(coords)
    h_cut = h_max_m  # cut at h_max

    print(f"\n{'='*55}")
    print(f"h_max = {h_max_m:,} m  (n = {n:,})")
    print(f"{'='*55}")

    # Graph
    print("  Building sparse graph ...")
    graph, t_graph, mem_graph = measure(
        lambda: spatial_dist_graph(coords, h_max_m, metric="haversine")
    )

    n_edges      = graph["n_edges"]
    n_components = graph["n_components"]
    comp_labels  = graph["components"]
    comp_sizes   = np.bincount(comp_labels)
    mean_k       = 2 * n_edges / n
    sparse_mb    = n_edges * 2 * 8 / 1e6
    dense_gb_n   = n ** 2 * 8 / 1e9
    max_c        = int(comp_sizes.max())
    max_sub_gb   = max_c ** 2 * 8 / 1e9

    print(f"  Graph: {t_graph:.1f} s  |  peak {mem_graph:.0f} MiB")
    print(f"  Edges: {n_edges:,}  (mean k = {mean_k:.1f})")
    print(f"  Sparse: {sparse_mb:.0f} MB  vs  dense n²: {dense_gb_n:.1f} GB  ({dense_gb_n/sparse_mb*1000:.0f}×)")
    print(f"  Components: {n_components:,}  |  max size: {max_c:,}")
    for p in [50, 90, 99]:
        print(f"    p{p}: {int(np.percentile(comp_sizes, p)):,}")
    print(f"    singletons: {(comp_sizes==1).sum():,}  ({(comp_sizes==1).mean()*100:.1f}%)")
    print(f"  Largest-component sub-matrix: {max_sub_gb:.1f} GB")

    # HAC feasibility — MST-based single linkage never forms the dense
    # sub-matrix, so the size check only applies to non-single methods.
    linkage_method = "single"
    if max_sub_gb > MAX_SUBMAT_GB and linkage_method != "single":
        print(f"  HAC SKIPPED: max sub-matrix {max_sub_gb:.1f} GB > {MAX_SUBMAT_GB} GB limit")
        hac_feasible  = False
        t_hac         = float("nan")
        mem_hac       = float("nan")
        n_clusters    = None
        hac_note      = f"skipped: max component sub-matrix {max_sub_gb:.1f} GB"
    else:
        if max_sub_gb > MAX_SUBMAT_GB:
            print(f"  Note: max sub-matrix {max_sub_gb:.1f} GB, but MST path avoids dense allocation")
        print(f"  Running sparse HAC ({linkage_method} linkage, h_cut = {h_cut} m) ...")
        result, t_hac, mem_hac = measure(
            lambda: sparse_hclust(graph, [h_cut], method=linkage_method)
        )
        n_clusters   = int(np.unique(result["labels"][h_cut]).size)
        hac_feasible = True
        hac_note     = "ok (MST path)" if max_sub_gb > MAX_SUBMAT_GB else "ok"
        print(f"  HAC: {t_hac:.1f} s  |  peak {mem_hac:.0f} MiB")
        print(f"  Clusters at h_cut={h_cut} m: {n_clusters:,}")

    t_total  = t_graph + (t_hac if hac_feasible else 0)
    print(f"  Total time: {t_total:.1f} s  ({t_total/60:.1f} min)")

    # Extrapolation to full dataset (n_full = 13.4M)
    n_full     = 13_412_837
    scale      = n_full / n
    t_full_est = t_total * scale  # rough linear estimate
    sparse_mb_full_est = sparse_mb * scale
    print(f"\n  --- Extrapolation to full n={n_full:,} ---")
    print(f"  Dense matrix:  {dense_gb_full:.0f} TB  (reference; infeasible)")
    print(f"  Sparse (est):  {sparse_mb_full_est/1e3:.0f} GB  (linear scale)")
    print(f"  Time est.:     {t_full_est:.0f} s  ({t_full_est/60:.1f} min)")

    return {
        "n_sample":       n,
        "n_full":         n_full,
        "h_max_m":        h_max_m,
        "h_cut_m":        h_cut if hac_feasible else "",
        "metric":         "haversine",
        "linkage":        "single" if hac_feasible else "",
        "n_edges":        n_edges,
        "mean_k":         round(mean_k, 2),
        "sparse_mb":      round(sparse_mb, 1),
        "dense_gb_sample": round(dense_gb_n, 1),
        "dense_tb_full":  round(dense_gb_full / 1e3, 0),
        "n_components":   n_components,
        "comp_p50":       int(np.percentile(comp_sizes, 50)),
        "comp_p90":       int(np.percentile(comp_sizes, 90)),
        "comp_p99":       int(np.percentile(comp_sizes, 99)),
        "comp_max":       max_c,
        "n_singletons":   int((comp_sizes == 1).sum()),
        "max_sub_gb":     round(max_sub_gb, 2),
        "graph_time_s":   round(t_graph, 1),
        "graph_mem_mib":  round(mem_graph, 0),
        "hac_time_s":     round(t_hac, 1) if hac_feasible else "",
        "hac_mem_mib":    round(mem_hac, 0) if hac_feasible else "",
        "total_time_s":   round(t_total, 1),
        "n_clusters":     n_clusters if hac_feasible else "",
        "hac_note":       hac_note,
        "t_full_est_s":   round(t_full_est, 0),
        "sparse_mb_full_est": round(sparse_mb_full_est, 0),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("=" * 60)
print(f"GeoNames benchmark  ({N_SAMPLE/1e6:.0f}M-point stratified sample)")
print("=" * 60)

if not TXT_PATH.exists():
    print(f"ERROR: {TXT_PATH} not found. Run explore_geonames.py first.")
    sys.exit(1)

df = load_coords()
n_full = len(df)
dense_gb_full = n_full ** 2 * 8 / 1e9

coords_sample = stratified_sample(df, N_SAMPLE, seed=RANDOM_SEED)
del df  # free ~3 GB before graph construction

print(f"\nCoordinate range: lon [{coords_sample[:,0].min():.1f}, {coords_sample[:,0].max():.1f}]"
      f"  lat [{coords_sample[:,1].min():.1f}, {coords_sample[:,1].max():.1f}]")

results = []
for h_max in H_MAX_VALS:
    r = run_benchmark(coords_sample, h_max, dense_gb_full)
    results.append(r)

# Write CSV
out = RESULTS_DIR / "benchmark_geonames.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults written to {out}")
