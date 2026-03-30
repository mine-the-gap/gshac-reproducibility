"""
benchmark_hdbscan.py

Resource comparison: sklearn HDBSCAN vs sparse HAC on:
  1. Global mining dataset  (n = 261,073)
  2. GeoNames 2M-point sample

HDBSCAN and sparse HAC solve different problems:
  - HDBSCAN  : density-based, noise labels, no cut-height parameter,
               no Ward/complete/average linkage
  - Sparse HAC: exact hierarchical, arbitrary cut height,
               all standard linkages, no noise

We report wall time and peak memory for each method and dataset.

Output: results/benchmark_hdbscan.csv
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


def measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

def load_mining():
    import geopandas as gpd
    gdf = gpd.read_file(DATA_DIR / "cluster_data.gpkg")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = gdf.geometry.centroid
    coords = np.column_stack([cx.x.to_numpy(), cx.y.to_numpy()]).astype(np.float64)
    print(f"  Mining: {len(coords):,} features")
    return coords


def load_geonames_sample(n_sample=2_000_000, seed=42):
    import pandas as pd
    txt = DATA_DIR / "allCountries.txt"
    df  = pd.read_csv(txt, sep="\t", header=None, usecols=[4, 5],
                      names=["lat", "lon"],
                      dtype={"lat": "float32", "lon": "float32"},
                      on_bad_lines="skip", engine="c")
    n_full = len(df)
    rng     = np.random.default_rng(seed)
    cell_id = (np.floor(df["lon"].to_numpy()).astype(np.int32) * 200
               + np.floor(df["lat"].to_numpy()).astype(np.int32))
    sort_idx   = np.argsort(cell_id, kind="stable")
    sorted_c   = cell_id[sort_idx]
    _, g_start = np.unique(sorted_c, return_index=True)
    g_end      = np.append(g_start[1:], n_full)
    g_sizes    = g_end - g_start
    alloc_f   = g_sizes / n_full * n_sample
    alloc     = np.floor(alloc_f).astype(np.int64)
    shortfall = int(n_sample - alloc.sum())
    topup     = np.argsort(alloc_f - alloc)[::-1][:shortfall]
    alloc[topup] += 1
    selected = []
    for start, end, k in zip(g_start, g_end, alloc):
        if k == 0:
            continue
        pool = sort_idx[start:end]
        selected.append(rng.choice(pool, size=min(int(k), len(pool)), replace=False))
    sel_idx = np.concatenate(selected)
    coords  = df[["lon", "lat"]].to_numpy(dtype=np.float64)[sel_idx]
    print(f"  GeoNames sample: {len(coords):,} / {n_full:,}")
    return coords


# ------------------------------------------------------------------
# Runners
# ------------------------------------------------------------------

def run_hdbscan(coords_lonlat, min_cluster_size, label):
    """
    sklearn HDBSCAN, haversine metric.
    coords_lonlat: (n, 2) float64 in (lon, lat) degrees.
    """
    from sklearn.cluster import HDBSCAN
    import warnings
    coords_rad = np.radians(coords_lonlat[:, [1, 0]])  # (lat, lon) radians

    def _fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = HDBSCAN(min_cluster_size=min_cluster_size,
                          metric="haversine", algorithm="ball_tree")
            clf.fit(coords_rad)
        return clf

    clf, elapsed, peak = measure(_fit)
    n_noise    = int((clf.labels_ == -1).sum())
    n_clusters = int(clf.labels_.max()) + 1
    pct_noise  = n_noise / len(coords_lonlat) * 100
    print(f"  HDBSCAN mcs={min_cluster_size}: "
          f"{elapsed:.1f} s | {peak:.0f} MiB | "
          f"{n_clusters:,} clusters | {n_noise:,} noise ({pct_noise:.1f}%)")
    return elapsed, peak, n_clusters, n_noise


def run_sparse_hac(coords_lonlat, h_max_m, h_cut_m):
    """Sparse HAC, single linkage, haversine."""
    coords = coords_lonlat.astype(np.float32)

    def _run():
        g = spatial_dist_graph(coords, h_max_m, metric="haversine")
        r = sparse_hclust(g, [h_cut_m], method="single")
        return g, r

    (g, r), elapsed, peak = measure(_run)
    n_clusters = int(np.unique(r["labels"][h_cut_m]).size)
    print(f"  Sparse HAC h_max={h_max_m/1000:.0f}km h_cut={h_cut_m/1000:.0f}km: "
          f"{elapsed:.1f} s | {peak:.0f} MiB | "
          f"{n_clusters:,} clusters | 0 noise")
    return elapsed, peak, n_clusters, 0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

print("=" * 60)
print("HDBSCAN vs sparse HAC resource comparison")
print("=" * 60)

rows = []

# ------------------------------------------------------------------
# 1. Mining dataset
# ------------------------------------------------------------------
print("\n--- Mining dataset (n=261,073) ---")
coords_m = load_mining()
n_m = len(coords_m)

# HDBSCAN
for mcs in [2, 5]:
    t, mem, nc, nn = run_hdbscan(coords_m, mcs, "mining")
    rows.append(dict(
        dataset="mining", n=n_m, method="HDBSCAN",
        min_cluster_size=mcs, h_max_m="", h_cut_m="",
        time_s=round(t, 1), peak_mib=round(mem, 0),
        n_clusters=nc, n_noise=nn,
        note="sklearn HDBSCAN, haversine, ball_tree"
    ))

# Sparse HAC (from pre-existing benchmark_mining.csv: total=69.9s)
# Re-run representative cut heights; h_max fixed at 20km per paper
for h_cut in [1_000, 5_000, 10_000, 20_000]:
    t, mem, nc, nn = run_sparse_hac(coords_m, 20_000, h_cut)
    rows.append(dict(
        dataset="mining", n=n_m, method="sparse_HAC_single",
        min_cluster_size="", h_max_m=20_000, h_cut_m=h_cut,
        time_s=round(t, 1), peak_mib=round(mem, 0),
        n_clusters=nc, n_noise=nn,
        note="sparse HAC, single linkage, haversine"
    ))

# ------------------------------------------------------------------
# 2. GeoNames 2M sample
# ------------------------------------------------------------------
print("\n--- GeoNames 2M sample ---")
coords_g = load_geonames_sample()
n_g = len(coords_g)

for mcs in [2, 5]:
    t, mem, nc, nn = run_hdbscan(coords_g, mcs, "geonames")
    rows.append(dict(
        dataset="geonames_2M", n=n_g, method="HDBSCAN",
        min_cluster_size=mcs, h_max_m="", h_cut_m="",
        time_s=round(t, 1), peak_mib=round(mem, 0),
        n_clusters=nc, n_noise=nn,
        note="sklearn HDBSCAN, haversine, ball_tree"
    ))

# Sparse HAC at h_max=1km (feasible per benchmark_geonames.csv)
for h_cut in [500, 1_000]:
    t, mem, nc, nn = run_sparse_hac(coords_g, 1_000, h_cut)
    rows.append(dict(
        dataset="geonames_2M", n=n_g, method="sparse_HAC_single",
        min_cluster_size="", h_max_m=1_000, h_cut_m=h_cut,
        time_s=round(t, 1), peak_mib=round(mem, 0),
        n_clusters=nc, n_noise=nn,
        note="sparse HAC, single linkage, haversine"
    ))

# ------------------------------------------------------------------
out = RESULTS_DIR / "benchmark_hdbscan.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResults written to {out}")
