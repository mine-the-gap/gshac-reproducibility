"""
benchmark_mining.py

Actual timing and memory benchmark of the sparse HAC approach on the
real-world global mining dataset (cluster_data.gpkg).

The dense baseline is analytically infeasible at this n (would require
~541 GB RAM); this script measures only the sparse approach.

Outputs (relative to repo root):
  results/benchmark_mining.csv          — timing, memory, graph stats
  results/benchmark_mining_components.csv — component size distribution
  results/benchmark_mining_clusters.csv  — cluster counts at each cut height
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import csv
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

_ROOT = Path(__file__).parent.parent
RESULTS_DIR = _ROOT / "results"
DATA_PATH   = _ROOT / "data" / "cluster_data.gpkg"
RESULTS_DIR.mkdir(exist_ok=True)

H_MAX   = 20_000          # 20 km, metres
H_CUTS  = [1_000, 2_000, 5_000, 10_000, 20_000]  # 1–20 km
METHOD  = "single"
METRIC  = "haversine"

# ------------------------------------------------------------------
# Load dataset and extract (lon, lat) centroid coordinates
# ------------------------------------------------------------------

print("Loading cluster_data.gpkg …")
import geopandas as gpd

gdf = gpd.read_file(DATA_PATH)
n_total    = len(gdf)
n_polygons = (gdf.geom_type == "Polygon").sum()
n_points   = (gdf.geom_type == "Point").sum()
print(f"  n = {n_total:,}  ({n_polygons:,} polygons, {n_points:,} points)")

# Compute centroids; for points the centroid is the point itself.
centroids = gdf.geometry.centroid
coords = np.column_stack([centroids.x, centroids.y])   # (lon, lat), EPSG:4326
print(f"  Coordinate range: lon [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]"
      f"  lat [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def measure(fn):
    """Run fn(), return (result, elapsed_seconds, peak_memory_mib)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


# ------------------------------------------------------------------
# Phase 1: graph construction
# ------------------------------------------------------------------

print(f"\nPhase 1: building sparse distance graph "
      f"(h_max = {H_MAX/1000:.0f} km, metric = {METRIC}) …")

graph, t_graph, m_graph = measure(
    lambda: spatial_dist_graph(coords, H_MAX, metric=METRIC)
)

n_edges      = graph["n_edges"]
n_components = graph["n_components"]
comp_labels  = graph["components"]
mean_k       = round(2 * n_edges / n_total, 1)
density      = graph["density"]

print(f"  Edges:       {n_edges:,}  (mean k = {mean_k})")
print(f"  Components:  {n_components:,}")
print(f"  Time:        {t_graph:.1f} s")
print(f"  Peak memory: {m_graph:.0f} MiB")

# Component size distribution
comp_sizes = np.bincount(comp_labels)
print(f"\n  Component size distribution:")
print(f"    min    = {comp_sizes.min()}")
print(f"    median = {int(np.median(comp_sizes))}")
print(f"    mean   = {comp_sizes.mean():.1f}")
print(f"    p90    = {int(np.percentile(comp_sizes, 90))}")
print(f"    p99    = {int(np.percentile(comp_sizes, 99))}")
print(f"    max    = {comp_sizes.max()}")
print(f"    singletons (size=1): {(comp_sizes == 1).sum():,}")


# ------------------------------------------------------------------
# Phase 2: sparse HAC
# ------------------------------------------------------------------

print(f"\nPhase 2: sparse HAC (linkage = {METHOD}, "
      f"cut heights = {[h//1000 for h in H_CUTS]} km) …")

result, t_hac, m_hac = measure(
    lambda: sparse_hclust(graph, H_CUTS, method=METHOD)
)

print(f"  Time:        {t_hac:.1f} s")
print(f"  Peak memory: {m_hac:.0f} MiB")

t_total = t_graph + t_hac
m_total = m_graph + m_hac   # upper bound; actual peak measured separately

print(f"\n  Total sparse runtime:    {t_total:.1f} s  ({t_total/60:.1f} min)")
print(f"  Graph fraction of total: {t_graph/t_total*100:.0f}%")

# Cluster counts at each cut height
print("\n  Cluster counts at each cut height:")
for h in H_CUTS:
    n_clusters = int(np.unique(result["labels"][h]).size)
    print(f"    h = {h/1000:>4.0f} km  ->  {n_clusters:,} clusters")


# ------------------------------------------------------------------
# Analytical dense baseline (not run — infeasible)
# ------------------------------------------------------------------

dense_dist_pairs = int(n_total * (n_total - 1) / 2)
dense_matrix_gb  = n_total ** 2 * 8 / 1e9
sparse_matrix_mb = n_edges * 2 * 8 / 1e6   # symmetric, float64

print(f"\n  Dense baseline (analytical, not run):")
print(f"    Distance pairs required: {dense_dist_pairs:.2e}")
print(f"    Matrix memory:           {dense_matrix_gb:.0f} GB  (infeasible)")
print(f"  Sparse matrix memory:      {sparse_matrix_mb:.0f} MB")
print(f"  Memory reduction:          {dense_matrix_gb*1000/sparse_matrix_mb:.0f}×")


# ------------------------------------------------------------------
# Write results/benchmark_mining.csv
# ------------------------------------------------------------------

main_row = {
    "n":                  n_total,
    "n_polygons":         int(n_polygons),
    "n_points":           int(n_points),
    "h_max_km":           H_MAX / 1000,
    "metric":             METRIC,
    "linkage":            METHOD,
    "n_edges":            n_edges,
    "mean_k":             mean_k,
    "graph_density":      round(density, 8),
    "n_components":       n_components,
    "comp_size_min":      int(comp_sizes.min()),
    "comp_size_median":   int(np.median(comp_sizes)),
    "comp_size_mean":     round(comp_sizes.mean(), 1),
    "comp_size_p90":      int(np.percentile(comp_sizes, 90)),
    "comp_size_p99":      int(np.percentile(comp_sizes, 99)),
    "comp_size_max":      int(comp_sizes.max()),
    "n_singletons":       int((comp_sizes == 1).sum()),
    "graph_time_s":       round(t_graph, 2),
    "graph_mem_mib":      round(m_graph, 1),
    "hac_time_s":         round(t_hac, 2),
    "hac_mem_mib":        round(m_hac, 1),
    "total_time_s":       round(t_total, 2),
    "graph_time_frac":    round(t_graph / t_total, 3),
    "sparse_matrix_mb":   round(sparse_matrix_mb, 1),
    "dense_matrix_gb":    round(dense_matrix_gb, 1),
    "memory_reduction_x": round(dense_matrix_gb * 1000 / sparse_matrix_mb),
}

for h in H_CUTS:
    key = f"n_clusters_h{h//1000}km"
    main_row[key] = int(np.unique(result["labels"][h]).size)

out_main = RESULTS_DIR / "benchmark_mining.csv"
with open(out_main, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=main_row.keys())
    writer.writeheader()
    writer.writerow(main_row)
print(f"\nResults written to {out_main}")


# ------------------------------------------------------------------
# Write results/benchmark_mining_components.csv
# ------------------------------------------------------------------

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
comp_rows = []
for p in pcts:
    comp_rows.append({
        "percentile": p,
        "component_size": int(np.percentile(comp_sizes, p)),
    })
# Add mean and mode
comp_rows.append({"percentile": "mean", "component_size": round(comp_sizes.mean(), 2)})
unique, counts = np.unique(comp_sizes, return_counts=True)
mode_size = int(unique[counts.argmax()])
comp_rows.append({"percentile": "mode", "component_size": mode_size})

out_comp = RESULTS_DIR / "benchmark_mining_components.csv"
with open(out_comp, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["percentile", "component_size"])
    writer.writeheader()
    writer.writerows(comp_rows)
print(f"Component stats written to {out_comp}")


# ------------------------------------------------------------------
# Write results/benchmark_mining_clusters.csv
# ------------------------------------------------------------------

cluster_rows = []
for h in H_CUTS:
    labels_h = result["labels"][h]
    unique_labels, label_counts = np.unique(labels_h, return_counts=True)
    cluster_rows.append({
        "h_cut_km":       h / 1000,
        "n_clusters":     len(unique_labels),
        "cluster_size_min":    int(label_counts.min()),
        "cluster_size_median": int(np.median(label_counts)),
        "cluster_size_mean":   round(label_counts.mean(), 1),
        "cluster_size_max":    int(label_counts.max()),
        "n_singletons":        int((label_counts == 1).sum()),
    })

out_clust = RESULTS_DIR / "benchmark_mining_clusters.csv"
with open(out_clust, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cluster_rows[0].keys())
    writer.writeheader()
    writer.writerows(cluster_rows)
print(f"Cluster stats written to {out_clust}")

print("\nDone.")
