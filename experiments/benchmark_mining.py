"""
benchmark_mining.py

Actual timing and memory benchmark of the sparse HAC approach on the
real-world global mining dataset (cluster_data.gpkg).

The dense baseline is analytically infeasible at this n; this script measures
only the sparse approach, over N_REP repetitions (mean +/- sd).

Memory is reported in three ways, all in binary units (MiB):
  * graph_mem_mib   — peak during graph construction (phase 1)
  * hac_mem_mib     — HAC working memory (phase 2; the lean MST-path figure,
                      new allocations on top of the resident graph)
  * pipeline_mem_mib — peak over the whole coords->labels pipeline (the honest
                      end-to-end peak; for this dataset the graph phase dominates)

All three come from a single graph build per repetition via tracemalloc's
reset_peak(), so repetitions stay cheap and the numbers are mutually consistent.

Outputs (relative to repo root):
  results/benchmark_mining.csv            — timing, memory, graph stats
  results/benchmark_mining_components.csv — component size distribution
  results/benchmark_mining_clusters.csv   — cluster counts at each cut height
"""

from __future__ import annotations

import os
import sys
import time
import gc
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
H_CUTS  = [1_000, 2_000, 5_000, 10_000, 20_000]  # 1-20 km
METHOD  = "single"
METRIC  = "haversine"
N_REP   = 5

_MIB = 1024 ** 2
_GIB = 1024 ** 3


# ------------------------------------------------------------------
# Two-phase pipeline measurement (single build; graph + HAC + whole-run peaks)
# ------------------------------------------------------------------

def measure_pipeline(build_fn, run_fn):
    """Measure a build-then-run pipeline in ONE tracemalloc scope.

    Returns (graph, result, metrics) where metrics has, all in MiB / seconds:
      t_graph, t_hac, t_total,
      graph_mem    — phase-1 peak,
      hac_mem      — phase-2 working memory (new allocations on top of the
                     resident graph; matches a fresh-tracemalloc HAC measure),
      pipeline_mem — whole-run peak = max(phase-1 peak, phase-2 peak incl. graph).
    The caller owns `graph`/`result` and must del + gc.collect() afterwards.
    """
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    graph = build_fn()
    t_graph = time.perf_counter() - t0
    cur_after_graph, peak_graph = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()      # peak <- current (resident graph)
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
# Load dataset and extract (lon, lat) centroid coordinates
# ------------------------------------------------------------------

print("Loading cluster_data.gpkg ...")
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

del gdf, centroids
gc.collect()


# ------------------------------------------------------------------
# Repeated measurement (graph construction + sparse HAC)
# ------------------------------------------------------------------

print(f"\nMeasuring sparse HAC over {N_REP} repetitions "
      f"(h_max = {H_MAX/1000:.0f} km, metric = {METRIC}, linkage = {METHOD}) ...")

graph_times, hac_times, total_times = [], [], []
graph_mems, hac_mems, pipeline_mems = [], [], []

# Deterministic stats captured once (identical across reps for fixed input).
n_edges = n_components = density = None
comp_sizes = None
cluster_counts = {}

for rep in range(N_REP):
    graph, result, m = measure_pipeline(
        lambda: spatial_dist_graph(coords, H_MAX, metric=METRIC),
        lambda g: sparse_hclust(g, H_CUTS, method=METHOD),
    )

    graph_times.append(m["t_graph"])
    hac_times.append(m["t_hac"])
    total_times.append(m["t_total"])
    graph_mems.append(m["graph_mem"])
    hac_mems.append(m["hac_mem"])
    pipeline_mems.append(m["pipeline_mem"])

    if rep == 0:
        n_edges      = graph["n_edges"]
        n_components = graph["n_components"]
        density      = graph["density"]
        comp_sizes   = np.bincount(graph["components"])
        cluster_counts = {
            h: int(np.unique(result["labels"][h]).size) for h in H_CUTS
        }

    print(f"  rep {rep+1}/{N_REP}:  graph {m['t_graph']:.2f}s / {m['graph_mem']:.0f} MiB"
          f"  |  HAC {m['t_hac']:.2f}s / {m['hac_mem']:.0f} MiB"
          f"  |  pipeline peak {m['pipeline_mem']:.0f} MiB")

    # Critical: free both before the next repetition.
    del graph, result
    gc.collect()

mean_k = round(2 * n_edges / n_total, 1)

gt_mean, gt_sd = _mean_sd(graph_times)
ht_mean, ht_sd = _mean_sd(hac_times)
tt_mean, tt_sd = _mean_sd(total_times)
gm_mean, gm_sd = _mean_sd(graph_mems)
hm_mean, hm_sd = _mean_sd(hac_mems)
pm_mean, pm_sd = _mean_sd(pipeline_mems)

print(f"\n  Edges:        {n_edges:,}  (mean k = {mean_k})")
print(f"  Components:   {n_components:,}")
print(f"  Graph:        {gt_mean:.2f} +/- {gt_sd:.2f} s   |  peak {gm_mean:.0f} MiB")
print(f"  HAC:          {ht_mean:.2f} +/- {ht_sd:.2f} s   |  working {hm_mean:.0f} MiB")
print(f"  Total:        {tt_mean:.2f} +/- {tt_sd:.2f} s   |  pipeline peak {pm_mean:.0f} MiB")


# ------------------------------------------------------------------
# Analytical dense baseline (not run; binary units, full AND condensed)
# ------------------------------------------------------------------

dense_pairs            = int(n_total * (n_total - 1) / 2)
sparse_matrix_mib      = n_edges * 2 * 8 / _MIB           # symmetric, float64
dense_full_gib         = n_total ** 2 * 8 / _GIB          # full n x n, float64
dense_condensed_gib    = dense_pairs * 8 / _GIB           # condensed vector
reduction_full_x       = round(dense_full_gib * 1024 / sparse_matrix_mib)
reduction_condensed_x  = round(dense_condensed_gib * 1024 / sparse_matrix_mib)

print(f"\n  Dense baseline (analytical, not run):")
print(f"    Distance pairs:     {dense_pairs:.3e}")
print(f"    Full n^2 matrix:    {dense_full_gib:.0f} GiB")
print(f"    Condensed vector:   {dense_condensed_gib:.0f} GiB")
print(f"  Sparse graph storage: {sparse_matrix_mib:.0f} MiB")
print(f"  Reduction (full):     {reduction_full_x}x")
print(f"  Reduction (condensed):{reduction_condensed_x}x")


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
    "n_reps":             N_REP,
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
    # Timings (s) — mean / sd over reps
    "graph_time_s":       gt_mean,
    "graph_time_s_sd":    gt_sd,
    "hac_time_s":         ht_mean,
    "hac_time_s_sd":      ht_sd,
    "total_time_s":       tt_mean,
    "total_time_s_sd":    tt_sd,
    "graph_time_frac":    round(gt_mean / tt_mean, 3) if tt_mean else "",
    # Memory (MiB, binary) — mean / sd over reps
    "graph_mem_mib":      round(gm_mean, 1),
    "graph_mem_mib_sd":   round(gm_sd, 1),
    "hac_mem_mib":        round(hm_mean, 1),
    "hac_mem_mib_sd":     round(hm_sd, 1),
    "pipeline_mem_mib":   round(pm_mean, 1),
    "pipeline_mem_mib_sd": round(pm_sd, 1),
    # Analytical sizes (binary; full AND condensed dense baseline)
    "sparse_matrix_mib":        round(sparse_matrix_mib, 1),
    "dense_matrix_full_gib":    round(dense_full_gib, 1),
    "dense_matrix_condensed_gib": round(dense_condensed_gib, 1),
    "memory_reduction_full_x":      reduction_full_x,
    "memory_reduction_condensed_x": reduction_condensed_x,
}

for h in H_CUTS:
    main_row[f"n_clusters_h{h//1000}km"] = cluster_counts[h]

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
# Cluster size distribution per cut height is recomputed from a single
# clustering pass (deterministic for fixed input).

graph = spatial_dist_graph(coords, H_MAX, metric=METRIC)
result = sparse_hclust(graph, H_CUTS, method=METHOD)
cluster_rows = []
for h in H_CUTS:
    labels_h = result["labels"][h]
    unique_labels, label_counts = np.unique(labels_h, return_counts=True)
    cluster_rows.append({
        "h_cut_km":            h / 1000,
        "n_clusters":          len(unique_labels),
        "cluster_size_min":    int(label_counts.min()),
        "cluster_size_median": int(np.median(label_counts)),
        "cluster_size_mean":   round(label_counts.mean(), 1),
        "cluster_size_max":    int(label_counts.max()),
        "n_singletons":        int((label_counts == 1).sum()),
    })
del graph, result
gc.collect()

out_clust = RESULTS_DIR / "benchmark_mining_clusters.csv"
with open(out_clust, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cluster_rows[0].keys())
    writer.writeheader()
    writer.writerows(cluster_rows)
print(f"Cluster stats written to {out_clust}")

print("\nDone.")
