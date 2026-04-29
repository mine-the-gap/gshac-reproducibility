"""
explore_geonames.py

Downloads GeoNames allCountries (~400 MB) and explores whether the
sparse HAC approach is feasible at n ~ 1M–12M features.

Phases:
  1. Download + parse GeoNames (lat/lon only)
  2. Build sparse distance graph at h_max = 1 km and 2 km
  3. Report component statistics and memory/time figures
  4. Estimate HAC feasibility from component size distribution

Does NOT run HAC — this is an exploration script only.

Output: results/explore_geonames.txt (summary) + console output.
"""

from __future__ import annotations

import os, sys, time, tracemalloc, urllib.request
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ZIP_URL   = "https://download.geonames.org/export/dump/allCountries.zip"
ZIP_PATH  = DATA_DIR / "geonames_allCountries.zip"
TXT_PATH  = DATA_DIR / "allCountries.txt"

H_MAX_VALS = [1_000, 2_000]   # metres
FEATURE_CLASSES = None              # None = all; or e.g. {"P", "S"} to filter


# ------------------------------------------------------------------
# 1. Download
# ------------------------------------------------------------------

def download_geonames():
    if TXT_PATH.exists():
        print(f"  Using cached {TXT_PATH}  ({TXT_PATH.stat().st_size/1e6:.0f} MB)")
        return
    if not ZIP_PATH.exists():
        print(f"  Downloading {ZIP_URL} ...")
        def progress(count, block, total):
            pct = count * block / total * 100
            print(f"\r  {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=progress)
        print()
        print(f"  Saved to {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")
    print(f"  Extracting ...")
    import zipfile
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract("allCountries.txt", DATA_DIR)
    print(f"  Extracted to {TXT_PATH}  ({TXT_PATH.stat().st_size/1e6:.0f} MB)")


# ------------------------------------------------------------------
# 2. Parse
# ------------------------------------------------------------------

def load_coords(feature_classes=None):
    """
    Read GeoNames TSV, return (n, 2) float32 array of (lon, lat).
    GeoNames columns (0-indexed):
      0  geonameid   4  latitude   5  longitude   6  feature_class
    Uses pandas for fast vectorised parsing (avoids slow Python csv loop).
    """
    import pandas as pd
    print(f"\nParsing {TXT_PATH} ...")
    t0 = time.perf_counter()
    # Read only the three columns we need; skip broken lines quietly
    df = pd.read_csv(
        TXT_PATH,
        sep="\t",
        header=None,
        usecols=[4, 5, 6],
        names=["lat", "lon", "fc"],
        dtype={"lat": "float32", "lon": "float32", "fc": "str"},
        on_bad_lines="skip",
        engine="c",
    )
    if feature_classes:
        df = df[df["fc"].isin(feature_classes)]
    elapsed = time.perf_counter() - t0
    coords = df[["lon", "lat"]].to_numpy(dtype=np.float32)
    print(f"  Loaded {len(coords):,} features in {elapsed:.1f} s")
    print("  Feature class breakdown:")
    for fc, cnt in df["fc"].value_counts().head(10).items():
        print(f"    {fc}: {cnt:,}")
    return coords


# ------------------------------------------------------------------
# 3. Graph + component stats (no HAC)
# ------------------------------------------------------------------

def measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024**2)


def explore_graph(coords, h_max_m):
    n = len(coords)
    print(f"\n--- h_max = {h_max_m:,} m  (n = {n:,}) ---")

    print("  Building sparse distance graph (haversine) ...")
    graph, t_graph, mem_graph = measure(
        lambda: spatial_dist_graph(coords, h_max_m, metric="haversine")
    )

    n_edges      = graph["n_edges"]
    n_components = graph["n_components"]
    comp_labels  = graph["components"]
    mean_k       = 2 * n_edges / n
    comp_sizes   = np.bincount(comp_labels)

    sparse_mb = n_edges * 2 * 8 / 1e6
    dense_gb  = n**2 * 8 / 1e9

    print(f"  Graph construction:  {t_graph:.1f} s  |  peak {mem_graph:.0f} MiB")
    print(f"  Edges:               {n_edges:,}  (mean k = {mean_k:.1f})")
    print(f"  Sparse storage:      {sparse_mb:.0f} MB")
    print(f"  Dense baseline:      {dense_gb:.0f} GB  (infeasible)")
    print(f"  Memory reduction:    {dense_gb*1000/sparse_mb:.0f}×")
    print(f"  Components:          {n_components:,}")
    print(f"  Component sizes:")
    for p in [50, 90, 95, 99, 100]:
        print(f"    p{p:3d}: {int(np.percentile(comp_sizes, p)):,}")
    print(f"    singletons: {(comp_sizes==1).sum():,}  ({(comp_sizes==1).mean()*100:.1f}%)")

    # HAC feasibility estimate
    max_c    = comp_sizes.max()
    max_submat_gb = max_c**2 * 8 / 1e9
    print(f"\n  HAC feasibility:")
    print(f"    Largest component: {max_c:,} features")
    print(f"    Its dense sub-matrix: {max_submat_gb:.1f} GB")
    if max_submat_gb < 8:
        verdict = "likely feasible on 30 GiB workstation"
    elif max_submat_gb < 25:
        verdict = "tight — may need >30 GiB RAM for largest component"
    else:
        verdict = "NOT feasible — largest component sub-matrix exceeds 30 GiB"
    print(f"    Verdict: {verdict}")

    return {
        "h_max_m": h_max_m, "n": n, "n_edges": n_edges,
        "mean_k": round(mean_k, 1), "sparse_mb": round(sparse_mb, 1),
        "dense_gb": round(dense_gb, 1),
        "n_components": n_components,
        "comp_p50": int(np.percentile(comp_sizes, 50)),
        "comp_p90": int(np.percentile(comp_sizes, 90)),
        "comp_p99": int(np.percentile(comp_sizes, 99)),
        "comp_max": int(comp_sizes.max()),
        "n_singletons": int((comp_sizes==1).sum()),
        "graph_time_s": round(t_graph, 1),
        "graph_mem_mib": round(mem_graph, 0),
        "max_submat_gb": round(max_submat_gb, 1),
        "hac_verdict": verdict,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

print("=" * 60)
print("GeoNames feasibility exploration")
print("=" * 60)

download_geonames()
coords = load_coords(feature_classes=FEATURE_CLASSES)

print(f"\nCoordinate range: lon [{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]"
      f"  lat [{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")

results = []
for h_max in H_MAX_VALS:
    r = explore_graph(coords, h_max)
    results.append(r)

# Write summary
import csv
out = RESULTS_DIR / "explore_geonames.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nSummary written to {out}")
