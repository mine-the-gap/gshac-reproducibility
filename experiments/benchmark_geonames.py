"""
benchmark_geonames.py

GeoNames threshold sweep: the 2M spatially-stratified sample at
h_max = 1 / 5 / 10 / 20 km, single-linkage HAC. Reports time and peak memory
(graph-phase, HAC-phase, whole-pipeline), component structure, and the
analytical dense sizes the MST path avoids.

Each (threshold, rep) measurement runs in a FRESH SUBPROCESS so the OS reclaims
all memory between reps. This is required at 20 km: that graph peaks at ~28 GiB,
and five *in-process* builds accumulate enough retained/fragmented memory to OOM
a 30 GiB workstation (the old single-pass code built each graph once and fit).
The 2M sample is also drawn in its own subprocess so the parser's transient
memory is released before the heavy workers run, and cached as a temp .npy.

Self-contained: downloads allCountries.txt on first run.

Modes:
  sample: python benchmark_geonames.py --sample <npy>
  worker: python benchmark_geonames.py --worker <npy> <h_max> <h_cut>
  main:   python benchmark_geonames.py

Output: results/benchmark_geonames.csv
"""

from __future__ import annotations

import sys, os, json, time, gc, tracemalloc, csv, subprocess, urllib.request, zipfile
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / ("results_fast" if os.environ.get("GSHAC_FAST") == "1" else "results")
TXT_PATH = DATA_DIR / "allCountries.txt"
ZIP_PATH = DATA_DIR / "geonames_allCountries.zip"
ZIP_URL = "https://download.geonames.org/export/dump/allCountries.zip"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Set GSHAC_FAST=1 for a reduced-scale reproducibility check (see make reproduce-fast).
# N_SAMPLE is imported by the two scaling scripts, so this fast value propagates there.
FAST        = os.environ.get("GSHAC_FAST") == "1"
N_SAMPLE    = 200_000 if FAST else 2_000_000
N_FULL_REF  = 13_412_837
RANDOM_SEED = 42
H_MAX_VALS  = [1_000, 5_000] if FAST else [1_000, 5_000, 10_000, 20_000]   # metres
N_REP       = 1 if FAST else 5

_MIB = 1024 ** 2
_GIB = 1024 ** 3


# ------------------------------------------------------------------
# Download / parse / sample  (also imported by benchmark_geonames_scaling.py)
# ------------------------------------------------------------------
def download_geonames():
    if TXT_PATH.exists():
        print(f"  Using cached {TXT_PATH}  ({TXT_PATH.stat().st_size/_MIB:.0f} MiB)")
        return
    if not ZIP_PATH.exists():
        print(f"  Downloading {ZIP_URL} ...")
        def progress(count, block, total):
            print(f"\r  {count*block/total*100:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=progress)
        print()
    print("  Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract("allCountries.txt", DATA_DIR)
    print(f"  Extracted to {TXT_PATH}  ({TXT_PATH.stat().st_size/_MIB:.0f} MiB)")


COORDS_CACHE = DATA_DIR / "geonames_coords.npy"  # parsed (lat, lon); skips the ~1.7 GB re-parse


def load_coords():
    import pandas as pd
    # Fast path: reuse the cached parse if it is at least as new as the source dump.
    if COORDS_CACHE.exists() and COORDS_CACHE.stat().st_mtime >= TXT_PATH.stat().st_mtime:
        arr = np.load(COORDS_CACHE)
        print(f"  Loaded {len(arr):,} features from cache {COORDS_CACHE.name} "
              f"(skipped the {TXT_PATH.stat().st_size/_GIB:.1f} GiB re-parse)")
        return pd.DataFrame({"lat": arr[:, 0], "lon": arr[:, 1]})
    print(f"Parsing {TXT_PATH} ...")
    t0 = time.perf_counter()
    df = pd.read_csv(
        TXT_PATH, sep="\t", header=None, usecols=[4, 5],
        names=["lat", "lon"], dtype={"lat": "float32", "lon": "float32"},
        on_bad_lines="skip", engine="c",
    )
    print(f"  {len(df):,} features in {time.perf_counter()-t0:.1f} s")
    np.save(COORDS_CACHE, df[["lat", "lon"]].to_numpy(dtype=np.float32))
    print(f"  Cached parse -> {COORDS_CACHE.name} ({COORDS_CACHE.stat().st_size/_MIB:.0f} MiB)")
    return df


def stratified_sample(df, n_sample, seed=42):
    """Draw n_sample rows using 1 deg x 1 deg grid cells as strata with
    proportional (largest-remainder) allocation. Returns (n_sample, 2)
    float32 array of (lon, lat)."""
    rng = np.random.default_rng(seed)
    n_full = len(df)
    cell_id = (np.floor(df["lon"].to_numpy()).astype(np.int32) * 200
               + np.floor(df["lat"].to_numpy()).astype(np.int32))
    sort_idx   = np.argsort(cell_id, kind="stable")
    sorted_c   = cell_id[sort_idx]
    _, g_start = np.unique(sorted_c, return_index=True)
    g_end      = np.append(g_start[1:], n_full)
    g_sizes    = g_end - g_start
    n_cells    = len(g_start)
    alloc_f    = g_sizes / n_full * n_sample
    alloc      = np.floor(alloc_f).astype(np.int64)
    shortfall  = int(n_sample - alloc.sum())
    topup      = np.argsort(alloc_f - alloc)[::-1][:shortfall]
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
# Worker: one isolated measurement at a single threshold (full 2M sample)
# ------------------------------------------------------------------
def _worker(npy_path, h_max, h_cut):
    from gshac.spatial_dist_graph import spatial_dist_graph
    from gshac.sparse_hclust import sparse_hclust

    coords = np.load(npy_path)
    n = len(coords)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    g = spatial_dist_graph(coords, h_max, metric="haversine")
    t_graph = time.perf_counter() - t0
    cur, peak_g = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    t1 = time.perf_counter()
    r = sparse_hclust(g, [h_cut], method="single")
    t_hac = time.perf_counter() - t1
    _, peak_h = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cs = np.bincount(g["components"])
    out = dict(
        n=n, h_max_m=h_max, n_edges=int(g["n_edges"]),
        mean_k=round(2 * g["n_edges"] / n, 2),
        n_components=int(g["n_components"]),
        comp_p50=int(np.percentile(cs, 50)),
        comp_p90=int(np.percentile(cs, 90)),
        comp_p99=int(np.percentile(cs, 99)),
        comp_max=int(cs.max()), n_singletons=int((cs == 1).sum()),
        n_clusters=int(np.unique(r["labels"][h_cut]).size),
        t_graph=t_graph, t_hac=t_hac, t_total=t_graph + t_hac,
        graph_mem=peak_g / _MIB, hac_mem=(peak_h - cur) / _MIB,
        pipeline_mem=max(peak_g, peak_h) / _MIB,
    )
    print("RESULT " + json.dumps(out), flush=True)


# ------------------------------------------------------------------
# Subprocess-mode dispatch (must run before the __main__ orchestrator)
# ------------------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "--sample":
    download_geonames()
    _df = load_coords()
    _coords = stratified_sample(_df, N_SAMPLE, seed=RANDOM_SEED).astype(np.float32)
    np.save(sys.argv[2], _coords)
    print(f"SAMPLED {len(_coords)}")
    sys.exit(0)


def _agg(reps, key):
    vals = [r[key] for r in reps]
    mean = round(float(np.mean(vals)), 3)
    sd = round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else 0.0
    return mean, sd


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print(f"GeoNames threshold sweep (2M sample, {N_REP} reps/threshold, "
          f"subprocess-isolated)")
    print(f"  thresholds = {[h//1000 for h in H_MAX_VALS]} km")
    print("=" * 64)

    npy = str(RESULTS_DIR / "_geonames_coords_tmp.npy")
    print("Drawing 2M stratified sample in a subprocess ...")
    cp = subprocess.run([sys.executable, __file__, "--sample", npy], text=True)
    if cp.returncode != 0 or not os.path.exists(npy):
        print("Sampling subprocess failed.")
        sys.exit(1)

    results = []
    for h_max in H_MAX_VALS:
        h_cut = h_max
        reps = []
        for rep in range(N_REP):
            cp = subprocess.run(
                [sys.executable, __file__, "--worker", npy, str(h_max), str(h_cut)],
                capture_output=True, text=True,
            )
            line = next((l for l in cp.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if line is None:
                tail = (cp.stderr or cp.stdout)[-400:]
                print(f"  h_max={h_max/1000:.0f}km rep {rep+1}: WORKER FAILED "
                      f"(rc={cp.returncode})\n  {tail}")
                continue
            m = json.loads(line[len("RESULT "):])
            reps.append(m)
            print(f"  h_max={h_max/1000:>2.0f}km rep {rep+1}/{N_REP}: "
                  f"graph {m['t_graph']:.1f}s/{m['graph_mem']:.0f}MiB  "
                  f"HAC {m['t_hac']:.1f}s/{m['hac_mem']:.0f}MiB  "
                  f"pipeline {m['pipeline_mem']:.0f}MiB")
        if not reps:
            print(f"  h_max={h_max}: all reps failed — skipping")
            continue

        d = reps[0]
        n, n_edges, max_c = d["n"], d["n_edges"], d["comp_max"]
        sparse_mib       = n_edges * 2 * 8 / _MIB
        # Analytical dense-matrix sizes (float64 = 8 bytes/entry). The dense baseline is
        # NEVER run at these n -- it cannot fit in RAM -- so these are the *hypothetical*
        # requirements behind the paper's infeasibility claims, not measurements. Two forms:
        #   full square = n^2 * 8            -- all n^2 pairwise distances ("the dense matrix")
        #   condensed   = n*(n-1)/2 * 8      -- upper triangle: what an optimised dense HAC
        #                                       (fastcluster/scipy) actually allocates
        # The same two formulas applied to the largest component (max_c) give the per-component
        # "max sub-matrix" a non-single linkage would materialise. Every TiB/GiB figure in the
        # manuscript is reproducible from these formulas and n / max_c alone.
        dense_full_gib   = n ** 2 * 8 / _GIB
        dense_cond_gib   = n * (n - 1) / 2 * 8 / _GIB
        max_sub_full_gib = max_c ** 2 * 8 / _GIB
        max_sub_cond_gib = max_c * (max_c - 1) / 2 * 8 / _GIB

        row = {
            "n_sample": n, "n_full": N_FULL_REF, "h_max_m": h_max, "h_cut_m": h_cut,
            "metric": "haversine", "linkage": "single", "n_reps": len(reps),
            "n_edges": n_edges, "mean_k": d["mean_k"],
            "sparse_mib": round(sparse_mib, 1),
            "dense_full_gib_sample": round(dense_full_gib, 1),
            "dense_condensed_gib_sample": round(dense_cond_gib, 1),
            "n_components": d["n_components"], "comp_p50": d["comp_p50"],
            "comp_p90": d["comp_p90"], "comp_p99": d["comp_p99"],
            "comp_max": max_c, "n_singletons": d["n_singletons"],
            "max_sub_full_gib": round(max_sub_full_gib, 2),
            "max_sub_condensed_gib": round(max_sub_cond_gib, 2),
        }
        for k, name in [("t_graph", "graph_time_s"), ("t_hac", "hac_time_s"),
                        ("t_total", "total_time_s"), ("graph_mem", "graph_mem_mib"),
                        ("hac_mem", "hac_mem_mib"), ("pipeline_mem", "pipeline_mem_mib")]:
            mean, sd = _agg(reps, k)
            row[name] = mean
            row[name + "_sd"] = sd
        row["n_clusters"] = d["n_clusters"]
        results.append(row)

        # Write incrementally so a later-threshold failure never loses earlier ones.
        out = RESULTS_DIR / "benchmark_geonames.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    print(f"\nResults written to {RESULTS_DIR / 'benchmark_geonames.csv'}")
    try:
        os.remove(npy)
    except OSError:
        pass
    print("Done.")
