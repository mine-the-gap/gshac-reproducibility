"""
benchmark_hdbscan.py

Fair resource comparison of sklearn HDBSCAN vs GSHAC sparse single-linkage HAC
on the global mining dataset (n = 261,073), plus an O(n^2) scaling probe used to
extrapolate HDBSCAN to the 2M-point GeoNames scale (a real run is ~hours).

Fairness: BOTH methods are measured in this one script, over the identical scope
(coordinates -> cluster labels), with the SAME peak-memory definition — the peak
of a single tracemalloc scope covering the whole pipeline. For GSHAC that peak
necessarily includes graph construction, so we ALSO break out the graph-phase and
HAC-phase peaks (the lean MST-path figure) for transparency. All memory is in
binary MiB. Each method is repeated N_REP times (mean +/- sd).

HDBSCAN and GSHAC solve different problems (density-based with noise vs exact
hierarchical at a distance threshold); this table reports resource profiles only.

Outputs:
  results/benchmark_hdbscan.csv          — mining: HDBSCAN (mcs 2,5) vs GSHAC
  results/benchmark_hdbscan_scaling.csv  — HDBSCAN O(n^2) probe + fit + extrapolation
"""

from __future__ import annotations

import sys, time, gc, tracemalloc, csv, warnings
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_REP          = 5       # GSHAC reps (cheap)
# HDBSCAN fits are ~25 min each on mining. A single run suffices: its variance is
# small relative to the >100x speed margin, and fairness here is about using the
# same whole-pipeline peak definition for both methods, not equal rep counts.
# The asymmetry is disclosed in the table footnote. Raise if you want a token sd.
N_REP_HDBSCAN  = 1
H_MAX          = 20_000
H_CUTS         = [1_000, 5_000, 10_000, 20_000]
HDB_MCS        = [2, 5]
GEONAMES_N     = 2_000_000   # extrapolation target for the scaling probe
PROBE_N        = [1_000, 2_000, 5_000, 10_000, 25_000]
PROBE_MCS      = 5

_MIB = 1024 ** 2
_GIB = 1024 ** 3


# ------------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------------

def measure(fn):
    """Whole-scope time + peak (MiB)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / _MIB


def measure_pipeline(build_fn, run_fn):
    """Build-then-run in ONE tracemalloc scope (single build).
    Returns (graph, result, metrics) with graph-phase peak, HAC working memory,
    and the whole-run peak — see benchmark_mining.py for the rationale."""
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
        "t_graph": t_graph, "t_hac": t_hac, "t_total": t_graph + t_hac,
        "graph_mem": peak_graph / _MIB,
        "hac_mem": (peak_during_hac - cur_after_graph) / _MIB,
        "pipeline_mem": max(peak_graph, peak_during_hac) / _MIB,
    }
    return graph, result, metrics


def _mean_sd(xs):
    if len(xs) < 2:
        return round(float(np.mean(xs)), 3), 0.0
    return round(float(np.mean(xs)), 3), round(float(np.std(xs, ddof=1)), 3)


# ------------------------------------------------------------------
# Data + runners
# ------------------------------------------------------------------

def load_mining():
    import geopandas as gpd
    gdf = gpd.read_file(DATA_DIR / "cluster_data.gpkg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = gdf.geometry.centroid
    coords = np.column_stack([cx.x.to_numpy(), cx.y.to_numpy()]).astype(np.float64)
    print(f"  Mining: {len(coords):,} features")
    return coords


def run_hdbscan_once(coords_lonlat, mcs):
    """One HDBSCAN fit (haversine, ball_tree). Returns (time, peak_mib,
    n_clusters, n_noise)."""
    from sklearn.cluster import HDBSCAN
    coords_rad = np.radians(coords_lonlat[:, [1, 0]])  # (lat, lon) radians

    def _fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = HDBSCAN(min_cluster_size=mcs, metric="haversine",
                          algorithm="ball_tree")
            clf.fit(coords_rad)
        return clf

    clf, elapsed, peak = measure(_fit)
    n_noise    = int((clf.labels_ == -1).sum())
    n_clusters = int(clf.labels_.max()) + 1
    return elapsed, peak, n_clusters, n_noise


def run_gshac_once(coords_lonlat, h_max_m, h_cuts):
    """One GSHAC pipeline (graph + single-linkage HAC, haversine). Returns
    (metrics, cluster_counts) where metrics carries phase + pipeline peaks."""
    coords = coords_lonlat.astype(np.float64)
    graph, result, m = measure_pipeline(
        lambda: spatial_dist_graph(coords, h_max_m, metric="haversine"),
        lambda g: sparse_hclust(g, h_cuts, method="single"),
    )
    counts = {h: int(np.unique(result["labels"][h]).size) for h in h_cuts}
    del graph, result
    gc.collect()
    return m, counts


# Flat CSV schema shared by HDBSCAN and GSHAC rows.
_FIELDS = ["dataset", "method", "min_cluster_size", "h_max_m", "n", "n_reps",
           "time_s", "time_s_sd", "pipeline_mem_mib", "pipeline_mem_mib_sd",
           "graph_time_s", "hac_time_s", "graph_mem_mib", "hac_mem_mib",
           "n_clusters", "n_noise", "noise_pct", "note"]


def _row(**kw):
    return {f: kw.get(f, "") for f in _FIELDS}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HDBSCAN vs GSHAC — fair resource comparison (mining)")
    print("=" * 60)

    coords = load_mining()
    n = len(coords)
    rows = []

    # ---- HDBSCAN (mcs = 2, 5), N_REP_HDBSCAN reps ----
    for mcs in HDB_MCS:
        print(f"\nHDBSCAN (mcs={mcs}), {N_REP_HDBSCAN} reps "
              f"(~25 min/rep on mining) ...")
        times, mems = [], []
        nc = nn = None
        for rep in range(N_REP_HDBSCAN):
            t, peak, c, noise = run_hdbscan_once(coords, mcs)
            times.append(t); mems.append(peak)
            if rep == 0:
                nc, nn = c, noise
            print(f"  rep {rep+1}/{N_REP_HDBSCAN}: {t:.1f} s | {peak:.0f} MiB "
                  f"| {c:,} clusters | {noise:,} noise")
            gc.collect()
        t_mean, t_sd = _mean_sd(times)
        m_mean, m_sd = _mean_sd(mems)
        rows.append(_row(
            dataset="mining", method="HDBSCAN", min_cluster_size=mcs, n=n,
            n_reps=N_REP_HDBSCAN, time_s=t_mean, time_s_sd=t_sd,
            pipeline_mem_mib=round(m_mean, 1), pipeline_mem_mib_sd=round(m_sd, 1),
            n_clusters=nc, n_noise=nn, noise_pct=round(nn / n * 100, 1),
            note="sklearn HDBSCAN, haversine, ball_tree; peak = whole-fit",
        ))

    # ---- GSHAC sparse single linkage, N_REP reps ----
    print(f"\nGSHAC sparse single linkage (h_max={H_MAX/1000:.0f} km), {N_REP} reps ...")
    g_t, g_pipe = [], []
    g_graph_t, g_hac_t, g_graph_m, g_hac_m = [], [], [], []
    counts = None
    for rep in range(N_REP):
        m, counts = run_gshac_once(coords, H_MAX, H_CUTS)
        g_t.append(m["t_total"]);   g_pipe.append(m["pipeline_mem"])
        g_graph_t.append(m["t_graph"]); g_hac_t.append(m["t_hac"])
        g_graph_m.append(m["graph_mem"]); g_hac_m.append(m["hac_mem"])
        print(f"  rep {rep+1}/{N_REP}: total {m['t_total']:.1f} s "
              f"| pipeline peak {m['pipeline_mem']:.0f} MiB "
              f"(graph {m['graph_mem']:.0f} / HAC {m['hac_mem']:.0f})")
    t_mean, t_sd = _mean_sd(g_t)
    p_mean, p_sd = _mean_sd(g_pipe)
    rows.append(_row(
        dataset="mining", method="GSHAC_sparse_single", h_max_m=H_MAX, n=n,
        n_reps=N_REP, time_s=t_mean, time_s_sd=t_sd,
        pipeline_mem_mib=round(p_mean, 1), pipeline_mem_mib_sd=round(p_sd, 1),
        graph_time_s=round(float(np.mean(g_graph_t)), 3),
        hac_time_s=round(float(np.mean(g_hac_t)), 3),
        graph_mem_mib=round(float(np.mean(g_graph_m)), 1),
        hac_mem_mib=round(float(np.mean(g_hac_m)), 1),
        n_clusters=counts[H_MAX], n_noise=0, noise_pct=0.0,
        note="GSHAC single linkage, haversine; peak = whole pipeline (incl. graph)",
    ))

    out = RESULTS_DIR / "benchmark_hdbscan.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nComparison written to {out}")

    # ------------------------------------------------------------------
    # HDBSCAN O(n^2) scaling probe (subsamples of mining) + extrapolation
    # ------------------------------------------------------------------
    print(f"\nHDBSCAN scaling probe (mcs={PROBE_MCS}) on mining subsamples ...")
    rng = np.random.default_rng(42)
    probe = []
    for pn in PROBE_N:
        idx = rng.choice(n, size=pn, replace=False)
        t, _, _, _ = run_hdbscan_once(coords[idx], PROBE_MCS)
        probe.append({"n": pn, "time_s": round(t, 4)})
        print(f"  n={pn:>6,}: {t:.3f} s")
        gc.collect()

    ns = np.array([p["n"] for p in probe], dtype=float)
    ts = np.array([p["time_s"] for p in probe], dtype=float)
    slope, intercept = np.polyfit(np.log(ns), np.log(ts), 1)
    t_geonames = float(np.exp(intercept) * GEONAMES_N ** slope)
    print(f"  log-log slope = {slope:.2f}")
    print(f"  extrapolated HDBSCAN @ n={GEONAMES_N:,}: "
          f"{t_geonames:.0f} s (~{t_geonames/3600:.1f} h)")

    scaling_rows = [
        {"n": int(p["n"]), "mcs": PROBE_MCS, "time_s": p["time_s"],
         "kind": "measured", "slope": ""} for p in probe
    ]
    scaling_rows.append({
        "n": GEONAMES_N, "mcs": PROBE_MCS, "time_s": round(t_geonames, 1),
        "kind": "extrapolated_geonames_2M", "slope": round(float(slope), 3),
    })
    out_scaling = RESULTS_DIR / "benchmark_hdbscan_scaling.csv"
    with open(out_scaling, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "mcs", "time_s", "kind", "slope"])
        w.writeheader()
        w.writerows(scaling_rows)
    print(f"Scaling probe written to {out_scaling}")
