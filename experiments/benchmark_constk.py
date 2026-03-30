"""
benchmark_constk.py

Constant-k scaling experiment: domain grows with sqrt(n) so that the
average neighbourhood size k stays approximately constant as n increases.

This simulates real-world scaling where larger datasets cover larger
geographic areas (e.g., national → continental → global inventories),
rather than packing more points into a fixed area.

Under constant k, GSHAC scales as O(n k log n) ≈ O(n log n) in time
and O(n k) ≈ O(n) in memory, whereas the dense baseline remains O(n²).

Outputs:
  results/benchmark_constk.csv
  figures/fig_constk.pdf
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import csv
import gc
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

H_MAX       = 10_000        # 10 km
K_TARGET    = 50            # target average neighbourhood size
METHOD      = "single"
N_REP       = 5
N_SIZES     = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]


def domain_for_k(n, h_max, k_target):
    """Domain size (m) that yields approximately k_target neighbours.

    For uniform random points in a square domain of side L, the expected
    number of neighbours within distance r is k = n * pi * r^2 / L^2.
    Solving for L: L = sqrt(n * pi * r^2 / k).
    """
    return math.sqrt(n * math.pi * h_max**2 / k_target)


def dense_feasible(n, max_gb=6.0):
    return n * (n - 1) / 2 * 8 / 1e9 <= max_gb


def generate_uniform_points(n_total, domain_m, seed=42):
    """Uniform random points — k depends only on global density."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, domain_m, size=(n_total, 2))


def measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak / (1024 ** 2)


# ------------------------------------------------------------------
# Run benchmark
# ------------------------------------------------------------------

out_path = os.path.join(RESULTS_DIR, "benchmark_constk.csv")
f_out = open(out_path, "w", newline="")
writer = None
rows = []

h_max_km = H_MAX / 1000
h_cuts = [H_MAX // 2, H_MAX]  # two cut heights

print(f"Constant-k scaling experiment")
print(f"  h_max = {h_max_km:.0f} km, target k ≈ {K_TARGET}")
print(f"  n sizes: {N_SIZES}")
print(f"  Dense feasible up to n = {max(n for n in N_SIZES if dense_feasible(n)):,}")
print()

for n in N_SIZES:
    domain_m = domain_for_k(n, H_MAX, K_TARGET)
    run_dense = dense_feasible(n)
    tag = "" if run_dense else " [sparse only]"
    print(f"  n = {n:>7,}  domain = {domain_m/1000:>7.1f} km{tag}", end="", flush=True)

    pts = generate_uniform_points(n, domain_m)

    dense_times, dense_mems = [], []
    sparse_times, sparse_mems = [], []
    graph_times, hac_times = [], []
    n_edges_val = mean_k_val = None
    dt = dm = None

    for rep in range(N_REP):
        if run_dense:
            _, dt, dm = measure(lambda: dense_hclust(pts, h_cuts, METHOD))
            dense_times.append(dt)
            dense_mems.append(dm)
            gc.collect()

        graph, gt, _ = measure(lambda: spatial_dist_graph(pts, H_MAX))
        _, ht, _ = measure(lambda: sparse_hclust(graph, h_cuts, METHOD))
        graph_times.append(gt)
        hac_times.append(ht)
        sparse_times.append(gt + ht)

        if rep == 0:
            n_edges_val = graph["n_edges"]
            mean_k_val = round(2 * n_edges_val / n, 1)

        del graph
        gc.collect()

        def run_sparse_mem():
            g2 = spatial_dist_graph(pts, H_MAX)
            res = sparse_hclust(g2, h_cuts, METHOD)
            return res

        _, _, sm = measure(run_sparse_mem)
        sparse_mems.append(sm)
        gc.collect()

    row = {
        "n":                n,
        "domain_km":        round(domain_m / 1000, 1),
        "h_max_km":         h_max_km,
        "k_target":         K_TARGET,
        "mean_k":           mean_k_val,
        "n_edges":          n_edges_val,
        "dense_time_mean":  round(np.mean(dense_times), 4) if run_dense else "",
        "dense_time_sd":    round(np.std(dense_times, ddof=1), 4) if run_dense else "",
        "dense_mem_mean":   round(np.mean(dense_mems), 2) if run_dense else "",
        "dense_mem_sd":     round(np.std(dense_mems, ddof=1), 2) if run_dense else "",
        "sparse_time_mean": round(np.mean(sparse_times), 4),
        "sparse_time_sd":   round(np.std(sparse_times, ddof=1), 4),
        "sparse_mem_mean":  round(np.mean(sparse_mems), 2),
        "sparse_mem_sd":    round(np.std(sparse_mems, ddof=1), 2),
        "graph_time_mean":  round(np.mean(graph_times), 4),
        "hac_time_mean":    round(np.mean(hac_times), 4),
        "speedup_time":     round(np.mean(dense_times) / np.mean(sparse_times), 2) if run_dense else "",
        "speedup_mem":      round(np.mean(dense_mems) / np.mean(sparse_mems), 2) if run_dense else "",
    }
    rows.append(row)

    if writer is None:
        writer = csv.DictWriter(f_out, fieldnames=row.keys())
        writer.writeheader()
    writer.writerow(row)
    f_out.flush()

    if run_dense:
        print(f"  k={mean_k_val:>5}  dense={row['dense_time_mean']:.3f}s/{row['dense_mem_mean']:.0f}MiB"
              f"  sparse={row['sparse_time_mean']:.3f}s/{row['sparse_mem_mean']:.0f}MiB"
              f"  speedup={row['speedup_time']}x time, {row['speedup_mem']}x mem")
    else:
        print(f"  k={mean_k_val:>5}  sparse={row['sparse_time_mean']:.3f}s/{row['sparse_mem_mean']:.0f}MiB")

    del pts
    gc.collect()

f_out.close()
print(f"\nResults written to {out_path}")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(10, 4))

ns       = [r["n"] for r in rows]
sp_times = [r["sparse_time_mean"] for r in rows]
sp_mems  = [r["sparse_mem_mean"] for r in rows]
dn_ns    = [r["n"] for r in rows if r["dense_time_mean"] != ""]
dn_times = [r["dense_time_mean"] for r in rows if r["dense_time_mean"] != ""]
dn_mems  = [r["dense_mem_mean"] for r in rows if r["dense_mem_mean"] != ""]
ks       = [r["mean_k"] for r in rows]

# Time
ax_time.plot(ns, sp_times, "o-", color="#1f77b4", linewidth=2, label="GSHAC")
if dn_ns:
    ax_time.plot(dn_ns, dn_times, "s--", color="#d62728", linewidth=2, label="fastcluster")
# Reference slopes
n_ref = np.array([ns[0], ns[-1]], dtype=float)
t0 = sp_times[0]
ax_time.plot(n_ref, t0 * (n_ref / n_ref[0]) ** 1.1, ":", color="grey", alpha=0.5, label=r"$O(n^{1.1})$ ref.")
ax_time.plot(n_ref, (dn_times[0] if dn_times else t0) * (n_ref / n_ref[0]) ** 2, ":", color="#d62728", alpha=0.3, label=r"$O(n^2)$ ref.")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("n (dataset size)", fontsize=10)
ax_time.set_ylabel("Wall time (s)", fontsize=10)
ax_time.set_title("Wall time (constant k)", fontsize=11)
ax_time.legend(fontsize=8)
ax_time.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

# Annotate k values
for i in range(0, len(ns), 2):
    ax_time.annotate(f"k={ks[i]:.0f}", (ns[i], sp_times[i]),
                     textcoords="offset points", xytext=(5, 8), fontsize=7, color="#1f77b4")

# Memory
ax_mem.plot(ns, sp_mems, "o-", color="#1f77b4", linewidth=2, label="GSHAC")
if dn_ns:
    ax_mem.plot(dn_ns, dn_mems, "s--", color="#d62728", linewidth=2, label="fastcluster")
ax_mem.plot(n_ref, sp_mems[0] * (n_ref / n_ref[0]), ":", color="grey", alpha=0.5, label=r"$O(n)$ ref.")
ax_mem.plot(n_ref, (dn_mems[0] if dn_mems else sp_mems[0]) * (n_ref / n_ref[0]) ** 2, ":", color="#d62728", alpha=0.3, label=r"$O(n^2)$ ref.")
ax_mem.set_xscale("log")
ax_mem.set_yscale("log")
ax_mem.set_xlabel("n (dataset size)", fontsize=10)
ax_mem.set_ylabel("Peak memory (MiB)", fontsize=10)
ax_mem.set_title("Peak memory (constant k)", fontsize=11)
ax_mem.legend(fontsize=8)
ax_mem.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig_constk.pdf")
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fig_path}")
