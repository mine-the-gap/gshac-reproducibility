"""
validate_correctness.py

Validates that GSHAC produces identical clustering results to the dense
fastcluster baseline across all benchmark scenarios and cut heights.

This is a standalone correctness check, separated from the performance
benchmark to avoid unnecessary memory and CPU overhead during timing runs.

For sizes where dense is feasible, compares cluster counts at every h_cut.
For larger sizes, verifies deterministic reproducibility across two
independent sparse runs.

Outputs:
  results/validation_correctness.csv
"""

from __future__ import annotations

import os
import sys
import gc
import numpy as np
from pathlib import Path

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Scenarios and parameters (must match benchmark.py)
# ------------------------------------------------------------------

SCENARIOS = {
    "tight":    dict(n_centres=100, spread_factor=0.03),
    "moderate": dict(n_centres=50,  spread_factor=0.10),
    "loose":    dict(n_centres=20,  spread_factor=0.30),
}

N_SIZES    = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
H_MAX_VALS = [10_000, 20_000, 50_000]
METHOD     = "single"
H_CUTS     = list(range(5_000, 55_000, 5_000))


def dense_feasible(n, max_gb=6.0):
    condensed_gb = n * (n - 1) / 2 * 8 / 1e9
    return condensed_gb <= max_gb


def generate_clustered_points(n_total, n_centres, spread_m,
                               domain_m=500_000, seed=42):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, domain_m, size=(n_centres, 2))
    assignments = rng.integers(0, n_centres, size=n_total)
    noise = rng.normal(0, spread_m, size=(n_total, 2))
    return centres[assignments] + noise


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

import csv

out_path = os.path.join(RESULTS_DIR, "validation_correctness.csv")
f_out = open(out_path, "w", newline="")
writer = None

n_configs = len(SCENARIOS) * len(N_SIZES) * len(H_MAX_VALS)
print(f"Correctness validation: {n_configs} configurations")
print(f"Dense feasible up to n={max(n for n in N_SIZES if dense_feasible(n)):,}\n")

all_pass = True

for scenario_name, scenario_cfg in SCENARIOS.items():
    n_centres = scenario_cfg["n_centres"]
    spread_factor = scenario_cfg["spread_factor"]

    for h_max in H_MAX_VALS:
        h_max_km = h_max / 1000
        spread_m = spread_factor * h_max

        for n in N_SIZES:
            valid_cuts = [h for h in H_CUTS if h <= h_max]
            if not valid_cuts:
                continue

            run_dense = dense_feasible(n)
            mode = "dense_vs_sparse" if run_dense else "reproducibility"
            print(f"  {scenario_name:<10} n={n:>7,}  h={h_max_km:.0f}km  [{mode}]",
                  end="", flush=True)

            pts = generate_clustered_points(n, n_centres, spread_m)

            if run_dense:
                # Compare sparse vs dense cluster counts
                g = spatial_dist_graph(pts, h_max)
                sp = sparse_hclust(g, H_CUTS, METHOD)
                del g
                gc.collect()

                dn = dense_hclust(pts, H_CUTS, METHOD)

                results = []
                for h in valid_cuts:
                    n_sp = len(np.unique(sp["labels"][h]))
                    n_dn = len(np.unique(dn["labels"][h]))
                    results.append((h, n_sp, n_dn, n_sp == n_dn))

                del sp, dn
                gc.collect()

                agree = all(r[3] for r in results)
                mismatches = [(h, ns, nd) for h, ns, nd, ok in results if not ok]

            else:
                # Verify reproducibility: two independent runs
                g1 = spatial_dist_graph(pts, h_max)
                r1 = sparse_hclust(g1, H_CUTS, METHOD)
                counts1 = {h: len(np.unique(r1["labels"][h])) for h in valid_cuts}
                del g1, r1
                gc.collect()

                g2 = spatial_dist_graph(pts, h_max)
                r2 = sparse_hclust(g2, H_CUTS, METHOD)
                counts2 = {h: len(np.unique(r2["labels"][h])) for h in valid_cuts}
                del g2, r2
                gc.collect()

                results = []
                for h in valid_cuts:
                    results.append((h, counts1[h], counts2[h],
                                    counts1[h] == counts2[h]))

                agree = all(r[3] for r in results)
                mismatches = [(h, c1, c2) for h, c1, c2, ok in results if not ok]

            del pts
            gc.collect()

            status = "PASS" if agree else "FAIL"
            if not agree:
                all_pass = False
            print(f"  {status}")
            if mismatches:
                for h, a, b in mismatches:
                    print(f"    h_cut={h/1000:.0f}km: {a} vs {b}")

            row = {
                "scenario":    scenario_name,
                "n":           n,
                "h_max_km":    h_max_km,
                "mode":        mode,
                "n_cuts":      len(valid_cuts),
                "all_agree":   agree,
                "n_mismatches": len(mismatches),
            }
            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f_out.flush()

f_out.close()
print(f"\nResults written to {out_path}")
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
