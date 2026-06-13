"""
demo_exactness.py

Demonstrates that GSHAC produces cluster assignments identical to dense
fastcluster on the standard 2-D clustering benchmarks used to test
density/connectivity-based methods: two interleaving moons and two
concentric circles (sklearn.datasets.make_moons / make_circles).

Single linkage recovers these non-convex shapes by connectivity, and GSHAC
reproduces the partition exactly (ARI = 1.0 at every cut h <= h_max) while
computing only the sub-h_max (local) distances.

Outputs:
  figures/fig_exactness.pdf
  results/demo_exactness.csv
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.optimize import linear_sum_assignment
import fastcluster

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(_ROOT, "paper")   # fig_exactness.pdf is \input by main.tex
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Datasets and parameters
# ------------------------------------------------------------------

N      = 500
H_MAX  = 0.20                 # euclidean; ~5% of pairs, 2 components per shape
H_CUTS = [0.10, 0.15, 0.20]   # all <= H_MAX
H_VIEW = 0.20                 # cut drawn in the figure (-> 2 clusters)
METHOD = "single"

DATASETS = [
    ("Two moons",   make_moons(n_samples=N, noise=0.06, random_state=42)[0]),
    ("Two circles", make_circles(n_samples=N, noise=0.05, factor=0.5, random_state=42)[0]),
]

_PALETTE = np.array(plt.cm.tab10.colors)


def _colors(labels):
    return _PALETTE[(np.asarray(labels) - 1) % 10]


def _align(ref, other):
    """Relabel `other` to best match `ref` (Hungarian) so identical partitions
    get identical colours across the GSHAC / dense panels."""
    lr, lo = np.unique(ref), np.unique(other)
    cost = np.zeros((len(lr), len(lo)))
    for i, a in enumerate(lr):
        for j, b in enumerate(lo):
            cost[i, j] = -np.sum((ref == a) & (other == b))
    ri, ci = linear_sum_assignment(cost)
    mapping = {lo[c]: lr[r] for r, c in zip(ri, ci)}
    return np.array([mapping.get(x, x) for x in other])


# ------------------------------------------------------------------
# Run + figure
# ------------------------------------------------------------------

fig, axes = plt.subplots(len(DATASETS), 3, figsize=(7.16, 4.7))
csv_rows = []

for di, (dname, X) in enumerate(DATASETS):
    n = len(X)
    total_pairs = n * (n - 1) // 2

    # GSHAC: sparse graph + single-linkage HAC
    g = spatial_dist_graph(X, H_MAX, metric="euclidean")
    r = sparse_hclust(g, H_CUTS, method=METHOD)
    n_edges      = g["n_edges"]
    density_pct  = g["density"] * 100
    n_components = g["n_components"]
    largest      = int(np.bincount(g["components"]).max())

    # Dense baseline: fastcluster on the full condensed distance vector
    Z_dense = fastcluster.linkage(pdist(X), method=METHOD)

    print(f"\n{dname}: n={n}, h_max={H_MAX}")
    print(f"  GSHAC computes {n_edges:,}/{total_pairs:,} pairs ({density_pct:.1f}%); "
          f"{n_components} components (largest {largest})")
    for h in H_CUTS:
        l_dense = fcluster(Z_dense, t=h, criterion="distance")
        l_gshac = r["labels"][h]
        ari = adjusted_rand_score(l_dense, l_gshac)
        nd, ng = len(np.unique(l_dense)), len(np.unique(l_gshac))
        print(f"    h={h}: dense={nd:>3} clusters, GSHAC={ng:>3}, ARI={ari:.4f}")
        csv_rows.append({
            "dataset": dname, "n": n, "n_features": 2, "h_max": H_MAX, "h_cut": h,
            "n_edges": n_edges, "total_pairs": total_pairs,
            "density_pct": round(density_pct, 2), "n_components": n_components,
            "largest_component": largest,
            "n_clusters_dense": nd, "n_clusters_gshac": ng, "ari": round(float(ari), 6),
        })

    # Figure row: (a) sparse graph, (b) GSHAC clusters, (c) dense clusters
    l_dense_v = fcluster(Z_dense, t=H_VIEW, criterion="distance")
    l_gshac_v = _align(l_dense_v, r["labels"][H_VIEW])
    ari_v = adjusted_rand_score(l_dense_v, r["labels"][H_VIEW])

    ax = axes[di, 0]
    coo = g["matrix"].tocoo()
    mk = coo.row < coo.col
    segs = np.stack([X[coo.row[mk]], X[coo.col[mk]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#b0b0b0", linewidths=0.2,
                                     alpha=0.4, zorder=1))
    ax.scatter(X[:, 0], X[:, 1], s=4, color="black", zorder=2, linewidths=0)
    ax.set_title(f"{dname}\nsparse graph: {density_pct:.1f}% of pairs, "
                 f"{n_components} comps", fontsize=8)

    ax = axes[di, 1]
    ax.scatter(X[:, 0], X[:, 1], s=6, color=_colors(l_gshac_v), linewidths=0)
    ax.set_title(f"GSHAC ($h={H_VIEW}$): "
                 f"{len(np.unique(l_gshac_v))} clusters", fontsize=8)

    ax = axes[di, 2]
    ax.scatter(X[:, 0], X[:, 1], s=6, color=_colors(l_dense_v), linewidths=0)
    ax.set_title(f"fastcluster (dense): "
                 f"{len(np.unique(l_dense_v))} clusters\nARI = {ari_v:.1f}", fontsize=8)

    for c in range(3):
        axes[di, c].set_aspect("equal")
        axes[di, c].set_xticks([])
        axes[di, c].set_yticks([])

fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig_exactness.pdf")
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved {fig_path}")

csv_path = os.path.join(RESULTS_DIR, "demo_exactness.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)
print(f"Saved {csv_path}")
