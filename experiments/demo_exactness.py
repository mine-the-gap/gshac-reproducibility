"""
demo_exactness.py

Demonstrates that GSHAC produces exact cluster assignments (identical
to fastcluster) on the Iris dataset, even when using only 23.5% of
pairwise distances.

Output: figures/fig_exactness.pdf
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.optimize import linear_sum_assignment
import fastcluster

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, dense_hclust, _build_Z_from_mst
from scipy.sparse.csgraph import minimum_spanning_tree as _mst

_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

iris = load_iris()
X = iris.data  # (150, 4)
n = X.shape[0]

pca = PCA(n_components=2)
X2d = pca.fit_transform(X)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

h_max = 1.0   # aggressive: only 23.5% of pairwise distances
h_cut = 0.7   # gives 4 clusters — visually distinct

# ------------------------------------------------------------------
# Dense baseline
# ------------------------------------------------------------------

Z_dense = fastcluster.linkage(pdist(X), method="single")
r_dense = dense_hclust(X, [h_cut], method="single")
l_dense = r_dense["labels"][h_cut]

# ------------------------------------------------------------------
# GSHAC
# ------------------------------------------------------------------

g = spatial_dist_graph(X, h_max, metric="euclidean")
r_gshac = sparse_hclust(g, [h_cut], method="single")
l_gshac = r_gshac["labels"][h_cut]

density_pct = g["density"] * 100
n_edges = g["n_edges"]
total_pairs = n * (n - 1) // 2

print(f"Iris: n={n}, features={X.shape[1]}")
print(f"h_max={h_max}: {n_edges:,} / {total_pairs:,} edges ({density_pct:.1f}%)")
print(f"Components: {g['n_components']}")

# ------------------------------------------------------------------
# Verify exactness
# ------------------------------------------------------------------

n_cl_dense = len(np.unique(l_dense))
n_cl_gshac = len(np.unique(l_gshac))
ari = adjusted_rand_score(l_dense, l_gshac)

print(f"\nAt h={h_cut}: dense={n_cl_dense} clusters, GSHAC={n_cl_gshac} clusters, ARI={ari:.4f}")

# Verify at multiple cuts
print(f"\nPartition equivalence at all h <= h_max:")
for h in [0.3, 0.5, 0.7, 1.0]:
    r_d = dense_hclust(X, [h], method="single")
    r_s = sparse_hclust(g, [h], method="single")
    a = adjusted_rand_score(r_d["labels"][h], r_s["labels"][h])
    nd = len(np.unique(r_d["labels"][h]))
    ns = len(np.unique(r_s["labels"][h]))
    print(f"  h={h}: dense={nd:>3}, GSHAC={ns:>3}, ARI={a:.4f}")

# ------------------------------------------------------------------
# Build per-component dendrogram for the largest component
# ------------------------------------------------------------------

mat = g["matrix"]
comp_labels = g["components"]

# Find largest component
comp_ids, comp_counts = np.unique(comp_labels, return_counts=True)
largest_comp = comp_ids[np.argmax(comp_counts)]
largest_idx = np.where(comp_labels == largest_comp)[0]
largest_size = len(largest_idx)

print(f"\nLargest component: {largest_size} points (of {n})")

# GSHAC Z for largest component
sub_csr = mat[np.ix_(largest_idx, largest_idx)].tocsr()
Z_gshac_comp = _build_Z_from_mst(sub_csr, largest_size)

# Dense Z for same subset
condensed_sub = pdist(X[largest_idx])
Z_dense_comp = fastcluster.linkage(condensed_sub, method="single")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def align_labels(ref, other):
    labels_ref = np.unique(ref)
    labels_oth = np.unique(other)
    cost = np.zeros((len(labels_ref), len(labels_oth)))
    for i, lr in enumerate(labels_ref):
        for j, lo in enumerate(labels_oth):
            cost[i, j] = -np.sum((ref == lr) & (other == lo))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {labels_oth[c]: labels_ref[r] for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(x, x) for x in other])

l_gshac_aligned = align_labels(l_dense, l_gshac)

fig = plt.figure(figsize=(7.16, 5.5))
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

# --- Top-left: fastcluster dendrogram (largest component) ---
ax1 = fig.add_subplot(gs[0, 0])
dendrogram(Z_dense_comp, ax=ax1, no_labels=True,
           color_threshold=h_cut, above_threshold_color="#888888")
ax1.set_title("(a) fastcluster (dense, 100% pairs)", fontsize=9)
ax1.set_ylabel("Distance", fontsize=9)
ax1.axhline(h_cut, color="#d62728", linestyle="--", linewidth=0.8)
ax1.text(0.97, h_cut + 0.02, f"$h = {h_cut}$",
         transform=ax1.get_yaxis_transform(),
         fontsize=7, color="#d62728", ha="right")
ax1.tick_params(axis="x", bottom=False, labelbottom=False)
ax1.set_ylim(0, 1.2)

# --- Top-right: GSHAC dendrogram (largest component) ---
ax2 = fig.add_subplot(gs[0, 1])
dendrogram(Z_gshac_comp, ax=ax2, no_labels=True,
           color_threshold=h_cut, above_threshold_color="#888888")
ax2.set_title(f"(b) GSHAC ($h_{{max}}$ = {h_max}, {density_pct:.0f}% pairs)",
              fontsize=9)
ax2.set_ylabel("Distance", fontsize=9)
ax2.axhline(h_cut, color="#d62728", linestyle="--", linewidth=0.8)
ax2.text(0.97, h_cut + 0.02, f"$h = {h_cut}$",
         transform=ax2.get_yaxis_transform(),
         fontsize=7, color="#d62728", ha="right")
ax2.tick_params(axis="x", bottom=False, labelbottom=False)
ax2.set_ylim(0, 1.2)

# --- Bottom-left: Dense cluster scatter (all points) ---
ax3 = fig.add_subplot(gs[1, 0])
for cl in sorted(np.unique(l_dense)):
    mask = l_dense == cl
    ax3.scatter(X2d[mask, 0], X2d[mask, 1], s=18, alpha=0.8,
                edgecolors="k", linewidths=0.3)
ax3.set_title(f"(c) fastcluster at $h = {h_cut}$: {n_cl_dense} clusters",
              fontsize=9)
ax3.set_xlabel("PC1", fontsize=9)
ax3.set_ylabel("PC2", fontsize=9)

# --- Bottom-right: GSHAC cluster scatter (all points) ---
ax4 = fig.add_subplot(gs[1, 1])
for cl in sorted(np.unique(l_gshac_aligned)):
    mask = l_gshac_aligned == cl
    ax4.scatter(X2d[mask, 0], X2d[mask, 1], s=18, alpha=0.8,
                edgecolors="k", linewidths=0.3)
ax4.set_title(f"(d) GSHAC at $h = {h_cut}$: {n_cl_gshac} clusters "
              f"(ARI = {ari:.1f})", fontsize=9)
ax4.set_xlabel("PC1", fontsize=9)
ax4.set_ylabel("PC2", fontsize=9)

fig_path = os.path.join(FIGURES_DIR, "fig_exactness.pdf")
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved {fig_path}")
