"""
generate_synthetic.py

Generates synthetic spatial datasets for benchmarking.

Two distributions are provided:
  1. Clustered (Gaussian mixture): n_centres clusters, each with Gaussian
     spread. Simulates real-world spatial features that tend to co-locate
     (e.g., mining sites, settlements, infrastructure nodes).
  2. Uniform (Poisson): points uniformly distributed in [0, domain_m]^2.
     Worst case for the sparse approach: few neighbours within h_max.

Outputs:
  data/synthetic_clustered_<n>.csv
  data/synthetic_uniform_<n>.csv

Each CSV has columns: id, x, y (projected metres, EPSG:3857 convention).
Also writes a GeoPackage for use with R:
  data/synthetic_clustered_<n>.gpkg
  data/synthetic_uniform_<n>.gpkg
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

DOMAIN_M    = 500_000   # 500 km square domain
N_CENTRES   = 30        # number of Gaussian cluster centres
N_SIZES     = [100, 500, 1_000, 2_500, 5_000, 10_000, 50_000]
SEED        = 42


def generate_clustered(n: int, n_centres: int = N_CENTRES,
                        spread_m: float = 8_000,
                        domain_m: float = DOMAIN_M,
                        seed: int = SEED) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, domain_m, size=(n_centres, 2))
    assigns = rng.integers(0, n_centres, size=n)
    noise   = rng.normal(0, spread_m, size=(n, 2))
    coords  = centres[assigns] + noise
    geoms   = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame(
        {"id": np.arange(n), "x": coords[:, 0], "y": coords[:, 1]},
        geometry=geoms, crs="EPSG:3857"
    )


def generate_uniform(n: int, domain_m: float = DOMAIN_M,
                     seed: int = SEED) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed + 1)
    coords = rng.uniform(0, domain_m, size=(n, 2))
    geoms  = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame(
        {"id": np.arange(n), "x": coords[:, 0], "y": coords[:, 1]},
        geometry=geoms, crs="EPSG:3857"
    )


if __name__ == "__main__":
    for n in N_SIZES:
        print(f"Generating n = {n:,}...")

        gdf_c = generate_clustered(n)
        gdf_u = generate_uniform(n)

        gdf_c[["id", "x", "y"]].to_csv(_DATA_DIR / f"synthetic_clustered_{n}.csv", index=False)
        gdf_u[["id", "x", "y"]].to_csv(_DATA_DIR / f"synthetic_uniform_{n}.csv",   index=False)

        gdf_c.to_file(_DATA_DIR / f"synthetic_clustered_{n}.gpkg", driver="GPKG", layer="points")
        gdf_u.to_file(_DATA_DIR / f"synthetic_uniform_{n}.gpkg",   driver="GPKG", layer="points")

    print("Done. Files written to data/")
