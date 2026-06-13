# GSHAC Benchmark

Code and experiments for the paper (submitted to **ACM Transactions on Spatial
Algorithms and Systems, TSAS**):

> **GSHAC: Scalable Exact Hierarchical Clustering via Sparse Geographic Distance Graphs**

GSHAC replaces the O(n²) distance matrix required by hierarchical agglomerative
clustering (HAC) with a sparse geographic graph containing only edges ≤ `h_max`,
found via a spatial index. The result is **exact** for any cut height `h ≤ h_max`,
with O(n·k̄) time and memory, where k̄ is the mean neighbourhood size. When the data
has bounded local density (k̄ does not grow with n), this is near-linear and scales
to millions of points on a workstation.

The GSHAC library itself lives at [vwmaus/gshac](https://github.com/vwmaus/gshac).
This repository contains the **benchmarks, figure/table generators, and paper source**.

## Repository layout

```
experiments/   benchmark + figure/table scripts (see table below)
results/        generated CSVs — every number the paper cites (git-ignored)
paper/          LaTeX source + generated figures (fig_*.pdf) and tables (tab_*.tex)
data/           input datasets (git-ignored; GeoNames auto-downloads, see Data)
Makefile        the full pipeline: setup → experiments → figures → tables → paper
```

## Setup

Requires Python ≥ 3.10.

```bash
make setup                 # create .venv and install dependencies (incl. the C extension)
source .venv/bin/activate
```

## Running the pipeline

```bash
make experiments    # run all benchmarks  (~6 h; GeoNames threshold sweep dominates)
make figures        # regenerate paper/fig_*.pdf  from results/*.csv
make tables         # regenerate paper/tab_*.tex  from results/*.csv
make paper          # compile paper/main.tex to PDF
make all            # setup → experiments → figures → tables → paper
make clean          # remove results/, figures/, and LaTeX build artifacts
```

Each benchmark writes its own CSV(s) to `results/` (flushed incrementally, so a
later failure never loses earlier output). `make figures` / `make tables` then read
those CSVs — so you can re-run a single benchmark and rebuild only the affected
figures/tables without repeating the whole stack.

## Benchmarks

Run any script individually with `python experiments/<script>.py`. Grouped by what
they measure:

| Script | What it measures | Key outputs |
|---|---|---|
| **Correctness** | | |
| `demo_exactness.py` | Visual exactness check: GSHAC vs dense `fastcluster` on the standard 2-D shapes (two moons, two circles); identical partitions at every cut (ARI = 1.0). | `demo_exactness.csv`, `fig_exactness.pdf` |
| `benchmark.py` *(RQ1 check)* | Folds in the exactness/determinism certification: across all dense-feasible synthetic configs it compares GSHAC's cluster counts to the dense baseline (sparse == dense); for sparse-only sizes it checks reproducibility across reps. | `exact_agree` column in `benchmark_python.csv` / `benchmark_python_reps.csv` |
| **Scaling** | | |
| `benchmark.py` | Synthetic fixed-domain benchmark: 3 density scenarios × sizes (≤ 50k) × `h_max`, 5 reps. Dense (`fastcluster`) vs GSHAC time + peak memory, plus PySAL and scikit-learn connectivity baselines. | `benchmark_python.csv` (+ `_reps`), `benchmark_pysal.csv`, `benchmark_sklearn_comparison.csv`, `benchmark_sklearn_timing.csv`; `fig_scaling.pdf`, `tab_summary/pysal/sklearn.tex` |
| `benchmark_constk.py` | Constant-`k̄` scaling: density held bounded as n grows → near-linear time and memory (the headline complexity claim). | `benchmark_constk.csv` (+ `_slopes`), `fig_constk.pdf`, `tab_constk.tex` |
| **Real-world (GeoNames — open data, ~2 M sample)** | | |
| `benchmark_geonames.py` | Threshold sweep at `h_max ∈ {1, 5, 10, 20} km` on the 2 M stratified sample; time, 3-phase peak memory, component structure, and the dense sizes avoided. Each rep runs in a fresh subprocess (20 km peaks ~28 GiB). | `benchmark_geonames.csv`, `tab_geonames.tex` |
| `benchmark_geonames_geographic.py` | **Headline scaling:** grow the geographic *window* at constant density (widening longitude strip) → near-linear (k̄ stays bounded). | `benchmark_geonames_geographic.csv` (+ `_slopes`) |
| `benchmark_geonames_scaling.py` | **Limitations:** fixed-extent subsampling → density (and k̄) grow ∝ n → memory ~quadratic. The adversarial regime; GSHAC still *runs* where dense cannot. | `benchmark_geonames_scaling.csv` (+ `_slopes`) |
| `benchmark_hdbscan.py` | Resource comparison vs scikit-learn HDBSCAN on a 100k GeoNames subset (5 reps each), plus a scaling probe extrapolating HDBSCAN to 2 M. Non-equivalent outputs — resource profiles only. | `benchmark_hdbscan.csv` (+ `_scaling`), `tab_hdbscan.tex` |
| **Motivating application (mining — not redistributable)** | | |
| `benchmark_mining.py` | Feasibility on a 261k-point mining inventory at scales where dense HAC is analytically impossible. Demoted to a cited motivating application (the data is **not public**); all reproducible results use GeoNames + synthetic. | `benchmark_mining*.csv` |
| **Build** | | |
| `make_figures.py` | Regenerate every `paper/fig_*.pdf` from `results/*.csv`. | `fig_*.pdf`, `scenario_illustration.csv`, `scaling_slopes.csv` |
| `make_tables.py` | Regenerate every `paper/tab_*.tex` from `results/*.csv`. | `tab_*.tex` |

## Data

- **GeoNames** (CC BY 4.0): downloaded automatically on first run
  (`allCountries.zip` → `allCountries.txt`, ~1.7 GB extracted) and cached in `data/`.
  No manual step needed.
- **Mining** (`data/cluster_data.gpkg`): a proprietary inventory that is **not
  redistributable**. `benchmark_mining.py` requires this file and will error without
  it; it is a motivating application only, so its absence does not affect any of the
  open, reproducible results.

## Reproducibility notes

- Every figure and table is regenerated from `results/*.csv`; no numbers are hand-typed
  into the scripts.
- Memory is reported in **binary units** (MiB / GiB / TiB) and as graph-phase,
  HAC-phase, and whole-pipeline `tracemalloc` peaks.
- Timed benchmarks use **5 repetitions** (mean ± sd); the exactness demo and PySAL
  graph build are deterministic single runs.
- The heavy GeoNames benchmarks isolate **each rep in a fresh subprocess** so the OS
  fully reclaims memory between reps — required because the 20 km graph is at the
  machine's RAM limit.
- Reference hardware: AMD Ryzen 7 PRO 8840U (8 cores), 30 GiB RAM, Ubuntu (kernel 6.17,
  Python 3.12). The 20 km / 2 M GeoNames point sits at this machine's limit.

## License

The GSHAC library is licensed separately at
[vwmaus/gshac](https://github.com/vwmaus/gshac). License terms for this
benchmark/paper repository are to be finalized.
