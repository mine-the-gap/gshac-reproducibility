# GSHAC Benchmark

Code and experiments for the paper (submitted to **ACM Transactions on Spatial
Algorithms and Systems, TSAS**):

> **Scalable Exact Hierarchical Agglomerative Clustering via Sparse Geographic Distance Graphs**
> Victor Maus, Vinicius Pozzobon Borin.
> Preprint: [arXiv:2604.11656](https://doi.org/10.48550/arXiv.2604.11656)

GSHAC replaces the O(n²) distance matrix required by hierarchical agglomerative
clustering (HAC) with a sparse geographic graph containing only edges ≤ `h_max`,
found via a spatial index. The result is **exact** for any cut height `h ≤ h_max`,
with O(n·k̄) time and memory, where k̄ is the mean neighbourhood size. When the data
has bounded local density (k̄ does not grow with n), this is near-linear and scales
to millions of points on a workstation.

The GSHAC library itself lives at [mine-the-gap/gshac](https://github.com/mine-the-gap/gshac).
This repository contains the **benchmarks, figure/table generators, and paper source**.

## Repository layout

```
experiments/   benchmark + figure/table scripts (see table below)
results/        generated CSVs — every number the paper cites (committed; `make compare`'s reference)
paper/          LaTeX source + generated figures (fig_*.pdf) and tables (tab_*.tex)
data/           input datasets (git-ignored; GeoNames auto-downloads, see Data)
Makefile        the full pipeline: setup → experiments → figures → tables → paper
```

## System requirements

Approximate resources to reproduce each track on a single machine — no cluster needed.
Both tracks download and parse the GeoNames dump (~1.7 GB) on first run.

| | Fast track (`make reproduce-fast`) | Full track (`make all`) |
|---|---|---|
| **RAM** | ~8 GiB (a few GiB used; the GeoNames parse is the peak) | **≥ 30 GiB** — the 20 km / 2 M GeoNames point peaks ~22 GiB |
| **Disk** | ~3 GiB free (GeoNames dump + generated outputs) | ~3 GiB free |
| **Time** | ~5–10 min (the GeoNames-ladder clustering dominates; the 1.7 GB parse, ~8 s, is cached to `.npy` after first use) | several hours (the GeoNames threshold sweep dominates) |
| **CPU** | any modern multi-core | multi-core recommended |
| **Internet** | first run only (GeoNames download) | first run only |
| **Software** | Python ≥ 3.10, Linux/macOS, a C compiler; pinned deps via `make setup` | same |

Reference machine: AMD Ryzen 7 PRO 8840U (8 cores), 30 GiB RAM, Ubuntu (kernel 6.17),
Python 3.12.3. The full track is feasible on any workstation with ~32 GiB RAM; the fast
track runs on a commodity laptop. To measure exact wall-time and peak memory on your own
hardware, run `make budget-fast` (or `make budget` for the full track).

## Setup

Requires Python ≥ 3.10 (reference environment: Python 3.12.3).

```bash
make setup                 # create .venv and install dependencies (incl. the C extension)
source .venv/bin/activate
```

All dependencies are **version-pinned** for reproducibility. `requirements.txt` pins the
direct dependencies and the GSHAC library to release tag **`1.0.1`**
([mine-the-gap/gshac](https://github.com/mine-the-gap/gshac)). For a byte-for-byte lock of
the entire transitive environment used to produce the reported numbers, install
`requirements-lock.txt` instead:

```bash
.venv/bin/pip install -r requirements-lock.txt
```

## Running the pipeline

```bash
make reproduce-fast # quick check: every benchmark at reduced scale, core findings in <1 h
make experiments    # run all benchmarks  (~6 h; GeoNames threshold sweep dominates)
make figures        # regenerate paper/fig_*.pdf  from results/*.csv
make tables         # regenerate paper/tab_*.tex  from results/*.csv
make paper          # compile paper/main.tex to PDF
make all            # setup → experiments → figures → tables → paper  (the master script)
make clean          # remove results/, figures/, and LaTeX build artifacts
```

**`make all`** is the single master command: it sets up the environment, fetches data,
reruns every experiment, regenerates all figures and tables in paper order, and compiles
the PDF. The full run takes ~6 h and the heaviest GeoNames point peaks near the machine's
RAM (see hardware note below).

**`make reproduce-fast`** is the recommended first step for an artifact reviewer: it runs
every benchmark at reduced scale (smaller samples; via `GSHAC_FAST=1`) so the paper's
**core findings** are verifiable in well under an hour on a commodity machine with a few GiB
of free RAM. The fast track runs each configuration **once** (no repetitions) — consistency
is established not by repeating the fast run, but by checking it against the full run's
confidence interval (see `make compare` below). The GeoNames parse (~13 M rows) is **cached**
to `data/geonames_coords.npy` on first use, so subsequent `load_coords()` calls skip the
~1.7 GB re-parse. Fast-mode runs write **only** to a separate `results_fast/`
directory and **never overwrite** the expensive full-scale results in `results/`. It does
**not** regenerate the publication figures (the reduced CSVs are intentionally sparse); the
trends to confirm are printed at the end of the run. Use the full `make all` to reproduce the
paper figures and exact numbers.

After a fast run, **`make compare`** contrasts `results_fast/` against the committed
full-scale `results/` and reports whether the fast-track estimates line up:
- **Scale-/hardware-invariant checks (pass/fail):** the scaling-law slopes, the exactness
  agreement, the mean neighbourhood size *k*, and whether the single fast measurement lands
  **inside the full run's confidence band** (mean ± 3σ from the 5-rep full run, with a ≥10%
  floor) for the near-deterministic memory metric.
- **Informational only:** absolute wall-time/memory speedups and the GeoNames scaling slopes
  (the latter are fit over the narrow fast *n*-range, 50k–200k, so they are estimates rather
  than precise) — these legitimately vary with CPU, RAM, OS, and threading.

Because the full-scale `results/*.csv` are committed to this repo, a reviewer can run
`make reproduce-fast && make compare` on a clean checkout to confirm the estimates align,
without ever running the multi-hour full sweep.

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
| **Build** | | |
| `make_figures.py` | Regenerate every `paper/fig_*.pdf` from `results/*.csv`. | `fig_*.pdf`, `scenario_illustration.csv`, `scaling_slopes.csv` |
| `make_tables.py` | Regenerate every `paper/tab_*.tex` from `results/*.csv`. | `tab_*.tex` |

## Data

- **GeoNames** (CC BY 4.0): downloaded automatically on first run via direct HTTP from
  `https://download.geonames.org/export/dump/allCountries.zip` (`allCountries.zip` →
  `allCountries.txt`, ~1.7 GB extracted) and cached in `data/`. No account, API key, or
  manual step is required.
- **Snapshot / version.** GeoNames `allCountries` is refreshed daily, so the live download
  evolves over time. The reported numbers were produced from the dump retrieved on
  **2026-03-25**; the 2 M analysis set is a spatially stratified sample of it drawn with a
  fixed seed (`RANDOM_SEED = 42`). Because the source dump drifts, an independent run may
  land on a slightly different sample — per the TSAS criteria, exact replication is not
  required, only agreement with the paper's core findings (feasibility at each threshold,
  near-linear constant-density scaling, the resource gap vs. HDBSCAN). To reproduce the
  exact sample, use a `2026-03-25` `allCountries.txt` (archive it alongside this repo if
  bit-exact reproduction is needed).

## Reproducibility notes

- Every figure and table is regenerated from `results/*.csv`; no numbers are hand-typed
  into the scripts.
- **Determinism:** all synthetic data and all sampling use a fixed seed (`42`), so a run
  on the same environment reproduces the same partitions and the same row-level numbers.
- **Pinned environment:** `requirements.txt` (direct deps + `1.0.1`) and
  `requirements-lock.txt` (full transitive lock) fix every package version.
- Memory is reported in **binary units** (MiB / GiB / TiB) and as graph-phase,
  HAC-phase, and whole-pipeline `tracemalloc` peaks.
- Timed benchmarks use **5 repetitions** (mean ± sd); the exactness demo and PySAL
  graph build are deterministic single runs. (Under `GSHAC_FAST=1`, reps are reduced.)
- The heavy GeoNames benchmarks isolate **each rep in a fresh subprocess** so the OS
  fully reclaims memory between reps — required because the 20 km graph is at the
  machine's RAM limit.
- Reference hardware: AMD Ryzen 7 PRO 8840U (8 cores), 30 GiB RAM, Ubuntu (kernel 6.17,
  Python 3.12). The 20 km / 2 M GeoNames point sits at this machine's limit.

## Expected results

After a run, confirm the paper's core findings in the CSVs — under `results/` for a full
run, or `results_fast/` for `make reproduce-fast`. The same trends hold in both modes; only
the magnitudes of *n* differ (paths below are shown for the full run):

| Claim (paper) | Where to look | What you should see |
|---|---|---|
| **Exactness** (Thm. 1): sparse == dense | `benchmark_python.csv` (`exact_agree`), `benchmark_linkage_exactness.csv` | `exact_agree == True` for every dense-feasible config; single/complete/average match the dense `fastcluster` partition at every cut; ARI `== 1.0` in `demo_exactness.csv` |
| **Determinism** across reps | `benchmark_python_reps.csv` | identical cluster counts across reps at each *n* |
| **Near-linear constant-k scaling** | `benchmark_constk_slopes.csv` | time slope ≈ 1.0–1.1, memory slope ≈ 1.0 (vs. dense ≈ 2.0) |
| **Crossover vs. dense** | `benchmark_python.csv` | GSHAC overtakes dense between *n* = 1k–5k; dense omitted past its RAM limit while GSHAC continues |
| **Constant-density geographic scaling stays near-linear** | `benchmark_geonames_geographic_slopes.csv` | slopes ≈ 1 (k̄ stays bounded as the window grows) |
| **Fixed-extent (growing-density) scaling is super-linear** — the limitation | `benchmark_geonames_scaling_slopes.csv` | slopes clearly > 1 (k̄ grows with *n*) |
| **Feasibility on 2 M GeoNames where dense is impossible** | `benchmark_geonames.csv` | each threshold completes; dense matrix size reported as analytically infeasible |
| **Resource gap vs. HDBSCAN** | `benchmark_hdbscan.csv` | GSHAC markedly faster and lighter at equal *n* |

## Citation

If you use this software or its benchmarks, please cite the paper:

> Maus, V., & Pozzobon Borin, V. (2026). *Scalable Exact Hierarchical Agglomerative
> Clustering via Sparse Geographic Distance Graphs.* arXiv:2604.11656.
> [https://doi.org/10.48550/arXiv.2604.11656](https://doi.org/10.48550/arXiv.2604.11656)

The method was first introduced and applied to global mining land-use in:

> Maus, V. (2026). *A data-driven approach to mapping global commodity-specific
> mining land-use.* Journal of Cleaner Production, 540, 147437.
> [https://doi.org/10.1016/j.jclepro.2025.147437](https://doi.org/10.1016/j.jclepro.2025.147437)

Machine-readable metadata for both is in [`CITATION.cff`](CITATION.cff).

## License

GPL-3.0. See [LICENSE](LICENSE). The GSHAC python library is licensed separately at
[mine-the-gap/gshac](https://github.com/mine-the-gap/gshac).
