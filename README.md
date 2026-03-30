# GSHAC Benchmark

Code and experiments for the ACM SIGSPATIAL 2026 paper:

> **GSHAC: Scalable Exact Hierarchical Clustering via Sparse Geographic Distance Graphs**

GSHAC replaces the O(n²) distance matrix required by hierarchical clustering with a sparse geographic graph containing only edges ≤ `h_max`, found via a spatial index. The result is **exact** for any cut height h ≤ h_max, with O(n·k) time and memory where k is the mean neighbourhood size.

The GSHAC library is at [vwmaus/gshac](https://github.com/vwmaus/gshac). This repo contains benchmarks and the paper source.

## Setup

Requires Python >= 3.10.

```bash
make setup
source .venv/bin/activate
```

## Running experiments

```bash
make data           # generate synthetic datasets
make experiments    # run all benchmarks (~3.5 hours)
make figures        # generate figures from results
make paper          # compile LaTeX to PDF
```

Or run everything at once: `make all`

Individual scripts are in `experiments/`. Results write to `results/`, figures to `figures/`.

## License

