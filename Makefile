.PHONY: setup experiments reproduce-fast compare budget budget-fast figures tables clean all

VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

## Full pipeline: setup -> experiments -> figures -> tables
## Synthetic datasets are generated internally by the benchmark scripts; there is
## no separate data-generation step.
all: setup experiments figures tables

## Create virtual environment and install all dependencies (incl. C extension)
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Run all experiments (order matters for dependencies)
## NOTE: RQ1 (§5.1) exactness + determinism is folded INTO benchmark.py — it compares
## the dense vs sparse labels it already computes (see the exact_agree column in
## benchmark_python[_reps].csv); the standalone validate_correctness.py was dropped.
experiments:
	$(PYTHON) experiments/demo_exactness.py
	$(PYTHON) experiments/benchmark_constk.py
	$(PYTHON) experiments/benchmark_geonames.py
	$(PYTHON) experiments/benchmark_geonames_scaling.py
	$(PYTHON) experiments/benchmark_geonames_geographic.py
	$(PYTHON) experiments/benchmark_hdbscan.py
	$(PYTHON) experiments/benchmark.py

## Fast reproducibility check (TSAS reproducibility track): runs every benchmark at
## reduced scale (small samples, fewer reps) via GSHAC_FAST=1 to verify the pipeline
## and the paper's CORE FINDINGS in well under an hour on a commodity machine. It does
## NOT regenerate the figures/tables (the reduced CSVs are intentionally sparse);
## use the full `make all` for that. Self-contained: `make setup` runs first.
reproduce-fast: setup
	GSHAC_FAST=1 $(PYTHON) experiments/demo_exactness.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark_constk.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark_geonames.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark_geonames_scaling.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark_geonames_geographic.py
	GSHAC_FAST=1 $(PYTHON) experiments/benchmark_hdbscan.py
	@echo "============================================================"
	@echo "Fast reproduction complete. Verify the paper's core findings:"
	@echo "Outputs are under results_fast/ (the full-scale results/ is never touched)."
	@echo " - Exactness: exact_agree == True for every dense-feasible row"
	@echo "     -> results_fast/benchmark_python.csv  (ARI == 1.0 in demo_exactness.csv)"
	@echo " - Near-linear constant-k scaling: time slope ~1.0-1.1, memory ~1.0"
	@echo "     -> results_fast/benchmark_constk_slopes.csv"
	@echo " - Geographic (constant-density) scaling stays near-linear"
	@echo "     -> results_fast/benchmark_geonames_geographic_slopes.csv"
	@echo " - Fixed-extent (growing-density) scaling is super-linear (the limitation)"
	@echo "     -> results_fast/benchmark_geonames_scaling_slopes.csv"
	@echo " - GSHAC feasible on GeoNames where dense is not; lighter/faster than HDBSCAN"
	@echo "     -> results_fast/benchmark_geonames.csv, results_fast/benchmark_hdbscan.csv"
	@echo "Then run 'make compare' to check these against the committed full-scale results (results/)."
	@echo "============================================================"

## Contrast the fast-track results (results_fast/) against the committed full-scale
## results (results/): checks that scaling slopes, exactness, and mean-k align (these are
## hardware-invariant); absolute timings are reported for information only.
compare:
	$(PYTHON) experiments/compare_fast_vs_full.py

## Measure wall time + peak memory per benchmark -> results[_fast]/compute_budget.csv.
## `budget` measures the full track (hours, GeoNames-dominated); `budget-fast` the fast track (minutes).
budget:
	$(PYTHON) experiments/measure_budget.py
budget-fast:
	GSHAC_FAST=1 $(PYTHON) experiments/measure_budget.py

## Generate all figures from results (into figures/)
figures:
	$(PYTHON) experiments/make_figures.py

## Generate all tables from results (into tables/)
tables:
	$(PYTHON) experiments/make_tables.py

## Remove generated results (regenerate with `make experiments figures tables`)
clean:
	rm -rf results/ results_fast/
