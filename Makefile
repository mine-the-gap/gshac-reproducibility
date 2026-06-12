.PHONY: setup experiments figures tables paper clean all

VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

## Full pipeline: setup -> experiments -> figures -> tables -> paper
## Synthetic datasets are generated internally by the benchmark scripts; there is
## no separate data-generation step.
all: setup experiments figures tables paper

## Create virtual environment and install all dependencies (incl. C extension)
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Run all experiments (order matters for dependencies)
experiments:
	$(PYTHON) experiments/demo_exactness.py
	$(PYTHON) experiments/benchmark_constk.py
	$(PYTHON) experiments/benchmark_mining.py
	$(PYTHON) experiments/benchmark_hdbscan.py
	$(PYTHON) experiments/benchmark_geonames.py
	$(PYTHON) experiments/benchmark.py

## Generate all paper figures from results
figures:
	$(PYTHON) experiments/make_figures.py

## Generate all paper tables from results
tables:
	$(PYTHON) experiments/make_tables.py

## Compile the LaTeX paper to PDF
paper:
	cd paper && \
	  pdflatex -interaction=nonstopmode main.tex && \
	  bibtex main && \
	  pdflatex -interaction=nonstopmode main.tex && \
	  pdflatex -interaction=nonstopmode main.tex

## Remove generated figures, results, and LaTeX build artifacts
clean:
	rm -rf figures/ results/
	cd paper && rm -f main.aux main.bbl main.blg main.log main.out main.pdf comment.cut texput.log
