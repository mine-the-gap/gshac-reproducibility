#!/usr/bin/env python3
"""
measure_budget.py — wall-clock time and peak memory per benchmark, for the
compute-budget table in the README.

Respects GSHAC_FAST: with it set, the fast (reduced-scale) track is measured and the
budget is written to results_fast/; otherwise the full track is measured and written
to results/. Each benchmark is run as a child process; we record its wall time and
peak resident set size (ru_maxrss). For the subprocess-isolated GeoNames benchmarks,
ru_maxrss reflects the orchestrator process (dominated by the ~1.7 GB GeoNames parse);
the per-configuration clustering peaks (tracemalloc) are reported inside the benchmark
CSVs themselves.

Usage:
  python experiments/measure_budget.py                      # full track, all benchmarks
  GSHAC_FAST=1 python experiments/measure_budget.py          # fast track, all benchmarks
  python experiments/measure_budget.py benchmark_constk.py   # measure only listed scripts

NB: the full track includes the GeoNames sweep and takes hours; the fast track is minutes.
"""
import os
import sys
import csv
import time
import resource

FAST = os.environ.get("GSHAC_FAST") == "1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "results_fast" if FAST else "results")
LOG_DIR = os.path.join(OUT_DIR, "run_logs")
os.makedirs(LOG_DIR, exist_ok=True)

ALL = [
    "demo_exactness.py",
    "benchmark.py",
    "benchmark_constk.py",
    "benchmark_geonames.py",
    "benchmark_geonames_scaling.py",
    "benchmark_geonames_geographic.py",
    "benchmark_hdbscan.py",
]


def run_one(script):
    path = os.path.join(ROOT, "experiments", script)
    log = os.path.join(LOG_DIR, script.replace(".py", ".log"))
    t0 = time.monotonic()
    pid = os.fork()
    if pid == 0:  # child
        os.chdir(ROOT)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.execv(sys.executable, [sys.executable, path])
        os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    wall = time.monotonic() - t0
    rc = os.waitstatus_to_exitcode(status)
    peak_mib = ru.ru_maxrss / 1024.0  # ru_maxrss is KiB on Linux
    return rc, wall, peak_mib


def main():
    scripts = sys.argv[1:] or ALL
    track = "FAST" if FAST else "FULL"
    print(f"Measuring {track} track ({len(scripts)} benchmarks) -> {OUT_DIR}/compute_budget.csv\n")
    rows = []
    for s in scripts:
        print(f"  running {s} ...", flush=True)
        rc, wall, peak = run_one(s)
        flag = "" if rc == 0 else f"  [exit {rc}]"
        print(f"    {wall:8.1f} s   peak {peak:8.1f} MiB ({peak/1024:.2f} GiB){flag}", flush=True)
        rows.append(dict(benchmark=s, wall_s=round(wall, 1),
                         peak_rss_mib=round(peak, 1), returncode=rc))

    out = os.path.join(OUT_DIR, "compute_budget.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "wall_s", "peak_rss_mib", "returncode"])
        w.writeheader()
        w.writerows(rows)

    total_wall = sum(r["wall_s"] for r in rows)
    peak_overall = max((r["peak_rss_mib"] for r in rows), default=0.0)
    print(f"\n  {'TOTAL':<34} {total_wall:8.1f} s ({total_wall/60:.1f} min)   "
          f"peak {peak_overall:8.1f} MiB ({peak_overall/1024:.2f} GiB)")
    print(f"  (benchmarks run sequentially, so the track's peak RAM is the single heaviest run)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
