#!/usr/bin/env python3
"""
compare_fast_vs_full.py

Contrast the reduced-scale FAST-track results (``results_fast/``, produced by
``make reproduce-fast``) against the FULL-scale results (``results/``, produced by
``make experiments`` and committed to this repository). The fast track is a quick
reproducibility *estimate*; this script checks that its scaling behaviour lines up
with the full run, so a reviewer can confirm the fast estimate is representative
without paying for the multi-hour full sweep.

What is compared, and why these quantities:
  * Scaling-law SLOPES (log-log exponents for time and memory). These describe how
    cost grows with n and are essentially hardware-independent — the headline claim
    of the paper is about these slopes, so they are the primary alignment check.
  * EXACTNESS agreement (sparse == dense). A correctness property, independent of
    machine; must hold in both runs.
  * Mean neighbourhood size k at shared n. A deterministic property of the (seeded)
    synthetic data; must match closely.

What is NOT a pass/fail criterion:
  * Absolute wall-time and peak-memory speedups. These depend on CPU, RAM headroom,
    OS, and BLAS/threading, and they legitimately differ between the fast and full
    configurations (different n). They are printed for information only.

Exit status is non-zero if any scale-invariant check diverges beyond tolerance.

Usage:
  python experiments/compare_fast_vs_full.py
  python experiments/compare_fast_vs_full.py --full results --fast results_fast
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Tolerances for the scale-invariant checks. Slopes estimated from the fast track
# use fewer and smaller n, so allow a modest absolute band on the exponent.
SLOPE_TOL = 0.20      # absolute difference in a log-log slope
MEAN_K_REL_TOL = 0.05  # relative difference in mean k at a shared n
CI_SIGMA = 3.0         # full-test confidence band half-width: CI_SIGMA * sd ...
CI_REL_FLOOR = 0.10    # ... but at least this fraction of the mean (a 5-rep sd can be tiny)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Report:
    def __init__(self):
        self.failures = 0
        self.skipped = []

    def section(self, title):
        print(f"\n{title}\n{'-' * len(title)}")

    def skip(self, what):
        self.skipped.append(what)
        print(f"  (skipped: {what} — file(s) not present in both runs)")

    def check(self, label, full, fast, *, tol, rel=False, informational=False):
        if full is None or fast is None:
            print(f"  {label:<34} full={_fmt(full):>10}  fast={_fmt(fast):>10}  [incomplete]")
            return
        if rel:
            diff = abs(full - fast) / (abs(full) or 1.0)
            shown = f"{diff * 100:5.1f}%"
            ok = diff <= tol
        else:
            diff = abs(full - fast)
            shown = f"{diff:.3f}"
            ok = diff <= tol
        if informational:
            status = "info"
        else:
            status = "OK" if ok else "DIVERGES"
            if not ok:
                self.failures += 1
        print(f"  {label:<34} full={full:>10.3f}  fast={fast:>10.3f}  "
              f"Δ={shown:>7}  [{status}]")

    def info(self, label, full, fast):
        f1 = "  n/a" if full is None else f"{full:>10.3f}"
        f2 = "  n/a" if fast is None else f"{fast:>10.3f}"
        print(f"  {label:<34} full={f1}  fast={f2}  [info only]")

    def band_check(self, label, in_band, total, *, informational=False):
        frac = (in_band / total) if total else 1.0
        ok = in_band == total
        if informational:
            status = "info"
        else:
            status = "OK" if ok else "DIVERGES"
            if not ok:
                self.failures += 1
        print(f"  {label:<34} {in_band}/{total} in band ({frac*100:3.0f}%)              [{status}]")


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def _index(rows, key):
    return {key(r): r for r in rows} if rows else {}


def compare_constk_slopes(rep, full_dir, fast_dir):
    rep.section("Constant-k scaling slopes (near-linear claim)")
    full = load(os.path.join(full_dir, "benchmark_constk_slopes.csv"))
    fast = load(os.path.join(fast_dir, "benchmark_constk_slopes.csv"))
    if full is None or fast is None:
        return rep.skip("benchmark_constk_slopes.csv")
    fi = _index(full, lambda r: r["k_target"])
    si = _index(fast, lambda r: r["k_target"])
    for k in sorted(set(fi) & set(si), key=int):
        rep.check(f"k={k}: time exponent", fnum(fi[k]["slope_time"]), fnum(si[k]["slope_time"]), tol=SLOPE_TOL)
        rep.check(f"k={k}: memory exponent", fnum(fi[k]["slope_mem"]), fnum(si[k]["slope_mem"]), tol=SLOPE_TOL)


def compare_geonames_slopes(rep, full_dir, fast_dir, fname, title):
    rep.section(title)
    full = load(os.path.join(full_dir, fname))
    fast = load(os.path.join(fast_dir, fname))
    if full is None or fast is None:
        return rep.skip(fname + " (run the GeoNames benchmark in both modes to compare)")
    print("  (informational: the fast GeoNames ladder spans a narrow small-n range, 50k-200k vs")
    print("   100k-2M for the full run, so these exponents -- memory especially, where fixed")
    print("   overhead dominates at small n -- are rough estimates, not pass/fail checks.)")
    fi = _index(full, lambda r: r["metric"])
    si = _index(fast, lambda r: r["metric"])
    for m in sorted(set(fi) & set(si)):
        rep.check(f"{m} exponent", fnum(fi[m]["slope"]), fnum(si[m]["slope"]),
                  tol=SLOPE_TOL, informational=True)


def compare_synthetic(rep, full_dir, fast_dir):
    rep.section("Synthetic benchmark: exactness, mean k, and speedups")
    full = load(os.path.join(full_dir, "benchmark_python.csv"))
    fast = load(os.path.join(fast_dir, "benchmark_python.csv"))
    if full is None or fast is None:
        return rep.skip("benchmark_python.csv")

    def exact_frac(rows):
        vals = [r.get("exact_agree", "") for r in rows]
        vals = [v for v in vals if v not in ("", "NA", "None")]
        if not vals:
            return None
        return sum(v == "True" for v in vals) / len(vals)

    rep.check("exactness agreement (frac True)", exact_frac(full), exact_frac(fast), tol=0.0)

    # Mean k and speedups at configurations present in BOTH runs.
    key = lambda r: (r["scenario"], r["n"], r["h_max_km"])
    fi = _index(full, key)
    si = _index(fast, key)
    shared = sorted(set(fi) & set(si), key=lambda t: (t[0], float(t[1]), float(t[2])))
    if not shared:
        print("  (no shared (scenario, n, h_max) configurations to compare)")
        return
    # Mean k is deterministic — average the relative gap across shared configs.
    import statistics
    k_rel = [abs(fnum(fi[c]["mean_k"]) - fnum(si[c]["mean_k"])) / (fnum(fi[c]["mean_k"]) or 1.0)
             for c in shared]
    rep.check("mean k agreement (avg rel gap)", statistics.mean(k_rel), 0.0, tol=MEAN_K_REL_TOL)

    def avg(rows_idx, col):
        xs = [fnum(rows_idx[c][col]) for c in shared if fnum(rows_idx[c].get(col)) is not None]
        return sum(xs) / len(xs) if xs else None

    print(f"  (averaged over {len(shared)} shared configs; speedups are hardware-dependent)")
    rep.info("avg time speedup (x)", avg(fi, "speedup_time"), avg(si, "speedup_time"))
    rep.info("avg memory speedup (x)", avg(fi, "speedup_mem"), avg(si, "speedup_mem"))


def compare_ci(rep, full_dir, fast_dir):
    rep.section("Fast single run inside the full-test confidence band (mean +/- 3 sigma, >=10% floor)")
    full = load(os.path.join(full_dir, "benchmark_python.csv"))
    fast = load(os.path.join(fast_dir, "benchmark_python.csv"))
    if full is None or fast is None:
        return rep.skip("benchmark_python.csv")
    key = lambda r: (r["scenario"], r["n"], r["h_max_km"])
    fi = _index(full, key)
    si = _index(fast, key)
    shared = sorted(set(fi) & set(si), key=lambda t: (t[0], float(t[1]), float(t[2])))
    if not shared:
        return print("  (no shared configurations)")
    print("  (the fast track runs each config once; here we check it lands inside the band the")
    print("   full 5-rep run establishes. Memory is hardware-independent -> pass/fail; timing")
    print("   depends on CPU/RAM/OS -> informational.)")

    def band_count(mean_c, sd_c):
        in_band = total = 0
        for c in shared:
            fm, fs, fv = fnum(fi[c].get(mean_c)), fnum(fi[c].get(sd_c)), fnum(si[c].get(mean_c))
            if fm is None or fv is None:
                continue
            sd = fs if (fs is not None and fs == fs) else 0.0  # nan-safe (fast sd is nan at 1 rep)
            band = max(CI_SIGMA * sd, CI_REL_FLOOR * abs(fm))
            total += 1
            in_band += int(abs(fv - fm) <= band)
        return in_band, total

    ib, tb = band_count("sparse_mem_mean", "sparse_mem_sd")
    rep.band_check("GSHAC memory in band", ib, tb)
    it, tt = band_count("sparse_time_mean", "sparse_time_sd")
    rep.band_check("GSHAC time in band", it, tt, informational=True)


def main():
    ap = argparse.ArgumentParser(description="Contrast fast-track vs full-scale results.")
    root = os.path.join(os.path.dirname(__file__), "..")
    ap.add_argument("--full", default=os.path.join(root, "results"))
    ap.add_argument("--fast", default=os.path.join(root, "results_fast"))
    args = ap.parse_args()

    if not os.path.isdir(args.fast) or not load(os.path.join(args.fast, "benchmark_constk_slopes.csv")):
        print(f"No fast-track results found in {args.fast!r}.\n"
              f"Run `make reproduce-fast` first, then re-run this comparison.", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Fast-track vs full-scale results — alignment check")
    print(f"  full = {os.path.abspath(args.full)}")
    print(f"  fast = {os.path.abspath(args.fast)}")
    print("=" * 72)

    rep = Report()
    compare_constk_slopes(rep, args.full, args.fast)
    compare_geonames_slopes(rep, args.full, args.fast,
                            "benchmark_geonames_geographic_slopes.csv",
                            "GeoNames geographic (constant-density) scaling slopes")
    compare_geonames_slopes(rep, args.full, args.fast,
                            "benchmark_geonames_scaling_slopes.csv",
                            "GeoNames fixed-extent (growing-density) scaling slopes")
    compare_synthetic(rep, args.full, args.fast)
    compare_ci(rep, args.full, args.fast)

    print("\n" + "=" * 72)
    print("NOTE: scaling slopes, exactness, and mean-k are scale-/hardware-invariant and")
    print("are checked against tolerances. Absolute timings and memory depend on CPU, RAM")
    print("headroom, OS, and threading, and are reported for information only.")
    if rep.failures == 0:
        print("RESULT: fast-track estimates ALIGN with the full-scale results. ✓")
    else:
        print(f"RESULT: {rep.failures} scale-invariant quantity(ies) DIVERGE beyond tolerance. ✗")
    print("=" * 72)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
