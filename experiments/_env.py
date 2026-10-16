"""Harness environment hardening: pin all thread pools to one thread.

Import this module FIRST (before numpy/scipy/sklearn) in every benchmark
script so that BLAS/OpenMP thread pools cannot silently parallelise any
timed region: the manuscript reports single-core, like-for-like
comparisons. None of the timed kernels (cKDTree.query_pairs, BallTree,
fastcluster, the gshac C extension) is multithreaded, so this is a
guard rail rather than a behaviour change; ``setdefault`` keeps the
variables overridable for deliberate experiments.
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Pin the memory-paging state. NumPy madvise()s transparent huge pages for
# large arrays; whether the kernel can grant them depends on memory
# fragmentation at run time, which makes the first-touch page-fault cost of
# big allocations bimodal (measured 2026-08-13: the cold dense baseline at
# n=25,000 takes ~3.5 s with 2 MiB huge pages vs ~7.6 s with 4 KiB pages,
# ~39K vs ~2.5M minor faults). Disabling the madvise pins every measurement
# to the deterministic 4 KiB-page state regardless of machine state. Note:
# only effective under THP policy "madvise" (the default on this machine);
# a system configured with THP "always" may still grant huge pages.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
