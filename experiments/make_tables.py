"""
make_tables.py

Generate all paper tables as standalone LaTeX files from benchmark CSV results.
Each table is saved as paper/tab_<name>.tex and can be included in main.tex
via \\input{tab_<name>.tex}.

Run after all benchmarks complete:

    python3 experiments/make_tables.py

Outputs (in paper/):
    tab_summary.tex           — Speedup across density scenarios
    tab_constk.tex            — Constant-k scaling
    tab_pysal.tex             — Graph-construction: GSHAC vs PySAL DistanceBand
    tab_sklearn.tex           — GSHAC vs scikit-learn connectivity HAC
    tab_geonames.tex          — GSHAC on 2M-point GeoNames sample
    tab_hdbscan.tex           — HDBSCAN vs GSHAC resource comparison
"""

from __future__ import annotations

import csv
import math
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "paper")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────

def read_csv(filename):
    """Read a CSV from the results directory, auto-converting numeric fields."""
    path = os.path.join(RESULTS_DIR, filename)
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for key in r:
                try:
                    r[key] = float(r[key])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def write_table(name, content):
    """Write a .tex file to the figures directory."""
    path = os.path.join(FIGURES_DIR, f"tab_{name}.tex")
    with open(path, "w") as f:
        f.write(content)
    print(f"  Saved {path}")


def fmt_int(v):
    """Format an integer with thousands separators for LaTeX."""
    if isinstance(v, float):
        v = int(round(v))
    s = f"{v:,}"
    return s.replace(",", "{,}")


def fmt_f(v, decimals=1):
    """Format a float with given decimals."""
    return f"{v:.{decimals}f}"


def fmt_speedup(v, bold_max=None):
    """Format a speedup value like 4.77\\times, optionally bold."""
    s = f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
    if bold_max is not None and abs(v - bold_max) < 0.005:
        return f"$\\mathbf{{{s}\\times}}$"
    return f"${s}\\times$"

# ── Table 2: Summary — speedup across density scenarios ───────────────

def make_tab_summary():
    rows = read_csv("benchmark_python.csv")

    configs = [
        (10000, 10.0),
        (10000, 20.0),
        (25000, 10.0),
        (25000, 20.0),
    ]
    scenarios = ["tight", "moderate", "loose"]

    # Find best speedups for bolding
    all_time_sp = []
    all_mem_sp = []
    for n, h in configs:
        for sc in scenarios:
            r = [x for x in rows
                 if x["n"] == n and x["h_max_km"] == h and x["scenario"] == sc]
            if r and r[0]["speedup_time"] != "":
                all_time_sp.append(float(r[0]["speedup_time"]))
            if r and r[0]["speedup_mem"] != "":
                all_mem_sp.append(float(r[0]["speedup_mem"]))

    best_time = max(all_time_sp) if all_time_sp else 0
    best_mem = max(all_mem_sp) if all_mem_sp else 0

    tex_lines = []
    for idx, (n, h) in enumerate(configs):
        for j, sc in enumerate(scenarios):
            r = [x for x in rows
                 if x["n"] == n and x["h_max_km"] == h and x["scenario"] == sc]
            if not r:
                continue
            r = r[0]

            sp_t = float(r["speedup_time"])
            sp_m = float(r["speedup_mem"])
            graph_pct = float(r["graph_time_frac"]) * 100
            sc_label = sc.capitalize()

            if j == 0:
                n_col = f"\\multirow{{3}}{{*}}{{{fmt_int(n)}}}"
                h_col = f"\\multirow{{3}}{{*}}{{{int(h):.0f} km}}"
            else:
                n_col = ""
                h_col = ""

            time_str = fmt_speedup(sp_t, best_time)
            mem_str = fmt_speedup(sp_m, best_mem)

            tex_lines.append(
                f" {n_col} & {h_col} & {sc_label} "
                f"& {time_str} & {mem_str} & {int(round(graph_pct))}\\% \\\\"
            )

        if idx < len(configs) - 1:
            tex_lines.append("\\addlinespace")

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[ht]
\\caption{{Speedup across density scenarios.
  ``Graph\\%'' is the fraction of sparse runtime spent on graph
  construction.}}
\\label{{tab:summary}}
\\small
\\begin{{tabular}}{{rrr rrr}}
\\toprule
$n$ & $\\hmax$ & Scenario & Time & Mem & Graph\\% \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("summary", content)


# ── Table 3: Constant-k scaling ───────────────────────────────────────

def make_tab_constk():
    # The CSV may contain multiple k_target values from the slope-invariance
    # sweep; tab_constk shows the primary k=126 row by row, with the other
    # k values left to fig_constk.
    PRIMARY_K = 126
    all_rows = read_csv("benchmark_constk.csv")
    primary = [r for r in all_rows
               if int(float(r.get("k_target", 0) or 0)) == PRIMARY_K]
    rows = sorted(primary if primary else all_rows, key=lambda r: r["n"])

    # Find best speedup for bolding
    speedups = [float(r["speedup_time"]) for r in rows
                if r.get("speedup_time", "") != ""
                and not (isinstance(r["speedup_time"], float)
                         and math.isnan(r["speedup_time"]))]
    best_sp = max(speedups) if speedups else 0

    tex_lines = []
    added_space = False
    for r in rows:
        n = int(r["n"])
        k = int(round(r["mean_k"]))
        sp_time = r["sparse_time_mean"]
        dn_time = r["dense_time_mean"]
        sp_mem = r["sparse_mem_mean"]
        dn_mem = r["dense_mem_mean"]
        speedup = r.get("speedup_time", "")

        has_dense = (dn_time != ""
                     and not (isinstance(dn_time, float) and math.isnan(dn_time)))

        if not has_dense and not added_space:
            tex_lines.append("\\addlinespace")
            added_space = True

        n_str = fmt_int(n)

        if has_dense:
            dn_t_str = fmt_f(float(dn_time), 3)
            dn_m_str = fmt_int(int(round(float(dn_mem))))
            sp_t_str = fmt_f(float(sp_time), 3)
            sp_m_str = fmt_int(int(round(float(sp_mem))))
            sp_str = fmt_speedup(float(speedup), best_sp)
        else:
            dn_t_str = "---"
            dn_m_str = "---"
            sp_t_str = fmt_f(float(sp_time), 3)
            sp_m_str = fmt_int(int(round(float(sp_mem))))
            sp_str = "---"

        tex_lines.append(
            f"{n_str} & {k} & {dn_t_str} & {sp_t_str} "
            f"& {dn_m_str} & {sp_m_str} & {sp_str} \\\\"
        )

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[t]
\\caption{{Constant-\\(\\bar{{k}}\\) scaling at the primary
  configuration (uniform points, \\(\\hmax = 20\\)\\,km,
  \\(\\bar{{k}} \\approx 126\\)). Domain grows as
  \\(L \\propto \\sqrt{{n}}\\). Dense infeasible for
  \\(n \\geq 50\\text{{K}}\\). Slope-invariance across
  \\(\\bar{{k}}_{{\\text{{target}}}} \\in \\{{20, 50, 126, 200\\}}\\)
  is shown in Figure~\\ref{{fig:constk}}.}}
\\label{{tab:constk}}
\\small
\\begin{{tabular}}{{rrrrrrr}}
\\toprule
\\(n\\) & \\(\\bar{{k}}\\) & \\multicolumn{{2}}{{c}}{{Time (s)}} & \\multicolumn{{2}}{{c}}{{Memory (MiB)}} & Time \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
 &  & Dense & GSHAC & Dense & GSHAC & speedup \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("constk", content)


# ── Table 4: PySAL graph-construction comparison ──────────────────────

def make_tab_pysal():
    rows = read_csv("benchmark_pysal.csv")

    target_ns = [1000, 5000, 10000, 25000]
    scenarios = ["tight", "moderate", "loose"]

    tex_lines = []
    for i, sc in enumerate(scenarios):
        sub = [r for r in rows
               if r["scenario"] == sc and r["h_max_km"] == 10.0
               and int(r["n"]) in target_ns]
        sub.sort(key=lambda r: r["n"])

        for r in sub:
            n = int(r["n"])
            our = float(r["graph_our_s"])
            dband = float(r["graph_dband_s"])
            ratio = float(r["dband_our_ratio"])

            tex_lines.append(
                f"    {sc.capitalize()} & {fmt_int(n)} & {our:.3f} "
                f"& {dband:.3f} & {ratio:.0f}$\\times$ \\\\"
            )

        if i < len(scenarios) - 1:
            tex_lines.append("    \\addlinespace")

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[ht]
  \\centering
  \\caption{{Graph-construction time: GSHAC \\texttt{{query\\_pairs}}
           vs.\\ PySAL \\texttt{{DistanceBand}}, \\(h_{{\\max}}=10\\)\\,km.}}
  \\label{{tab:pysal}}
  \\small
  \\begin{{tabular}}{{lrrrr}}
    \\toprule
    Scenario & \\(n\\) & GSHAC (s) & DistanceBand (s) & Ratio \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

    write_table("pysal", content)


# ── Table 5: scikit-learn comparison ──────────────────────────────────

def make_tab_sklearn():
    rows = read_csv("benchmark_sklearn_timing.csv")
    rows.sort(key=lambda r: r["n"])

    tex_lines = []
    for r in rows:
        n = int(r["n"])
        our = float(r["our_time_mean"])
        skl = float(r["sklearn_conn_time_mean"])
        sp = float(r["speedup"])
        nc = int(r["n_clusters_ours"])

        tex_lines.append(
            f"{fmt_int(n)} & {our:.3f} & {skl:.3f} "
            f"& {sp:.1f}$\\times$ & {nc} \\\\"
        )

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[t]
\\caption{{GSHAC vs.\\ scikit-learn connectivity HAC (single linkage,
  moderate, \\(\\hmax = 10\\)\\,km, \\(h_{{\\text{{cut}}}} = 5\\)\\,km), timed
  \\emph{{end-to-end}} (sparse-structure construction + clustering) so both
  sides are measured symmetrically. Both recover the identical partition
  (ARI\\,=\\,1.0).}}
\\label{{tab:sklearn}}
\\small
\\begin{{tabular}}{{r r r r r}}
\\toprule
\\(n\\) & GSHAC (s) & sklearn (s) & Ratio & Clusters \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("sklearn", content)


# ── Table 8: GeoNames 2M-point benchmark ──────────────────────────────

def make_tab_geonames():
    rows = read_csv("benchmark_geonames.csv")
    rows.sort(key=lambda r: r["h_max_m"])

    data = {int(r["h_max_m"]): r for r in rows}
    h_vals = sorted(data.keys())   # adapt to whatever thresholds are present

    def _mem(mib):
        mib = float(mib)
        if mib >= 1024:
            return f"{mib / 1024:.1f}\\,GiB"
        return f"{int(round(mib))}\\,MiB"

    def _edges(h):
        e = int(data[h]["n_edges"])
        if e >= 1_000_000:
            return f"{e / 1_000_000:.1f}M"
        return f"{round(e / 1000)}k"

    def _sparse(h):
        return _mem(data[h]["sparse_mib"])

    def _comps(h):
        return fmt_int(int(data[h]["n_components"]))

    def _sing(h):
        ns = int(data[h]["n_singletons"])
        nc = int(data[h]["n_components"])
        return f"{ns / nc * 100:.1f}\\%"

    def _largest(h):
        return fmt_int(int(data[h]["comp_max"]))

    def _maxsub(h):
        gib = float(data[h]["max_sub_full_gib"])
        if gib >= 1024:
            return f"{gib / 1024:.1f}\\,TiB"
        if gib < 1:
            return f"{gib:.2f}\\,GiB"
        return f"{gib:.1f}\\,GiB"

    def _graph_t(h):
        return f"{int(round(float(data[h]['graph_time_s'])))}\\,s"

    def _hac_t(h):
        return f"{int(round(float(data[h]['hac_time_s'])))}\\,s"

    def _graph_mem(h):
        return _mem(data[h]["graph_mem_mib"])

    def _hac_mem(h):
        return _mem(data[h]["hac_mem_mib"])

    def _total(h):
        t = float(data[h]["total_time_s"])
        return f"\\textbf{{{t / 60:.1f}\\,min}}"

    n_full = fmt_int(int(rows[0]["n_full"]))
    n_reps = int(rows[0]["n_reps"])
    dense_full_tib = float(rows[0]["dense_full_gib_sample"]) / 1024
    dense_cond_tib = float(rows[0]["dense_condensed_gib_sample"]) / 1024

    metrics = [
        ("Edges",                 _edges),
        ("Sparse storage",        _sparse),
        ("Components",            _comps),
        ("Singletons",            _sing),
        ("Largest component",     _largest),
        ("Max sub-matrix",        _maxsub),
        ("Graph time",            _graph_t),
        ("HAC time",              _hac_t),
        ("Peak memory (graph)",   _graph_mem),
        ("Peak memory (HAC)",     _hac_mem),
        ("\\textbf{Total time}",  _total),
    ]

    tex_lines = []
    for i, (label, fn) in enumerate(metrics):
        vals = " & ".join(fn(h) for h in h_vals)
        tex_lines.append(f"{label} & {vals} \\\\")
        if i == 5:
            tex_lines.append("\\addlinespace")

    body = "\n".join(tex_lines)
    ncol = len(h_vals)
    col_headers = " & ".join(f"\\textbf{{{h // 1000}\\,km}}" for h in h_vals)

    content = f"""\\begin{{table}}[t]
\\caption{{GSHAC results on a 2M-point GeoNames sample
  (\\(n_{{\\text{{full}}}} = {n_full}\\)), single linkage, haversine,
  mean over {n_reps} repetitions.
  All rows are measured runs except `Max sub-matrix', the analytical dense size
  the MST path avoids materialising; the dense \\(n^2\\) baseline
  (\\({dense_full_tib:.0f}\\,\\text{{TiB}}\\) full /
  \\({dense_cond_tib:.0f}\\,\\text{{TiB}}\\) condensed at 2M) is analytical and
  not run. \\emph{{Peak memory (graph)}} is the binding peak; the MST path keeps
  \\emph{{Peak memory (HAC)}} far below it, and the 20\\,km graph sits at the
  workstation's memory limit.}}
\\label{{tab:geonames}}
\\small
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'r' * ncol}}}
\\toprule
& \\multicolumn{{{ncol}}}{{c}}{{\\(\\hmax\\)}} \\\\
\\cmidrule(l){{2-{ncol + 1}}}
& {col_headers} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    write_table("geonames", content)


# ── Table 9: HDBSCAN comparison ───────────────────────────────────────

def make_tab_hdbscan():
    rows = read_csv("benchmark_hdbscan.csv")

    hdb = [r for r in rows
           if r["dataset"] == "geonames" and r["method"] == "HDBSCAN"]
    hdb.sort(key=lambda r: float(r["min_cluster_size"]))
    # GSHAC measured in the SAME run, same whole-pipeline peak definition.
    gshac = next(r for r in rows
                 if r["dataset"] == "geonames"
                 and r["method"] == "GSHAC_sparse_single")

    n = int(gshac["n"])
    n_reps = int(gshac["n_reps"])
    hdb_reps = int(hdb[0]["n_reps"]) if hdb else 1
    hdb_reps_desc = "single run" if hdb_reps == 1 else f"mean over {hdb_reps} reps"
    c5 = int(float(gshac["n_clusters_h5km"]))
    c10 = int(float(gshac["n_clusters_h10km"]))

    def _mem(mib):
        mib = float(mib)
        return (f"{mib / 1024:.1f}\\,GiB" if mib >= 1024
                else f"{int(round(mib))}\\,MiB")

    def _time(s):
        return f"{fmt_int(int(round(float(s))))}\\,s"

    tex_lines = []
    for r in hdb:
        mcs = int(float(r["min_cluster_size"]))
        nc = fmt_int(int(r["n_clusters"]))
        noise_pct = float(r["noise_pct"])
        tex_lines.append(
            f"HDBSCAN (mcs\\,=\\,{mcs}) & {_time(r['time_s'])} "
            f"& {_mem(r['pipeline_mem_mib'])} & {nc} & {noise_pct:.1f}\\% \\\\"
        )

    g_pipeline = float(gshac["pipeline_mem_mib"])
    g_graph = float(gshac["graph_mem_mib"])
    g_hac = float(gshac["hac_mem_mib"])

    tex_lines.append("\\addlinespace")
    tex_lines.append(
        f"GSHAC (\\(\\hmax\\!=\\!10\\)\\,km)\\(^{{*}}\\) "
        f"& {_time(gshac['time_s'])} & {_mem(g_pipeline)} & --- & 0\\% \\\\"
    )
    tex_lines.append(
        f"\\quad\\(\\to h\\!=\\!5\\)\\,km  & & & {fmt_int(c5)} & \\\\"
    )
    tex_lines.append(
        f"\\quad\\(\\to h\\!=\\!10\\)\\,km & & & {fmt_int(c10)} & \\\\"
    )

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[!h]
\\caption{{Resource comparison: \\texttt{{sklearn}} HDBSCAN vs.\\ GSHAC on a 100k
  subset of the GeoNames sample (\\(n = {fmt_int(n)}\\), haversine), both measured
  in this run with the same whole-pipeline peak-memory definition.
  Non-equivalent outputs; illustrates resource profiles only. The gap widens with
  \\(n\\): HDBSCAN scales super-linearly (extrapolated to hours at 2M), GSHAC
  near-linearly (Table~\\ref{{tab:geonames}}).}}
\\label{{tab:hdbscan}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
\\textbf{{Method}} & \\textbf{{Time}} & \\textbf{{Peak mem}} & \\textbf{{Clusters}} & \\textbf{{Noise}} \\\\
\\midrule
{body}
\\bottomrule
\\multicolumn{{5}}{{l}}{{\\footnotesize \\(^{{*}}\\)GSHAC: mean over {n_reps} reps; HDBSCAN: {hdb_reps_desc}. One dendrogram cut at any \\(h \\leq \\hmax\\).}} \\\\
\\multicolumn{{5}}{{l}}{{\\footnotesize GSHAC peak is whole-pipeline: graph phase {_mem(g_graph)}, MST-path HAC phase {_mem(g_hac)}.}}
\\end{{tabular}}
\\end{{table}}"""

    write_table("hdbscan", content)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("Generating LaTeX tables from benchmark results...")

    make_tab_summary()
    make_tab_constk()
    make_tab_pysal()
    make_tab_sklearn()
    make_tab_geonames()
    make_tab_hdbscan()

    print("Done — all tables written to paper/")
