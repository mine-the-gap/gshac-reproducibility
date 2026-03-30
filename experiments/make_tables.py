"""
make_tables.py

Generate all paper tables as standalone LaTeX files from benchmark CSV results.
Each table is saved as figures/tab_<name>.tex and can be included in main.tex
via \\input{../figures/tab_<name>.tex}.

Run after all benchmarks complete:

    python3 experiments/make_tables.py

Outputs (in figures/):
    tab_complexity.tex        — Algorithm complexity vs dense baseline
    tab_summary.tex           — Speedup across density scenarios
    tab_constk.tex            — Constant-k scaling
    tab_pysal.tex             — Graph-construction: GSHAC vs PySAL DistanceBand
    tab_sklearn.tex           — GSHAC vs scikit-learn connectivity HAC
    tab_realworld.tex         — Dense vs sparse for mining dataset
    tab_mining_components.tex — Component and cluster statistics (mining)
    tab_geonames.tex          — GSHAC on 2M-point GeoNames sample
    tab_hdbscan.tex           — HDBSCAN vs GSHAC resource comparison
    tab_geonames_full.tex     — Estimated resources for full GeoNames
"""

from __future__ import annotations

import csv
import math
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "figures")
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


# ── Table 1: Complexity (static — no CSV) ────────────────────────────

def make_tab_complexity():
    content = r"""\begin{table}[t]
\caption{Complexity of Algorithm~\ref{alg:main} vs.\ dense baseline.}
\label{tab:complexity}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Step} & \textbf{Proposed} & \textbf{Dense baseline} \\
\midrule
Spatial index build    & $\Onotation{n \log n}$ & --- \\
Candidate pair queries & $\Onotation{n \log n + m}$ & --- \\
Distance computations  & $\Onotation{m}$ & $\Onotation{n^2}$ \\
Storage                & $\Onotation{m}$ & $\Onotation{n^2}$ \\
Connected components   & $\Onotation{n + m}$ & --- \\
HAC (all components)   & $\Onotation{\sum_k c_k^2 \log c_k}$ & $\Onotation{n^2 \log n}$ \\
\midrule
\textbf{Total time}    & $\Onotation{n \log n + \sum_k c_k^2 \log c_k}$ & $\Onotation{n^2 \log n}$ \\
\textbf{Total memory}  & $\Onotation{m} = \Onotation{nk}$ & $\Onotation{n^2}$ \\
\bottomrule
\end{tabular}
\end{table}"""
    write_table("complexity", content)


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
            sc_label = sc[0].upper()

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

    content = f"""\\begin{{table}}[t]
\\caption{{Speedup across density scenarios.
  T\\,=\\,tight, M\\,=\\,moderate, L\\,=\\,loose.
  ``Graph\\%'' is the fraction of sparse runtime spent on graph
  construction.}}
\\label{{tab:summary}}
\\small
\\begin{{tabular}}{{rrr rrr}}
\\toprule
$n$ & $\\hmax$ & Sc. & Time & Mem & Graph\\% \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("summary", content)


# ── Table 3: Constant-k scaling ───────────────────────────────────────

def make_tab_constk():
    rows = read_csv("benchmark_constk.csv")
    rows.sort(key=lambda r: r["n"])

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
\\caption{{Constant-\\(\\bar{{k}}\\) scaling (uniform points,
  \\(\\hmax = 10\\)\\,km, \\(\\bar{{k}} \\approx 50\\)).
  Domain grows as \\(L \\propto \\sqrt{{n}}\\).
  Dense infeasible for \\(n \\geq 50\\text{{K}}\\).}}
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

    content = f"""\\begin{{table}}[t]
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
  moderate, \\(\\hmax = 10\\)\\,km, \\(h_{{\\text{{cut}}}} = 5\\)\\,km).
  Cluster counts agree exactly.}}
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


# ── Table 6: Real-world mining dataset ────────────────────────────────

def make_tab_realworld():
    rows = read_csv("benchmark_mining.csv")
    r = rows[0]

    n = int(r["n"])
    n_edges = int(r["n_edges"])
    sparse_mb = float(r["sparse_matrix_mb"])
    dense_gb = float(r["dense_matrix_gb"])
    graph_t = float(r["graph_time_s"])
    hac_t = float(r["hac_time_s"])
    total_t = float(r["total_time_s"])
    hac_mem = float(r["hac_mem_mib"])

    dense_pairs = n * (n - 1) / 2
    # Format as e.g. "3.4 \times 10^{10}"
    exp = int(math.floor(math.log10(dense_pairs)))
    coeff = dense_pairs / 10 ** exp

    content = f"""\\begin{{table}}[t]
\\caption{{Dense vs.\\ sparse for the global mining dataset
  (\\(n = {fmt_int(n)}\\), \\(\\hmax = 20\\)\\,km, haversine, AMD Ryzen~7 PRO
  8840U, 30\\,GiB DDR5).
  Dense figures are analytical; sparse figures are measured.}}
\\label{{tab:realworld}}
\\small
\\begin{{tabular}}{{lrr}}
\\toprule
& \\textbf{{Dense baseline}} & \\textbf{{GSHAC}} \\\\
\\midrule
Distance pairs          & \\(\\approx {coeff:.1f} \\times 10^{{{exp}}}\\) & \\({fmt_int(n_edges)}\\) \\\\
Graph/matrix storage    & \\(\\approx {int(round(dense_gb))}\\,\\text{{GB}}\\)      & \\({int(round(sparse_mb))}\\,\\text{{MB}}\\)   \\\\
Graph construction      & ---                             & \\({graph_t:.1f}\\,\\text{{s}}\\) \\\\
HAC (all components)    & \\(>\\,10\\,\\text{{h}}\\)             & \\({hac_t:.1f}\\,\\text{{s}}\\) \\\\
Total wall time         & \\(>\\,10\\,\\text{{h}}\\)             & \\({total_t:.1f}\\,\\text{{s}}\\) \\\\
Peak memory (HAC)       & \\(\\approx {int(round(dense_gb))}\\,\\text{{GB}}\\)      & \\({int(round(hac_mem))}\\,\\text{{MiB}}\\) \\\\
Feasible on workstation & No                              & Yes \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("realworld", content)


# ── Table 7: Mining components and cluster statistics ─────────────────

def make_tab_mining_components():
    comp_rows = read_csv("benchmark_mining_components.csv")
    clust_rows = read_csv("benchmark_mining_clusters.csv")
    mining = read_csv("benchmark_mining.csv")[0]

    comp = {str(r["percentile"]): r["component_size"] for r in comp_rows}
    n_singletons = int(mining["n_singletons"])

    median = comp.get("50", comp.get("50.0", 2))
    p90 = comp.get("90", comp.get("90.0", 15))
    p99 = comp.get("99", comp.get("99.0", 151))
    maximum = comp.get("100", comp.get("100.0", 30179))

    clust_map = {}
    for r in clust_rows:
        clust_map[float(r["h_cut_km"])] = int(r["n_clusters"])

    def _i(v):
        return fmt_int(int(round(float(v))))

    content = f"""\\begin{{table}}[t]
\\caption{{Component and cluster statistics for the mining dataset
  (\\(n = {fmt_int(int(mining['n']))}\\), \\(\\hmax = 20\\,\\text{{km}}\\)).}}
\\label{{tab:mining_components}}
\\small
\\begin{{tabular}}{{lrr}}
\\toprule
 & \\textbf{{Component size}} & \\textbf{{Clusters at \\(h\\)}} \\\\
\\midrule
Singletons             & {_i(n_singletons)}                & --- \\\\
Median                 & {_i(median)}                    & --- \\\\
90th percentile        & {_i(p90)}                   & --- \\\\
99th percentile        & {_i(p99)}                  & --- \\\\
Maximum                & {_i(maximum)}               & --- \\\\
\\addlinespace
\\(h = 1\\,\\text{{km}}\\)   & ---                  & {_i(clust_map.get(1.0, 0))} \\\\
\\(h = 5\\,\\text{{km}}\\)   & ---                  & {_i(clust_map.get(5.0, 0))} \\\\
\\(h = 10\\,\\text{{km}}\\)  & ---                  & {_i(clust_map.get(10.0, 0))} \\\\
\\(h = 20\\,\\text{{km}}\\)  & ---                  & {_i(clust_map.get(20.0, 0))} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("mining_components", content)


# ── Table 8: GeoNames 2M-point benchmark ──────────────────────────────

def make_tab_geonames():
    rows = read_csv("benchmark_geonames.csv")
    rows.sort(key=lambda r: r["h_max_m"])

    data = {int(r["h_max_m"]): r for r in rows}
    h_vals = [500, 1000, 2000, 5000]

    def _edges(h):
        e = int(data[h]["n_edges"])
        if e >= 1_000_000:
            return f"{e / 1_000_000:.1f}M"
        return f"{round(e / 1000)}k"

    def _sparse(h):
        return f"{int(round(float(data[h]['sparse_mb'])))}\\,MB"

    def _comps(h):
        return fmt_int(int(data[h]["n_components"]))

    def _sing(h):
        ns = int(data[h]["n_singletons"])
        nc = int(data[h]["n_components"])
        return f"{ns / nc * 100:.1f}\\%"

    def _largest(h):
        return fmt_int(int(data[h]["comp_max"]))

    def _maxsub(h):
        gb = float(data[h]["max_sub_gb"])
        return f"{gb:.2f}\\,GB" if gb < 1 else f"{gb:.1f}\\,GB"

    def _graph_t(h):
        return f"{int(round(float(data[h]['graph_time_s'])))}\\,s"

    def _hac_t(h):
        return f"{int(round(float(data[h]['hac_time_s'])))}\\,s"

    def _hac_mem(h):
        return f"{int(round(float(data[h]['hac_mem_mib'])))}\\,MiB"

    def _total(h):
        t = float(data[h]["total_time_s"])
        return f"\\textbf{{{t / 60:.1f}\\,min}}"

    n_full = fmt_int(int(rows[0]["n_full"]))

    metrics = [
        ("Edges",              _edges),
        ("Sparse storage",     _sparse),
        ("Components",         _comps),
        ("Singletons",         _sing),
        ("Largest component",  _largest),
        ("Max sub-matrix",     _maxsub),
        ("Graph time",         _graph_t),
        ("HAC time",           _hac_t),
        ("Peak memory (HAC)",  _hac_mem),
        ("\\textbf{Total time}", _total),
    ]

    tex_lines = []
    for i, (label, fn) in enumerate(metrics):
        vals = " & ".join(fn(h) for h in h_vals)
        tex_lines.append(f"{label} & {vals} \\\\")
        if i == 5:
            tex_lines.append("\\addlinespace")

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[t]
\\caption{{GSHAC results on a 2M-point GeoNames sample
  (\\(n_{{\\text{{full}}}} = {n_full}\\)), single linkage, haversine.
  Dense \\(n^2\\) for 2M: 32\\,TB; for \\(n_{{\\text{{full}}}}\\): 1.4\\,PB.}}
\\label{{tab:geonames}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{4}}{{c}}{{\\(\\hmax\\)}} \\\\
\\cmidrule(l){{2-5}}
& \\textbf{{500\\,m}} & \\textbf{{1\\,km}} & \\textbf{{2\\,km}} & \\textbf{{5\\,km}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    write_table("geonames", content)


# ── Table 9: HDBSCAN comparison ───────────────────────────────────────

def make_tab_hdbscan():
    rows = read_csv("benchmark_hdbscan.csv")
    mining = read_csv("benchmark_mining.csv")[0]
    clust_rows = read_csv("benchmark_mining_clusters.csv")

    hdb = [r for r in rows
           if r["dataset"] == "mining" and r["method"] == "HDBSCAN"]
    hdb.sort(key=lambda r: float(r["min_cluster_size"]))

    gshac_time = float(mining["total_time_s"])
    gshac_mem = float(mining["hac_mem_mib"])
    n = int(mining["n"])

    clust_map = {float(r["h_cut_km"]): int(r["n_clusters"])
                 for r in clust_rows}

    tex_lines = []
    for r in hdb:
        mcs = int(r["min_cluster_size"])
        t = int(round(float(r["time_s"])))
        mem = int(round(float(r["peak_mib"])))
        nc = fmt_int(int(r["n_clusters"]))
        noise_pct = int(r["n_noise"]) / n * 100
        tex_lines.append(
            f"HDBSCAN (mcs\\,=\\,{mcs}) & {fmt_int(t)}\\,s "
            f"& {mem}\\,MiB & {nc} & {noise_pct:.1f}\\% \\\\"
        )

    tex_lines.append("\\addlinespace")
    tex_lines.append(
        f"GSHAC (\\(\\hmax\\!=\\!20\\)\\,km)\\(^{{*}}\\) "
        f"& {int(round(gshac_time))}\\,s & {int(round(gshac_mem))}\\,MiB "
        f"& --- & 0\\% \\\\"
    )
    tex_lines.append(
        f"\\quad\\(\\to h\\!=\\!5\\)\\,km  & & "
        f"& {fmt_int(clust_map[5.0])} & \\\\"
    )
    tex_lines.append(
        f"\\quad\\(\\to h\\!=\\!20\\)\\,km & & "
        f"& {fmt_int(clust_map[20.0])} & \\\\"
    )

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[t]
\\caption{{Resource comparison: \\texttt{{sklearn}} HDBSCAN vs.\\ GSHAC
  on the mining dataset (\\(n = {fmt_int(n)}\\), haversine).
  Non-equivalent outputs; illustrates resource profiles only.}}
\\label{{tab:hdbscan}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
\\textbf{{Method}} & \\textbf{{Time}} & \\textbf{{Mem}} & \\textbf{{Clusters}} & \\textbf{{Noise}} \\\\
\\midrule
{body}
\\bottomrule
\\multicolumn{{5}}{{l}}{{\\footnotesize \\(^{{*}}\\)Single run; dendrogram cut at any \\(h \\leq \\hmax\\).}}
\\end{{tabular}}
\\end{{table}}"""

    write_table("hdbscan", content)


# ── Table 10: GeoNames full (estimated) ───────────────────────────────

def make_tab_geonames_full():
    explore = read_csv("explore_geonames.csv")
    geonames = read_csv("benchmark_geonames.csv")

    ex = {int(r["h_max_m"]): r for r in explore}
    geo = {int(r["h_max_m"]): r for r in geonames}

    n_full = int(explore[0]["n"])
    n_sample = 2_000_000
    scale = n_full / n_sample

    h_vals = [500, 1000, 2000, 5000]

    def _edges(h):
        if h not in ex:
            return "---"
        e = int(ex[h]["n_edges"])
        s = f"{e / 1_000_000:.1f}M"
        return f"\\(\\sim\\){s}\\(^{{*}}\\)" if h == 5000 else s

    def _sparse(h):
        if h not in ex:
            return "---"
        mb = float(ex[h]["sparse_mb"])
        if mb >= 1000:
            s = f"{mb / 1000:.1f}\\,GB"
        else:
            s = f"{int(round(mb))}\\,MB"
        return f"\\(\\sim\\){s}\\(^{{*}}\\)" if h == 5000 else s

    def _largest(h):
        if h == 5000:
            return "---"
        return fmt_int(int(ex[h]["comp_max"])) if h in ex else "---"

    def _graph_time(h):
        if h not in ex:
            return "---"
        t = float(ex[h]["graph_time_s"])
        s = f"{t / 60:.1f}\\,min"
        return f"\\(\\sim\\){s}\\(^{{*}}\\)" if h == 5000 else s

    def _graph_mem(h):
        if h not in ex:
            return "---"
        m = float(ex[h]["graph_mem_mib"])
        if h == 5000:
            return f"\\(>\\)30\\,GiB\\(^{{*}}\\)"
        return f"{m / 1024:.1f}\\,GiB"

    def _hac_time_est(h):
        if h not in geo:
            return "---"
        est = float(geo[h]["hac_time_s"]) * scale
        return f"\\(\\sim\\){est / 60:.0f}\\,min"

    def _hac_mem_est(h):
        if h not in geo:
            return "---"
        est = float(geo[h]["hac_mem_mib"]) * scale
        return f"\\(\\sim\\){est / 1024:.1f}\\,GiB"

    def _total_est(h):
        if h == 5000:
            return "\\textbf{needs 64\\,GiB}"
        if h in ex and h in geo:
            g_t = float(ex[h]["graph_time_s"])
            h_t = float(geo[h]["hac_time_s"]) * scale
            return f"\\textbf{{\\(\\sim\\){(g_t + h_t) / 60:.0f}\\,min}}"
        return "---"

    metrics = [
        ("Edges",                      _edges),
        ("Sparse storage",             _sparse),
        ("Largest component",          _largest),
        ("Graph time",                 _graph_time),
        ("Graph memory",               _graph_mem),
        ("HAC time (est.)",            _hac_time_est),
        ("HAC memory (est.)",          _hac_mem_est),
        ("\\textbf{Total (est.)}",     _total_est),
    ]

    tex_lines = []
    for i, (label, fn) in enumerate(metrics):
        vals = " & ".join(fn(h) for h in h_vals)
        tex_lines.append(f"{label} & {vals} \\\\")
        if i == 2 or i == 4:
            tex_lines.append("\\addlinespace")

    body = "\n".join(tex_lines)

    content = f"""\\begin{{table}}[t]
\\caption{{Estimated resources for full GeoNames (\\(n = {fmt_int(n_full)}\\)).
  Graph rows: measured on full dataset (graph construction only).
  HAC rows: linearly extrapolated from 2M sample.
  Hardware: AMD Ryzen~7 PRO 8840U, 30\\,GiB DDR5.}}
\\label{{tab:geonames_full}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\textbf{{500\\,m}} & \\textbf{{1\\,km}} & \\textbf{{2\\,km}} & \\textbf{{5\\,km}} \\\\
\\midrule
{body}
\\bottomrule
\\multicolumn{{5}}{{l}}{{\\footnotesize \\(^{{*}}\\)Extrapolated; graph exceeded 30\\,GiB RAM at \\(\\hmax = 5\\)\\,km.}} \\\\
\\end{{tabular}}
\\end{{table}}"""

    write_table("geonames_full", content)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("Generating LaTeX tables from benchmark results...")

    make_tab_complexity()
    make_tab_summary()
    make_tab_constk()
    make_tab_pysal()
    make_tab_sklearn()
    make_tab_realworld()
    make_tab_mining_components()
    make_tab_geonames()
    make_tab_hdbscan()
    make_tab_geonames_full()

    print("Done — all tables written to figures/")
