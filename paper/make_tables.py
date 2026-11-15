"""Builds the paper's LaTeX tables from the CSV files written by aggregate.py (run: python3 make_tables.py ASSETS_DIR)."""
import os, sys, re
import numpy as np
import pandas as pd

A = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)
ATT = ["BadNet", "AddSent", "SynBkd", "StyleBkd"]
DS = ["SST-2", "AG News", "OLID", "HSOL"]


def tex(s):
    s = "--" if (s is None or (isinstance(s, float) and np.isnan(s)) or str(s) in ("–", "nan")) else str(s)
    return s.replace("±", "$\\pm$").replace("%", "\\%").replace("→", "$\\rightarrow$").replace("–", "--")


def short(s):
    """'12.3 ± 4.5' -> '12.3\\std{4.5}' for compact tables."""
    s = str(s)
    m = re.fullmatch(r"\s*(-?[\d.]+) ± ([\d.]+)\s*", s)
    if m:
        return f"{m.group(1)}\\std{{{m.group(2)}}}"
    return tex(s)


def pcell(v, sd=True):
    """Compact a pooled cell: 'a ± b' -> a\\std{b} (sd=True) or a (sd=False); arrows -> $\\to$."""
    v = str(v)
    v = PM.sub((lambda m: f"{m.group(1)}@@STD{m.group(2)}@@") if sd else (lambda m: m.group(1)), v)
    v = v.replace(" of runs", "").replace(" → ", "→")
    v = tex(v).replace("$\\rightarrow$", "$\\to$")
    return re.sub(r"@@STD([\d.]+)@@", r"\\std{\1}", v)


def write(name, body):
    with open(os.path.join(OUT, f"{name}.tex"), "w") as f:
        f.write(body)


def order(df, cols=("Dataset", "Attack")):
    df = df.copy()
    df["_d"] = df["Dataset"].map({d: i for i, d in enumerate(DS)}) if "Dataset" in df else 0
    df["_a"] = df["Attack"].map({a: i for i, a in enumerate(ATT)})
    return df.sort_values(["_d", "_a"] + (["Model"] if "Model" in df else []))


# ---------------------------------------------------------------- Table 1: attacks
t = order(pd.read_csv(f"{A}/table1_attacks.csv"))
lines = []
for (d, a), g in t.groupby(["Dataset", "Attack"], sort=False):
    b, r = g[g["Model"] == "BERT"], g[g["Model"] == "RoBERTa"]
    cell = lambda x, c: short(x[c].iloc[0]) if len(x) else "--"
    lines.append(f"{d} & {a} & {cell(b, 'CACC (%)')} & {cell(b, 'ASR (%)')} & {cell(b, 'Clean model on triggered inputs (%)')} & "
                 f"{cell(r, 'CACC (%)')} & {cell(r, 'ASR (%)')} & {cell(r, 'Clean model on triggered inputs (%)')} \\\\")
write("t1_attacks", "\n".join(lines))

# ---------------------------------------------------------------- Table 2: oracle
t = order(pd.read_csv(f"{A}/table2_oracle.csv"))
lines = []
for (d, a), g in t.groupby(["Dataset", "Attack"], sort=False):
    cells = []
    for m in ("BERT", "RoBERTa"):
        x = g[g["Model"] == m]
        if not len(x):
            cells += ["--"] * 3
            continue
        x = x.iloc[0]
        cells += [short(x["Token oracle"]), short(x["INLP k=4"]), short(x["INLP k=8"])]
    limit = g["ASR limit"].iloc[0].split(" ±")[0]
    lines.append(f"{d} & {a} & " + " & ".join(cells) + f" & {limit} \\\\")
write("t2_oracle", "\n".join(lines))

# ---------------------------------------------------------------- Table 3: claim, pooled over the two models
pr = pd.read_csv(f"{A}/table3_claim_per_run.csv")
pr["Dataset"] = pr["dataset"].map({"sst2": "SST-2", "agnews": "AG News", "olid": "OLID", "hsol": "HSOL"})
pr["Attack"] = pr["attack"].map({"badnet": "BadNet", "addsent": "AddSent", "synbkd": "SynBkd", "stylebkd": "StyleBkd"})
f = lambda v, dg=1: f"{np.mean(v):.{dg}f}\\std{{{np.std(v, ddof=1):.{dg}f}}}" if len(v) > 1 else f"{np.mean(v):.{dg}f}"
lines = []
for (d, a), g in order(pr).groupby(["Dataset", "Attack"], sort=False):
    lines.append(f"{d} & {a} & {f(g['usage_ratio_backdoored'])} & {f(g['usage_ratio_clean'])} & "
                 f"{f(g['sensitivity_ratio_backdoored'])} & {f(g['sensitivity_ratio_clean'])} & "
                 f"{f(g['distance_sd_backdoored'])} & {f(g['distance_sd_clean'])} & {np.median(g['ddp_rank_backdoored']):.0f} \\\\")
write("t3_claim", "\n".join(lines))


# ---------------------------------------------------------------- pooled pivot tables (4 to 7)
PM = re.compile(r"(-?\d[\d.]*) ± ([\d.]+)")


def pivot(name, csv, first, rename=None, sd=True):
    t = pd.read_csv(f"{A}/{csv}.csv")
    lines = []
    for _, r in t.iterrows():
        label = (rename or {}).get(r[first], r[first])
        cells = []
        for a in ATT:
            v = r.get(a, "--")
            cells.append(pcell(v, sd))
        extra = [pcell(r[c], sd) for c in t.columns if c not in [first] + ATT]
        lines.append(f"{tex(label)} & " + " & ".join(cells + extra) + " \\\\")
    write(name, "\n".join(lines))


pivot("t4_alignment", "table4_alignment", "Finder", sd=False)
pivot("t5_finder", "table5_finder_erasure", "Method")
pivot("t6_rules", "table6_rules", "Rule", {"Hindsight: some single candidate removes it": "Hindsight: one suffices"}, sd=False)
pivot("t7_baselines", "table7_baselines", "Defense")

# ---------------------------------------------------------------- Table 8: rates
t = pd.read_csv(f"{A}/table8_rates.csv")
t["_a"] = t["Attack"].map({a: i for i, a in enumerate(ATT)})
t["_r"] = t["rate"].str.rstrip("%").astype(float)
lines = []
for _, r in t.sort_values(["_a", "_r"]).iterrows():
    lines.append(f"{r['Attack']} & {tex(r['rate'])} & {short(r['ASR before'])} & {short(r['Oracle best ASR'])} & "
                 f"{short(r['Weight change best ASR'])} & {short(r['CACC'])} \\\\")
write("t8_rates", "\n".join(lines))

# ---------------------------------------------------------------- Appendix: HSOL at 10%
p = f"{A}/tableA_hsol_10pct.csv"
if os.path.exists(p):
    t = pd.read_csv(p)
    lines = [f"{r['Attack']} & {r['Model']} & {short(r['CACC'])} & {short(r['ASR'])} & {short(r['Clean model ASR'])} & "
             f"{short(r['Token oracle'])} & {short(r['INLP k=8'])} \\\\" for _, r in t.iterrows()]
    write("tA_hsol10", "\n".join(lines))
print("tables written to", OUT)
