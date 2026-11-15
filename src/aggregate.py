"""Turns the study's result files into the paper's tables (LaTeX and CSV) and figures (PDF and PNG).

Conventions
- Every cell is a mean over seeds with the standard deviation after the plus-minus sign; n is the number of runs.
- "Removal" means ASR at or below the job's limit (20%, or the clean model's ASR plus 10 points if higher) with a
  clean-accuracy drop of at most 2 points. These are the pass marks fixed before the pilots.
- "Best case" numbers choose the position and the number of directions with the trigger. They are upper bounds on
  what a defender could achieve with that method, not what a defender would get.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ATTACK_ORDER = ["badnet", "addsent", "synbkd", "stylebkd"]
ATTACK_NAME = {"badnet": "BadNet", "addsent": "AddSent", "synbkd": "SynBkd", "stylebkd": "StyleBkd"}
DATASET_NAME = {"sst2": "SST-2", "agnews": "AG News", "hsol": "HSOL", "olid": "OLID"}
MODEL_NAME = {"bert": "BERT", "roberta": "RoBERTa"}
FINDER_NAME = {"ddp": "DDP (sensitivity / usage)", "sensitivity": "Sensitivity only", "weight_change": "Weight change",
               "push": "Single-token push", "pca": "PCA (control)", "random": "Random (control)",
               "oracle_token": "Token oracle", "oracle_inlp": "INLP oracle"}
BASELINE_NAME = {"none": "No defense", "clean_ft": "Clean fine-tuning", "fine_pruning": "Fine-pruning",
                 "fine_mixing": "Fine-mixing", "onion": "ONION", "orthopurify": "OrthoPurify-style"}
COLORS = {"badnet": "#2a78d6", "addsent": "#eb6834", "synbkd": "#1baf7a", "stylebkd": "#eda100"}
MARKERS = {"badnet": "o", "addsent": "s", "synbkd": "^", "stylebkd": "D"}
INK = "#52514e"
CACC_DROP_MAX = 2.0
# The main poison rate per dataset. HSOL has few hate examples (12%), so 10% of the training set would turn most of
# them into poisoned examples; its main rate is set so the poisoned share of the non-target class matches the others.
MAIN_RATE = {"hsol": 0.025}
MID = range(1, 9)          # positions 1 to 8: the middle of the network, where the claim and alignment are summarized

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
                     "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
                     "axes.axisbelow": True})


# =============================================================================================== loading
def load(results_dir):
    R = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f)) as fh:
                    R.append(json.load(fh))
    return R


def jobs(R, attack=None, rate=None, backdoored=True):
    out = []
    for r in R:
        j = r["job"]
        if backdoored and j["attack"] == "clean":
            continue
        if not backdoored and j["attack"] != "clean":
            continue
        if attack and j["attack"] != attack:
            continue
        if rate is not None and abs(j["rate"] - rate) > 1e-9:
            continue
        out.append(r)
    return out


def main(R):
    """Backdoored runs at each dataset's main poison rate. HSOL falls back to 10% only if the 2.5% runs are absent."""
    have = {(r["job"]["dataset"], r["job"]["rate"]) for r in R}
    out = []
    for r in jobs(R):
        d, rate = r["job"]["dataset"], r["job"]["rate"]
        want = MAIN_RATE.get(d, 0.1)
        if (d, want) not in have:
            want = 0.1 if d != "hsol" else None
        if want is not None and abs(rate - want) < 1e-9:
            out.append(r)
    return out


def rows(R, key):
    out = []
    for r in R:
        for row in r.get(key) or []:
            out.append({**r["job"], **{f"base_{k}": v for k, v in r.get("baseline", {}).items()}, **row})
    return pd.DataFrame(out)


def fmt(values, digits=1):
    v = pd.Series(values, dtype=float).dropna()
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return f"{v.iloc[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


# =============================================================================================== output
def latex_escape(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_") \
        .replace("#", "\\#").replace("±", "$\\pm$").replace("–", "--")


def write_table(df, out_dir, name, caption, label, index=True):
    os.makedirs(out_dir, exist_ok=True)
    df = df.astype(object).where(df.notna(), "–")
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=index)
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}",
             "\\small", f"\\begin{{tabular}}{{{'l' * (len(cols) - len(cols) + 1)}{'c' * (len(cols) - 1)}}}",
             "\\toprule", " & ".join(latex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in d.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in r.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return df


def save_figure(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")


# =============================================================================================== per-run summaries
def best_erasure(er, method, k=None):
    """Lowest ASR over positions with clean-accuracy drop of at most 2 points, for one run. Also the position."""
    e = er[er["method"] == method]
    if k is not None:
        e = e[e["k"] == k]
    ok = e[e["cacc_drop"] <= CACC_DROP_MAX]
    if len(ok) == 0:
        return np.nan, np.nan
    best = ok.loc[ok["asr"].idxmin()]
    return float(best["asr"]), int(best["position"])


def run_summaries(R, rate=None):
    """One row per backdoored run with the quantities the tables use (rate None = each dataset's main rate)."""
    out = []
    for r in (main(R) if rate is None else jobs(R, rate=rate)):
        j, b = r["job"], r["baseline"]
        er = pd.DataFrame(r.get("erase") or [])
        row = {**j, "cacc": b["cacc"], "asr": b["asr"], "clean_asr": b["clean_model_asr"], "limit": b["asr_limit"]}
        if len(er):
            if (er["method"] == "oracle_token").any():
                row["token_best_asr"], row["token_best_pos"] = best_erasure(er, "oracle_token")
                ok = er[(er["method"] == "oracle_token") & (er["cacc_drop"] <= CACC_DROP_MAX) & (er["asr"] <= b["asr_limit"])]
                row["token_positions_removing"] = int(ok["position"].nunique())
            for k in sorted(er[er["method"] == "oracle_inlp"]["k"].unique()):
                row[f"inlp{k}_best_asr"], row[f"inlp{k}_best_pos"] = best_erasure(er, "oracle_inlp", k)
            for m in er["method"].unique():
                if not m.startswith("oracle"):
                    row[f"{m}_best_asr"], _ = best_erasure(er, m)
        out.append(row)
    return pd.DataFrame(out)


def oracle_position(s):
    """The position where the oracle erases best (token oracle for insertion triggers, else INLP with 4 directions)."""
    if not np.isnan(s.get("token_best_pos", np.nan)):
        return int(s["token_best_pos"]), "token"
    for k in (4, 8, 2, 1):
        p = s.get(f"inlp{k}_best_pos", np.nan)
        if p == p:
            return int(p), "inlp"
    return None, None


def by_group(df, cols, fn):
    out = []
    for key, g in df.groupby(cols, sort=False):
        out.append({**dict(zip(cols, key if isinstance(key, tuple) else (key,))), **fn(g)})
    return pd.DataFrame(out)


def order_attacks(df, col="attack"):
    df[col] = pd.Categorical(df[col], ATTACK_ORDER, ordered=True)
    return df.sort_values([c for c in ["dataset", col, "model"] if c in df.columns])


def pretty(df):
    d = df.copy()
    if "dataset" in d:
        d["dataset"] = d["dataset"].map(DATASET_NAME).fillna(d["dataset"])
    if "attack" in d:
        d["attack"] = d["attack"].astype(str).map(ATTACK_NAME).fillna(d["attack"])
    if "model" in d:
        d["model"] = d["model"].map(MODEL_NAME).fillna(d["model"])
    return d.rename(columns={"dataset": "Dataset", "attack": "Attack", "model": "Model"})


# =============================================================================================== tables
def table_attacks(R, out):
    S = run_summaries(R)
    clean = pd.DataFrame([{**r["job"], "cacc": r["baseline"]["cacc"]} for r in jobs(R, backdoored=False)])
    t = by_group(order_attacks(S), ["dataset", "attack", "model"], lambda g: {
        "n": len(g), "CACC (%)": fmt(g["cacc"]), "ASR (%)": fmt(g["asr"]),
        "Clean model on triggered inputs (%)": fmt(g["clean_asr"])})
    if len(clean):
        cc = by_group(clean, ["dataset", "model"], lambda g: {"Clean model CACC (%)": fmt(g["cacc"])})
        t = t.merge(cc, on=["dataset", "model"], how="left")
    t = pretty(t).set_index(["Dataset", "Attack", "Model"])
    return write_table(t, out, "table1_attacks", "Backdoored models before any defense (10% poison).",
                       "tab:attacks")


def table_oracle(R, out):
    S = order_attacks(run_summaries(R))
    def f(g):
        row = {"n": len(g), "No defense ASR": fmt(g["asr"])}
        if "token_best_asr" in g and g["token_best_asr"].notna().any():
            row["Token oracle"] = fmt(g["token_best_asr"])
            row["Positions removing (of 13)"] = fmt(g["token_positions_removing"], 1)
        else:
            row["Token oracle"], row["Positions removing (of 13)"] = "–", "–"
        for k in (1, 2, 4, 8):
            c = f"inlp{k}_best_asr"
            row[f"INLP k={k}"] = fmt(g[c]) if c in g else "–"
        row["ASR limit"] = fmt(g["limit"])
        return row
    t = pretty(by_group(S, ["dataset", "attack", "model"], f)).set_index(["Dataset", "Attack", "Model"])
    return write_table(t, out, "table2_oracle",
                       "Lowest ASR (%) after erasing the true trigger subspace at one position, with at most 2 points "
                       "of clean accuracy lost. Token oracle: trigger-token directions (insertion triggers only). "
                       "INLP: the first k directions a linear probe uses to detect the trigger.", "tab:oracle")


def figure_oracle_by_position(R, out):
    er = rows(main(R), "erase")
    if not len(er):
        return
    er = er[((er["method"] == "oracle_token")) | ((er["method"] == "oracle_inlp") & (er["k"] == 4) &
                                                  (~er["attack"].isin(["badnet", "addsent"])))]
    datasets = [d for d in DATASET_NAME if d in set(er["dataset"])]
    models = [m for m in MODEL_NAME if m in set(er["model"])]
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(3.1 * len(datasets), 2.5 * len(models)),
                             sharey=True, squeeze=False)
    for i, m in enumerate(models):
        for jx, d in enumerate(datasets):
            ax = axes[i][jx]
            part = er[(er["model"] == m) & (er["dataset"] == d)]
            for a in ATTACK_ORDER:
                p = part[part["attack"] == a]
                if not len(p):
                    continue
                g = p.groupby("position")["asr"].agg(["mean", "std"]).reset_index()
                ax.plot(g["position"], g["mean"], color=COLORS[a], marker=MARKERS[a], markersize=3.5, linewidth=1.6,
                        label=ATTACK_NAME[a])
                ax.fill_between(g["position"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                                color=COLORS[a], alpha=0.12, linewidth=0)
            ax.axhline(20, color=INK, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.set_ylim(-2, 102)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{DATASET_NAME[d]}, {MODEL_NAME[m]}", loc="left", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("Position (0 = embedding output)")
            if jx == 0:
                ax.set_ylabel("ASR after oracle erasure (%)")
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4)
    save_figure(fig, out, "fig1_oracle_by_position")
    plt.show()


def table_claim(R, out):
    cl = rows(main(R), "claim")
    if not len(cl):
        return None
    cl = cl[cl["position"].isin(MID)]
    cl["kind"] = np.where(cl["attack"].isin(["badnet", "addsent"]), "token", "inlp")
    cl = cl[cl["oracle"] == cl["kind"]]
    per_run = cl.groupby(["dataset", "attack", "model", "seed", "model_role"])[
        ["usage_ratio", "sensitivity_ratio", "distance_sd", "ddp_rank"]].median().unstack("model_role")
    per_run.columns = [f"{a}_{b}" for a, b in per_run.columns]
    per_run = order_attacks(per_run.reset_index())
    per_run.to_csv(os.path.join(out, "table3_claim_per_run.csv"), index=False)
    t = by_group(per_run, ["dataset", "attack", "model"], lambda g: {
        "n": len(g), "Usage ratio": fmt(g["usage_ratio_backdoored"], 1), "Usage, clean model": fmt(g["usage_ratio_clean"], 1),
        "Sensitivity ratio": fmt(g["sensitivity_ratio_backdoored"], 1),
        "Sensitivity, clean model": fmt(g["sensitivity_ratio_clean"], 1),
        "Distance (SD)": fmt(g["distance_sd_backdoored"], 1), "Distance, clean model": fmt(g["distance_sd_clean"], 1),
        "DDP rank": fmt(g["ddp_rank_backdoored"], 0)})
    t = pretty(t).set_index(["Dataset", "Attack", "Model"])
    return write_table(t, out, "table3_claim",
                       "The dormant-direction claim, measured on the main trigger direction (median over positions 1 to 8). "
                       "Ratios are relative to an average direction (1 = average). The claim predicts a usage ratio "
                       "below 1. Clean-model columns use the same trigger in a model without a backdoor. DDP rank: "
                       "where the direction falls in the DDP ranking of 768.", "tab:claim")


def figure_probe(R, out):
    pr = rows(main(R), "inlp")
    if not len(pr):
        return
    pr["acc1"] = pr["probe_acc"].apply(lambda a: a[0] if a else np.nan)
    datasets = [d for d in DATASET_NAME if d in set(pr["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.1 * len(datasets), 2.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], datasets):
        part = pr[(pr["dataset"] == d) & (pr["model"] == "bert")] if (pr["model"] == "bert").any() else pr[pr["dataset"] == d]
        for a in ATTACK_ORDER:
            for role, ls in (("backdoored", "-"), ("clean", ":")):
                p = part[(part["attack"] == a) & (part["model_role"] == role)]
                if not len(p):
                    continue
                g = p.groupby("position")["acc1"].mean()
                ax.plot(g.index, g.values, color=COLORS[a], linestyle=ls, marker=MARKERS[a] if role == "backdoored" else None,
                        markersize=3, linewidth=1.5, label=f"{ATTACK_NAME[a]}{'' if role == 'backdoored' else ', clean model'}")
        ax.set_ylim(0.45, 1.02)
        ax.set_title(DATASET_NAME[d], loc="left", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Position")
    axes[0][0].set_ylabel("Probe accuracy")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    save_figure(fig, out, "fig2_probe_accuracy")
    plt.show()


def table_alignment(R, out):
    al = rows(main(R), "align")
    if not len(al):
        return None
    al = al[al["position"].isin(set(MID) | ({0, 2, 4, 6, 8} if True else set()))]
    al["kind"] = np.where(al["attack"].isin(["badnet", "addsent"]), "token", "inlp")
    al = al[(al["oracle"] == al["kind"]) & (al["position"].isin(MID) | (al["finder"] == "push"))]
    per_run = al.groupby(["dataset", "model", "attack", "seed", "finder"]).agg(
        bd=("cos_best", "median"), ctrl=("cos_best_clean_control", "median"), rnd=("cos_random", "mean")).reset_index()
    per_run.to_csv(os.path.join(out, "table4_alignment_per_run.csv"), index=False)
    recs = []
    for (f, a), g in per_run.groupby(["finder", "attack"]):
        recs.append({"Finder": FINDER_NAME.get(f, f), "attack": a,
                     "cell": f"{fmt(g['bd'], 2)} / {fmt(g['ctrl'], 2)} / {np.nanmean(g['rnd']):.2f}"})
    D = pd.DataFrame(recs)
    t = D.pivot(index="Finder", columns="attack", values="cell")
    t = t.reindex([FINDER_NAME[f] for f in ("ddp", "sensitivity", "push", "weight_change") if FINDER_NAME[f] in t.index])
    t = t[[a for a in ATTACK_ORDER if a in t.columns]].rename(columns=ATTACK_NAME)
    return write_table(t, out, "table4_alignment",
                       "Cosine between the best of a finder's 10 candidates and the trigger subspace (median over "
                       "positions 1 to 8): backdoored model / clean-model control / random directions. Token oracle "
                       "for BadNet and AddSent, INLP oracle (4 directions) for SynBkd and StyleBkd. Pooled over "
                       "datasets, models and seeds.", "tab:alignment")


def table_finder_erasure(R, out):
    S = run_summaries(R)
    methods = ["token", "inlp4", "ddp", "sensitivity", "weight_change", "push", "pca", "random"]
    recs = []
    for a in ATTACK_ORDER:
        g = S[S["attack"] == a]
        if not len(g):
            continue
        for m in methods:
            c = f"{m}_best_asr"
            if c not in g or g[c].isna().all():
                continue
            removed = (g[c] <= g["limit"]).mean() * 100
            name = {"token": "Token oracle", "inlp4": "INLP oracle, k=4"}.get(m, FINDER_NAME.get(m, m))
            recs.append({"Method": name, "attack": a, "cell": f"{fmt(g[c])} ({removed:.0f}%)"})
    D = pd.DataFrame(recs)
    t = D.pivot(index="Method", columns="attack", values="cell")
    t = t.reindex([r for r in ["Token oracle", "INLP oracle, k=4"] + [FINDER_NAME[m] for m in methods[2:]] if r in t.index])
    t = t[[a for a in ATTACK_ORDER if a in t.columns]].rename(columns=ATTACK_NAME)
    return write_table(t, out, "table5_finder_erasure",
                       "Lowest ASR (%) reachable by erasing up to 10 of a method's directions at one position with at most "
                       "2 points of clean accuracy lost (best case: position and number chosen with the trigger). In "
                       "brackets: share of runs meeting the removal criterion. Pooled over datasets, models and seeds.",
                       "tab:finder_erasure")


def table_rules(R, out):
    ru = rows(main(R), "rules")
    cr = rows(jobs(R, backdoored=False), "rules")
    cd = rows(main(R), "cand")
    recs = []
    for a in ATTACK_ORDER:
        g = ru[ru["attack"] == a] if len(ru) else ru
        if not len(g):
            continue
        for rule in ("harmless_only", "both_tests"):
            p = g[g["rule"] == rule]
            removed = ((p["asr"] <= p["base_asr_limit"]) & (p["cacc_drop"] <= CACC_DROP_MAX)).mean() * 100
            recs.append({"Rule": {"harmless_only": "Harmless to erase", "both_tests": "Harmless and switch"}[rule],
                         "attack": a, "cell": f"{fmt(p['base_asr'])} → {fmt(p['asr'])} ({removed:.0f}%)"})
        if len(cd) and "removes_backdoor" in cd:
            c = cd[cd["attack"] == a]
            hs = c.groupby(["dataset", "model", "seed"])["removes_backdoor"].any().mean() * 100
            recs.append({"Rule": "Hindsight: some single candidate removes it", "attack": a, "cell": f"{hs:.0f}% of runs"})
    D = pd.DataFrame(recs)
    t = D.pivot(index="Rule", columns="attack", values="cell")
    t = t[[a for a in ATTACK_ORDER if a in t.columns]].rename(columns=ATTACK_NAME)
    if len(cr):
        fa = by_group(cr, ["rule"], lambda g: {"Clean models: directions erased": fmt(g["directions"], 1),
                                               "Clean models: CACC drop": fmt(g["cacc_drop"], 2)})
        fa["Rule"] = fa["rule"].map({"harmless_only": "Harmless to erase", "both_tests": "Harmless and switch"})
        t = t.join(fa.set_index("Rule")[["Clean models: directions erased", "Clean models: CACC drop"]], how="left")
    return write_table(t, out, "table6_rules",
                       "Trigger-free selection among weight-change candidates: ASR before → after (share of runs meeting "
                       "the removal criterion), and the rules' false alarms on clean models.", "tab:rules")


def table_baselines(R, out):
    bl = rows(main(R), "baselines")
    if not len(bl):
        return None
    bl = bl[bl.get("error").isna()] if "error" in bl else bl
    bl["removed"] = (bl["asr"] <= bl["base_asr_limit"]) & (bl.get("cacc_drop", 0).fillna(0) <= CACC_DROP_MAX)
    bl.to_csv(os.path.join(out, "table7_baselines_per_run.csv"), index=False)
    recs = []
    for (m, a), g in bl.groupby(["method", "attack"]):
        drop = g["cacc_drop"] if "cacc_drop" in g else pd.Series(dtype=float)
        recs.append({"Defense": BASELINE_NAME.get(m, m), "attack": a,
                     "cell": f"{fmt(g['asr'])} / {fmt(drop if m != 'none' else pd.Series([0.0]), 1)}"})
    D = pd.DataFrame(recs)
    t = D.pivot(index="Defense", columns="attack", values="cell")
    t = t.reindex([BASELINE_NAME[m] for m in BASELINE_NAME if BASELINE_NAME[m] in t.index])
    t = t[[a for a in ATTACK_ORDER if a in t.columns]].rename(columns=ATTACK_NAME)
    return write_table(t, out, "table7_baselines",
                       "Published defenses: ASR (%) / clean-accuracy drop (points), pooled over datasets, models and "
                       "seeds. ONION is measured on a fixed subset of 500 sentences.", "tab:baselines")


def table_rates(R, out):
    rates = sorted({r["job"]["rate"] for r in jobs(R) if r["job"]["dataset"] == "sst2" and r["job"]["model"] == "bert"})
    recs = []
    for rate in rates:
        S = run_summaries([r for r in R if r["job"]["dataset"] == "sst2" and r["job"]["model"] == "bert"], rate=rate)
        for a in ATTACK_ORDER:
            g = S[S["attack"] == a]
            if not len(g):
                continue
            oc = "token_best_asr" if "token_best_asr" in g and g["token_best_asr"].notna().any() else "inlp4_best_asr"
            recs.append({"attack": a, "rate": f"{rate * 100:g}%", "n": len(g), "ASR before": fmt(g["asr"]),
                         "Oracle best ASR": fmt(g.get(oc)), "Weight change best ASR": fmt(g.get("weight_change_best_asr")),
                         "CACC": fmt(g["cacc"])})
    if not recs:
        return None
    D = order_attacks(pd.DataFrame(recs))
    t = pretty(D).set_index(["Attack", "rate"])
    return write_table(t, out, "table8_rates", "Effect of the poison rate (SST-2, BERT).", "tab:rates")


def table_hsol_10(R, out):
    """HSOL at 10% poison: kept for the appendix. Poisoning 10% of the training set turns most hate examples into
    poisoned examples, so the model largely stops detecting hate on clean inputs."""
    runs = [r for r in jobs(R, rate=0.1) if r["job"]["dataset"] == "hsol"]
    if not runs:
        return None
    S = run_summaries(runs, rate=0.1)
    recs = []
    for _, s in S.iterrows():
        r = next(x for x in runs if x["job"]["attack"] == s["attack"] and x["job"]["model"] == s["model"]
                 and x["job"]["seed"] == s["seed"])
        b = r["baseline"]
        nt = sum(1 for _ in range(1))
        recs.append({"attack": s["attack"], "model": s["model"], "cacc": s["cacc"], "asr": s["asr"],
                     "clean_asr": s["clean_asr"], "poisoned": b["n_poison"], "token": s.get("token_best_asr", np.nan),
                     "inlp8": s.get("inlp8_best_asr", np.nan)})
    D = order_attacks(pd.DataFrame(recs))
    t = by_group(D, ["attack", "model"], lambda g: {"n": len(g), "Poisoned examples": fmt(g["poisoned"], 0),
                                                    "CACC": fmt(g["cacc"]), "ASR": fmt(g["asr"]),
                                                    "Clean model ASR": fmt(g["clean_asr"]),
                                                    "Token oracle": fmt(g["token"]), "INLP k=8": fmt(g["inlp8"])})
    t = pretty(t).set_index(["Attack", "Model"])
    return write_table(t, out, "tableA_hsol_10pct",
                       "HSOL at 10% poison (appendix). The poisoned examples are drawn from the 866 hate examples, so "
                       "most of the class is relabeled and the models largely stop detecting hate on clean inputs.",
                       "tab:hsol10")


def key_numbers(R):
    """Numbers quoted in the paper's text."""
    S = run_summaries(R)
    out = {"runs_main": len(S), "runs_total": len(R)}
    ins = S[S["attack"].isin(["badnet", "addsent"])]
    out["token_removes_share"] = float((ins["token_best_asr"] <= ins["limit"]).mean())
    for d in DATASET_NAME:
        g = ins[ins["dataset"] == d]
        if len(g):
            out[f"token_removes_{d}"] = float((g["token_best_asr"] <= g["limit"]).mean())
            out[f"token_best_asr_{d}"] = float(g["token_best_asr"].mean())
            out[f"token_positions_{d}"] = float(g["token_positions_removing"].mean())
    para = S[S["attack"].isin(["synbkd", "stylebkd"])]
    for k in (4, 8):
        c = f"inlp{k}_best_asr"
        out[f"inlp{k}_removes_paraphrase"] = float((para[c] <= para["limit"]).mean())
        out[f"inlp{k}_best_pos_paraphrase_median"] = float(para[f"inlp{k}_best_pos"].median())
    out["token_best_pos_insertion_median"] = float(ins["token_best_pos"].median())
    cl = rows(main(R), "claim")
    cl = cl[cl["position"].isin(MID)]
    cl["kind"] = np.where(cl["attack"].isin(["badnet", "addsent"]), "token", "inlp")
    cl = cl[cl["oracle"] == cl["kind"]]
    pr = cl.groupby(["dataset", "attack", "model", "seed", "model_role"])[["usage_ratio", "sensitivity_ratio",
                                                                             "distance_sd", "ddp_rank"]].median().unstack("model_role")
    out["claim_runs"] = len(pr)
    out["usage_above_1_share"] = float((pr[("usage_ratio", "backdoored")] > 1).mean())
    out["usage_min"] = float(pr[("usage_ratio", "backdoored")].min())
    out["sens_bd_gt_clean_share"] = float((pr[("sensitivity_ratio", "backdoored")] > pr[("sensitivity_ratio", "clean")]).mean())
    out["sens_ratio_bd_over_clean_median"] = float((pr[("sensitivity_ratio", "backdoored")] / pr[("sensitivity_ratio", "clean")]).median())
    out["ddp_rank_median"] = float(pr[("ddp_rank", "backdoored")].median())
    out["ddp_rank_min"] = float(pr[("ddp_rank", "backdoored")].min())
    return out
