"""
Experiments 1-3 

  Exp 1  Fusion strategies (across the 5 judges, per criterion):  F1 mean | F2 median | F3 majority vote | F4 triangular fuzzy (min, median, max)
  Exp 2  Aggregation operators (across the 9 criteria):           A1 mean | A2 crisp OWA (Yager) | A3 T1OWA Yager | A4 T1OWA equal weights
         + Kendall tau, paired bootstrap differences, sensitivity to the Yager exponent
  Exp 3  Leave-one-judge-out robustness

INPUT : judge_scores_nine_criteria.xlsx  (written by Step 2; columns A1_<judge> ... A9_<judge>)
        2_User_profiles_extended 1.csv (user_labels_gt1.xlsx)            (external label GT1 = risk_level ) 
        type1owa_zhou.py                          
OUTPUT: experiments_1_3.xlsx 

"""
import os, re, sys, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

# paths (relative to the repository root) and settings
USER_SCORES     = "data/judge_scores_nine_criteria.xlsx"
PROFILES_FILE   = "data/user_labels_gt1.xlsx"
T1OWA_DIR       = "code"
OUTPUT_FILE     = "results/experiments_1_3.xlsx"

# F3 vote threshold, minimum valid judges per criterion, Yager exponent, q grid, bootstrap, T1OWA grid
PRESENT_THR  = 0.50
MIN_JUDGES   = 2
YAGER_Q      = 0.5
Q_GRID       = [0.25, 0.5, 0.75, 1.0]
N_BOOT, SEED = 2000, 42
GRID_STEP    = 101
N_ALPHA      = 50

# T1OWA implementation (alpha-cuts, centroid defuzzification)
sys.path.insert(0, T1OWA_DIR)
from type1owa_zhou import aggregate_triangular_t1owa
domainX = np.round(np.linspace(0.0, 1.0, GRID_STEP), 2)
alphas  = np.linspace(0.0, 1.0, N_ALPHA)
GT1 = {"none": 0, "low": 1, "moderate": 2, "high": 3}

# load judge scores (users x criteria x judges) and GT1 labels
raw = pd.read_excel(USER_SCORES, index_col=0)
raw.index = raw.index.astype(str).str.strip()
pat = re.compile(r"^A([1-9])_(.+)$")
judges = []
for c in raw.columns:
    m = pat.match(c)
    if m and m.group(2) not in ("n_llm",) and m.group(2) not in judges:
        judges.append(m.group(2))
print(f"judges: {judges}")
X = np.full((len(raw), 9, len(judges)), np.nan)
for k in range(9):
    for j, jd in enumerate(judges):
        col = f"A{k+1}_{jd}"
        if col in raw.columns:
            X[:, k, j] = pd.to_numeric(raw[col], errors="coerce").values

prof = pd.read_excel(PROFILES_FILE)
idc = [c for c in prof.columns if c.lower() in ("user_id", "main_user", "user", "id")][0]
prof[idc] = prof[idc].astype(str).str.strip()
ymap = prof.set_index(idc)["risk_level"].astype(str).str.strip().str.lower().map(GT1)
y = ymap.reindex(raw.index)
keep = y.notna().values
X, users, y = X[keep], raw.index[keep], y[keep].astype(int).values
print(f"users with label: {len(users)} | label counts: {np.bincount(y).tolist()} | scores: {int(np.isfinite(X).sum())}")

# Stage 1: fuse the judge scores per criterion (F1-F4); criteria with fewer than MIN_JUDGES valid scores are not assessed
def fuse(Xs, strategy):
    n_valid = np.isfinite(Xs).sum(axis=2)
    assessed = n_valid >= MIN_JUDGES
    with np.errstate(all="ignore"):
        mean = np.nanmean(Xs, axis=2); med = np.nanmedian(Xs, axis=2)
        mn = np.nanmin(Xs, axis=2); mx = np.nanmax(Xs, axis=2)
        share = np.nansum(Xs >= PRESENT_THR, axis=2) / np.where(n_valid > 0, n_valid, 1)

    mean, med, mn, mx = (np.round(v, 4) for v in (mean, med, mn, mx))
    if strategy == "F1_mean":   out = (mean,)
    elif strategy == "F2_median": out = (np.round(med, 2),)
    elif strategy == "F3_majority": out = ((share > 0.5).astype(float),)
    elif strategy == "F4_fuzzy": out = (mn, med, mx)
    else: raise ValueError(strategy)
    return [np.where(assessed, o, 0.0) for o in out], assessed

# Yager OWA weights from the quantifier Q(r) = r**q
def yager_w(n, q):
    Q = lambda r: r ** q
    w = np.array([Q(i / n) - Q((i - 1) / n) for i in range(1, n + 1)])
    return w / w.sum()

# Stage 2 with crisp values: mean or OWA over the assessed criteria
def agg_crisp(vals, assessed, op, q=YAGER_Q):
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        v = vals[i][assessed[i]]
        if len(v) == 0: continue
        out[i] = v.mean() if op == "mean" else np.dot(yager_w(len(v), q), np.sort(v)[::-1])
    return out

# Stage 2 with fuzzy values: T1OWA (equal or Yager weights), returns the centroid
def agg_fuzzy(abc, assessed, op, q=YAGER_Q):
    a, b, c = abc
    out = np.zeros(len(a))
    for i in range(len(a)):
        m = assessed[i]
        if not m.any(): continue
        P = []
        for k in np.where(m)[0]:
            bb = round(float(b[i, k]), 2); aa = min(round(float(a[i, k]), 2), bb); cc = max(round(float(c[i, k]), 2), bb)
            if bb == 0.0 and cc - aa < 1e-3: cc = 1e-3
            P.append((aa, bb, cc, 1.0))
        w = None if op == "equal" else yager_w(len(P), q)
        try:
            _, _, cen = aggregate_triangular_t1owa(P, domain_x=domainX, alpha_levels=alphas, weights=w)
        except Exception:
            cen = np.nan
        out[i] = np.mean([p[1] for p in P]) if not np.isfinite(cen) else cen
    return out

# fusion + aggregation for one pipeline
def run(Xs, fusion, op, q=YAGER_Q):
    f, assessed = fuse(Xs, fusion)
    if fusion == "F4_fuzzy":
        return agg_fuzzy(tuple(f), assessed, "equal" if op == "mean" else "yager", q)
    return agg_crisp(f[0], assessed, op, q)

# evaluation metrics and paired bootstrap (same resamples for every method)
def auc(s, pos):
    pos = np.asarray(pos, bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auc_nh(s, yy):
    m = (yy == 0) | (yy == 3); return auc(s[m], yy[m] == 3)

def auc_mh(s, yy): return auc(s, yy >= 2)

def sp(s, yy):
    r = spearmanr(s, yy)[0]; return 0.0 if np.isnan(r) else r

def kt(s, yy):
    r = kendalltau(s, yy)[0]; return 0.0 if np.isnan(r) else r

METRICS = {"spearman": sp, "kendall": kt, "auc_none_vs_high": auc_nh, "auc_nonelow_vs_modhigh": auc_mh}
rng = np.random.default_rng(SEED)
BOOT = rng.integers(0, len(y), size=(N_BOOT, len(y)))

def fmt(v, bs):
    bs = bs[~np.isnan(bs)]; return f"{v:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"

def table_row(name, s):
    row = {"method": name}
    for mn, fn in METRICS.items():
        row[mn] = fmt(fn(s, y), np.array([fn(s[b], y[b]) for b in BOOT]))
    return row

# cross-validated cut-points (Youden) for macro-F1
def youden(score, pos):
    fpr, tpr, thr = roc_curve(pos, score); k = np.argmax((tpr - fpr)[1:]) + 1; return thr[k]

def cv_macro_f1(s):
    P = np.zeros(len(s), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(s, y):
        cuts = np.sort([youden(s[tr], y[tr] >= k) for k in (1, 2, 3)])
        P[te] = (s[te][:, None] >= cuts[None, :]).sum(1)
    return f1_score(y, P, labels=[0, 1, 2, 3], average="macro", zero_division=0)

# paired bootstrap difference between two methods (* = interval excludes 0)
def paired_diff(s1, s2, name):
    row = {"comparison": name}
    for mn in ("spearman", "kendall", "auc_none_vs_high", "auc_nonelow_vs_modhigh"):
        fn = METRICS[mn]; d0 = fn(s1, y) - fn(s2, y)
        d = np.array([fn(s1[b], y[b]) - fn(s2[b], y[b]) for b in BOOT]); d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        row[mn] = f"{d0:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if lo > 0 or hi < 0 else "")
    return row

# ---- Experiment 1 ----
print("\nExperiment 1 - fusion strategies")
FUS = ["F1_mean", "F2_median", "F3_majority", "F4_fuzzy"]
S = {}
exp1 = []
for f in FUS:
    for op in ("mean", "owa"):
        label_op = "A1 mean" if op == "mean" else "A2/A3 OWA (Yager)"
        if f == "F4_fuzzy": label_op = "A4 T1OWA equal" if op == "mean" else "A3 T1OWA Yager"
        S[(f, op)] = run(X, f, op)
        r = table_row(f"{f} + {label_op}", S[(f, op)]); r["macro_F1_cv"] = round(cv_macro_f1(S[(f, op)]), 3)
        exp1.append(r); print(" ", r["method"], "|", r["spearman"])
exp1 = pd.DataFrame(exp1)

pd1 = []
for op in ("mean", "owa"):
    for f1_, f2_ in itertools.combinations(FUS, 2):
        pd1.append(paired_diff(S[(f1_, op)], S[(f2_, op)], f"{f1_} - {f2_} ({op})"))
pd1 = pd.DataFrame(pd1)

# fuzzy width and judge disagreement per user
f4, assessed = fuse(X, "F4_fuzzy")
width = np.where(assessed, f4[2] - f4[0], np.nan)
sd = np.nanstd(X, axis=2)
w_user = np.nanmean(width, axis=1)
wid = [{"quantity": "mean fuzzy width (c-a) per user", **{LV: round(float(np.nanmean(w_user[y == i])), 4) for i, LV in enumerate(GT1)},
        "spearman_with_GT1": round(sp(w_user, y), 3)},
       {"quantity": "mean judge SD per user", **{LV: round(float(np.nanmean(np.nanmean(sd, 1)[y == i])), 4) for i, LV in enumerate(GT1)},
        "spearman_with_GT1": round(sp(np.nanmean(sd, 1), y), 3)}]
wid = pd.DataFrame(wid)

# pairwise judge agreement and single-judge references
J = len(judges); agree = np.full((J, J), np.nan)
for i in range(J):
    for j in range(J):
        m = np.isfinite(X[:, :, i]) & np.isfinite(X[:, :, j])
        agree[i, j] = spearmanr(X[:, :, i][m], X[:, :, j][m])[0]
agree = pd.DataFrame(agree, index=judges, columns=judges).round(3)

single = []
for j, jd in enumerate(judges):
    s = run(X[:, :, [j]], "F1_mean", "mean") 
    single.append(table_row(f"single judge: {jd} (mean)", s))
single = pd.DataFrame(single)

# ---- Experiment 2 ----
print("\nExperiment 2 - aggregation operators (on the same fused criteria)")
A1 = S[("F2_median", "mean")];  A2 = S[("F2_median", "owa")]
A3 = S[("F4_fuzzy", "owa")];    A4 = S[("F4_fuzzy", "mean")]
ops = {"A1 mean (medians)": A1, "A2 crisp OWA (medians)": A2, "A3 T1OWA Yager": A3, "A4 T1OWA equal": A4}
exp2 = []
for nm, s in ops.items():
    r = table_row(nm, s); r["macro_F1_cv"] = round(cv_macro_f1(s), 3); exp2.append(r); print(" ", nm, "|", r["spearman"])
exp2 = pd.DataFrame(exp2)
pd2 = pd.DataFrame([paired_diff(ops[a], ops[b], f"{a}  -  {b}") for a, b in
                    [("A3 T1OWA Yager", "A1 mean (medians)"), ("A3 T1OWA Yager", "A2 crisp OWA (medians)"),
                     ("A3 T1OWA Yager", "A4 T1OWA equal"), ("A2 crisp OWA (medians)", "A1 mean (medians)")]])

# check: equal-weight T1OWA vs fuzzy arithmetic mean
a_, b_, c_ = f4
fz_mean = np.array([np.mean(((a_ + b_ + c_) / 3)[i][assessed[i]]) if assessed[i].any() else 0 for i in range(len(y))])
check = pd.DataFrame([{"max_abs_diff(T1OWA equal, fuzzy mean)": float(np.max(np.abs(A4 - fz_mean))),
                       "mean_abs_diff": float(np.mean(np.abs(A4 - fz_mean))),
                       "spearman(T1OWA equal, fuzzy mean)": float(spearmanr(A4, fz_mean)[0])}])

# sensitivity to the Yager exponent q (q = 1 is equal weights)
sens = []
for q in Q_GRID:
    s_c = run(X, "F2_median", "owa", q); s_f = run(X, "F4_fuzzy", "owa", q)
    sens.append({"q": q, "w1": round(yager_w(9, q)[0], 3), "crisp OWA spearman": round(sp(s_c, y), 4), "crisp OWA AUC none-vs-high": round(auc_nh(s_c, y), 4),
                 "T1OWA spearman": round(sp(s_f, y), 4), "T1OWA AUC none-vs-high": round(auc_nh(s_f, y), 4)})
    print(f"  q={q}: crisp {sens[-1]['crisp OWA spearman']} | T1OWA {sens[-1]['T1OWA spearman']}")
sens = pd.DataFrame(sens)

# ---- Experiment 3 ----
print("\nExperiment 3 - leave-one-judge-out")
full = {"F1_mean": S[("F1_mean", "mean")], "F2_median": S[("F2_median", "mean")], "F3_majority": S[("F3_majority", "mean")], "F4_fuzzy (T1OWA Yager)": A3}
loo = []
for j, jd in enumerate(judges + ["(none: all judges)"]):
    idx = [k for k in range(len(judges)) if k != j] if j < len(judges) else list(range(len(judges)))
    Xs = X[:, :, idx]
    for nm, f, op, ref in (("F1_mean", "F1_mean", "mean", full["F1_mean"]),
                           ("F2_median", "F2_median", "mean", full["F2_median"]),
                           ("F3_majority", "F3_majority", "mean", full["F3_majority"]),
                           ("F4_fuzzy (T1OWA Yager)", "F4_fuzzy", "owa", full["F4_fuzzy (T1OWA Yager)"])):
        s = run(Xs, f, op) if j < len(judges) else ref
        loo.append({"removed judge": jd, "pipeline": nm, "n_judges": len(idx),
                    "spearman": round(sp(s, y), 4), "kendall": round(kt(s, y), 4),
                    "auc_none_vs_high": round(auc_nh(s, y), 4), "auc_nonelow_vs_modhigh": round(auc_mh(s, y), 4),
                    "rank_agreement_with_full (Spearman)": round(float(spearmanr(s, ref)[0]), 4)})
    print(" ", jd)
loo = pd.DataFrame(loo)
loo_range = loo[loo["removed judge"] != "(none: all judges)"].groupby("pipeline").agg(
    spearman_min=("spearman", "min"), spearman_max=("spearman", "max"),
    auc_nh_min=("auc_none_vs_high", "min"), auc_nh_max=("auc_none_vs_high", "max"),
    rank_agreement_min=("rank_agreement_with_full (Spearman)", "min")).round(4).reset_index()

# save all tables
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUTPUT_FILE) as w:
    exp1.to_excel(w, sheet_name="exp1_fusion", index=False)
    pd1.to_excel(w, sheet_name="exp1_paired_diff", index=False)
    wid.to_excel(w, sheet_name="exp1_width", index=False)
    agree.to_excel(w, sheet_name="exp1_judge_agreement")
    single.to_excel(w, sheet_name="exp1_single_judges", index=False)
    exp2.to_excel(w, sheet_name="exp2_operators", index=False)
    pd2.to_excel(w, sheet_name="exp2_paired_diff", index=False)
    check.to_excel(w, sheet_name="exp2_T1OWA_equal_check", index=False)
    sens.to_excel(w, sheet_name="exp2_yager_sensitivity", index=False)
    loo.to_excel(w, sheet_name="exp3_loo", index=False)
    loo_range.to_excel(w, sheet_name="exp3_loo_range", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for nm, t in (("EXP 1 fusion", exp1), ("EXP 1 paired", pd1), ("EXP 1 width", wid), ("EXP 2 operators", exp2),
              ("EXP 2 paired", pd2), ("EXP 2 check", check), ("EXP 2 sensitivity", sens), ("EXP 3 range", loo_range)):
    print(f"\n--- {nm} ---\n{t.to_string(index=False)}")
print(f"\nSaved -> {OUTPUT_FILE}")
