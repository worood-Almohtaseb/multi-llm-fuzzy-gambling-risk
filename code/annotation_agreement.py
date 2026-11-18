# Input : annotation/annotator_1.xlsx, annotation/annotator_2.xlsx   
#         annotation/sample_private.xlsx                              
#         annotation/consensus.xlsx                                   
#         Data/judge_scores_nine_criteria.xlsx, Data/user_labels_gt1.xlsx, Data/user_texts_concatenated.csv,
#         result/baseline_llm_zeroshot_all.xlsx, code/type1owa_zhou.py
# Output: result/annotation_agreement.xlsx

import os, re, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")

ANNOT_FILES = ["annotation/annotator_1.xlsx", "annotation/annotator_2.xlsx"]
SAMPLE_FILE = "annotation/sample_private.xlsx"
CONSENSUS_FILE = "annotation/consensus.xlsx"
USER_SCORES = "data/judge_scores_nine_criteria.xlsx"
PROFILES_FILE = "data/user_labels_gt1.xlsx"
TEXTS_FILE = "data/user_texts_concatenated.csv"
ZEROSHOT_FILE = "results/baseline_llm_zeroshot_all.xlsx"
OUTPUT_FILE = "result/annotation_agreement.xlsx"
T1OWA_DIR = os.path.dirname(os.path.abspath(__file__))

N_BOOT = int(os.environ.get("N_BOOT", 2000))
SEED = 42
PRESENT_HUMAN = 2          # human score >= 2 means "criterion present"
DSM_MIN = 4                # DSM-5: at least 4 criteria
MIN_JUDGES, YAGER_Q, PRESENT_THR = 2, 0.5, 0.50
BIN_CUTS = [0.05, 0.40, 0.60]   # judge score -> 4 levels: 0 | 0.10-0.30 | 0.50 | >=0.70
LLM_THRS = [0.5, 0.7]
GRID_STEP, N_ALPHA = 101, 50
GT1 = {"none": 0, "low": 1, "moderate": 2, "high": 3}
LEVELS = list(GT1)
CODES = [f"A{k}" for k in range(1, 10)]

sys.path.insert(0, T1OWA_DIR)
from type1owa_zhou import aggregate_triangular_t1owa
domainX = np.round(np.linspace(0.0, 1.0, GRID_STEP), 2)
alphas = np.linspace(0.0, 1.0, N_ALPHA)


#  weighted statistics
def _uid(v):
    return pd.Series(v).astype(str).str.strip().values


def wrank(x, w):
    """weighted mid-rank scaled to (0,1)"""
    u, inv = np.unique(x, return_inverse=True)
    tw = np.bincount(inv, weights=w, minlength=len(u))
    W = tw.sum()
    if W <= 0:
        return np.full(len(x), np.nan)
    return ((np.cumsum(tw) - tw + tw / 2) / W)[inv]


def wspearman(x, y, w):
    if w.sum() <= 0:
        return np.nan
    rx, ry = wrank(x, w), wrank(y, w)
    W = w.sum()
    mx, my = (w * rx).sum() / W, (w * ry).sum() / W
    cxy = (w * (rx - mx) * (ry - my)).sum()
    vx, vy = (w * (rx - mx) ** 2).sum(), (w * (ry - my) ** 2).sum()
    return np.nan if vx <= 0 or vy <= 0 else cxy / np.sqrt(vx * vy)


def wauc(s, pos, w):
    pos = np.asarray(pos, bool)
    wp, wn = (w * pos).sum(), (w * ~pos).sum()
    if wp <= 0 or wn <= 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    P = np.bincount(inv, weights=w * pos, minlength=len(u))
    Nn = np.bincount(inv, weights=w * ~pos, minlength=len(u))
    return float((P * (np.cumsum(Nn) - Nn + 0.5 * Nn)).sum() / (wp * wn))


def wkappa(a, b, w, K=4, quadratic=True):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((K, K))
    np.add.at(O, (a, b), w)
    W = O.sum()
    if W <= 0:
        return np.nan
    E = np.outer(O.sum(1), O.sum(0)) / W
    i, j = np.indices((K, K))
    wm = ((i - j) ** 2 / (K - 1) ** 2) if quadratic else (i != j).astype(float)
    den = (wm * E).sum()
    return np.nan if den <= 0 else 1 - (wm * O).sum() / den


def bin_llm(s):
    return np.digitize(s, BIN_CUTS)


def rhu(x):                       # round half up (consensus mean -> integer level)
    return np.floor(np.asarray(x, float) + 0.5).astype(int)


def bin_metrics(score, h, w, thr):
    pos, pred = h >= PRESENT_HUMAN, score >= thr
    tp, fp = (w * (pred & pos)).sum(), (w * (pred & ~pos)).sum()
    fn, tn = (w * (~pred & pos)).sum(), (w * (~pred & ~pos)).sum()
    T = tp + fp + fn + tn
    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
    po = (tp + tn) / T
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / T ** 2
    kap = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(sensitivity=sens, specificity=spec, precision=ppv, accuracy=po, cohen_kappa=kap)


# framework definitions (= experiments_1_3.py)
def fuse(Xs, strategy, thr=PRESENT_THR):
    n_valid = np.isfinite(Xs).sum(axis=2)
    assessed = n_valid >= MIN_JUDGES
    with np.errstate(all="ignore"):
        mean = np.nanmean(Xs, axis=2); med = np.nanmedian(Xs, axis=2)
        mn = np.nanmin(Xs, axis=2); mx = np.nanmax(Xs, axis=2)
        share = np.nansum(Xs >= thr, axis=2) / np.where(n_valid > 0, n_valid, 1)
    mean, med, mn, mx = (np.round(v, 4) for v in (mean, med, mn, mx))
    if strategy == "F1_mean": out = (mean,)
    elif strategy == "F2_median": out = (np.round(med, 2),)
    elif strategy == "F3_share": out = (share,)
    elif strategy == "F4_fuzzy": out = (mn, med, mx)
    else: raise ValueError(strategy)
    return [np.where(assessed, o, 0.0) for o in out], assessed


def yager_w(n, q):
    w = np.array([(i / n) ** q - ((i - 1) / n) ** q for i in range(1, n + 1)])
    return w / w.sum()


def agg_crisp(vals, assessed, op, q=YAGER_Q):
    out = np.zeros(len(vals))
    for i in range(len(vals)):
        v = vals[i][assessed[i]]
        if len(v) == 0: continue
        out[i] = v.mean() if op == "mean" else np.dot(yager_w(len(v), q), np.sort(v)[::-1])
    return out


def triangles(abc, assessed, i):
    a, b, c = abc
    P = []
    for k in np.where(assessed[i])[0]:
        bb = round(float(b[i, k]), 2); aa = min(round(float(a[i, k]), 2), bb); cc = max(round(float(c[i, k]), 2), bb)
        if bb == 0.0 and cc - aa < 1e-3: cc = 1e-3
        P.append((aa, bb, cc, 1.0))
    return P


def t1owa(P, q=YAGER_Q, equal=False):
    w = None if equal else yager_w(len(P), q)
    try:
        agg, _, cen = aggregate_triangular_t1owa(P, domain_x=domainX, alpha_levels=alphas, weights=w)
    except Exception:
        return np.mean([p[1] for p in P])
    return cen if np.isfinite(cen) else np.mean([p[1] for p in P])


def agg_fuzzy(abc, assessed, equal=False, q=YAGER_Q):
    out = np.zeros(assessed.shape[0])
    for i in range(len(out)):
        P = triangles(abc, assessed, i)
        if P: out[i] = t1owa(P, q, equal)
    return out


def run(Xs, fusion, op, q=YAGER_Q):
    f, assessed = fuse(Xs, fusion)
    if fusion == "F4_fuzzy":
        return agg_fuzzy(tuple(f), assessed, equal=(op == "mean"), q=q)
    return agg_crisp(f[0], assessed, op, q)


# load data
print("loading ...")
raw = pd.read_excel(USER_SCORES, index_col=0)
raw.index = _uid(raw.index)
pat = re.compile(r"^A([1-9])_(.+)$")
judges = []
for c in raw.columns:
    m = pat.match(c)
    if m and m.group(2) != "n_llm" and m.group(2) not in judges:
        judges.append(m.group(2))
J = len(judges)
Xall = np.full((len(raw), 9, J), np.nan)
for k in range(9):
    for j, jd in enumerate(judges):
        col = f"A{k+1}_{jd}"
        if col in raw.columns:
            Xall[:, k, j] = pd.to_numeric(raw[col], errors="coerce").values
pos_of = {u: i for i, u in enumerate(raw.index)}

prof = pd.read_excel(PROFILES_FILE)
idc = [c for c in prof.columns if c.lower() in ("user_id", "main_user", "user", "id")][0]
gt1_map = prof.assign(u=_uid(prof[idc])).set_index("u")["risk_level"].astype(str).str.strip().str.lower().map(GT1)

S = pd.read_excel(SAMPLE_FILE, sheet_name="sample")
S["user"] = _uid(S["user"])
miss = [u for u in S["user"] if u not in pos_of]
assert not miss, f"sampled users without judge scores: {miss[:5]}"
users = list(S["user"])
n = len(users)
w = S["weight"].astype(float).values
strat = S["gt1"].map(GT1).values
y = gt1_map.reindex(users).astype(int).values
assert (y == strat).all(), "GT1 in the sample file differs from Data/user_labels_gt1.xlsx"
Xs = Xall[[pos_of[u] for u in users]]

warn = []


def read_annot(path):
    d = pd.read_excel(path, sheet_name="scoring")
    d["user"] = _uid(d["user"])
    d = d[d["part"].astype(str).str.lower() == "main"].drop_duplicates("user").set_index("user")
    d = d.reindex(users)
    A = d[CODES + ["OVERALL"]].apply(pd.to_numeric, errors="coerce")
    return A, d.get("unsure", pd.Series("", index=d.index)).fillna("").astype(str)


annots, unsure = [], []
for p in ANNOT_FILES:
    A, U = read_annot(p)
    nmiss = int(A.isna().sum().sum())
    bad = int(((A < 0) | (A > 3)).sum().sum())
    if nmiss or bad:
        warn.append(f"{p}: {nmiss} missing and {bad} out-of-range values")
    annots.append(A); unsure.append(U)
if any(a.isna().any().any() for a in annots):
    print("WARNING: missing scores; rows with missing values get weight 0 where used:\n  " + "\n  ".join(warn))

H1, H2 = annots[0][CODES].values, annots[1][CODES].values          # (n,9)
O1, O2 = annots[0]["OVERALL"].values, annots[1]["OVERALL"].values  # (n,)
if os.path.exists(CONSENSUS_FILE):
    cons = pd.read_excel(CONSENSUS_FILE); cons["user"] = _uid(cons["user"])
    cons = cons.set_index("user").reindex(users)
    Hc, Oc = cons[CODES].values.astype(float), cons["OVERALL"].values.astype(float)
    consensus_src = "adjudicated consensus file"
else:
    Hc, Oc = (H1 + H2) / 2, (O1 + O2) / 2
    consensus_src = "mean of the two annotators"
print(f"users: {n} | judges: {judges} | consensus = {consensus_src}")

#  bootstrap indices (stratified, users resampled)
rng = np.random.default_rng(SEED)
BOOT = np.empty((N_BOOT, n), int)
for lv in np.unique(strat):
    idx = np.where(strat == lv)[0]
    BOOT[:, idx] = rng.choice(idx, size=(N_BOOT, len(idx)), replace=True)
ALL = np.arange(n)


def rows(u):                       # user positions -> flattened (user, criterion) rows
    return (np.asarray(u)[:, None] * 9 + np.arange(9)).ravel()


def boot_matrix(fn):
    pt = np.asarray(fn(ALL), float)
    M = np.array([fn(u) for u in BOOT], float)
    return pt, M


def summarise(pt, M, names, extra=None):
    lo, hi = np.nanpercentile(M, 2.5, axis=0), np.nanpercentile(M, 97.5, axis=0)
    return pd.DataFrame({"name": names, "estimate": pt, "ci_low": lo, "ci_high": hi})


def fmt(v):
    return "" if not np.isfinite(v) else f"{v:+.3f}"


def diff_table(pt, M, names, pairs, label):
    out = []
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        d = M[:, ia] - M[:, ib]
        p = 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))
        out.append({"measure": label, "comparison": f"{a} - {b}", "difference": pt[ia] - pt[ib],
                    "ci_low": np.nanpercentile(d, 2.5), "ci_high": np.nanpercentile(d, 97.5), "p_bootstrap": min(1.0, p)})
    return out


result = {}

#  1. inter-annotator agreement
print("inter-annotator agreement ...")
groups = [(c, k) for k, c in enumerate(CODES)] + [("pooled A1-A9", None), ("OVERALL", "ov")]


def get(arr1, arr2, g, u):
    if g is None:
        r = rows(u); return arr1.reshape(-1)[r], arr2.reshape(-1)[r], np.repeat(w[u], 9)
    if g == "ov":
        return O1[u], O2[u], w[u]
    return arr1[u, g], arr2[u, g], w[u]


def ia_fn(g):
    def f(u):
        a, b, ww = get(H1, H2, g, u)
        ok = np.isfinite(a) & np.isfinite(b); ww = ww * ok
        a, b = np.where(ok, a, 0).astype(int), np.where(ok, b, 0).astype(int)
        pr = (a >= PRESENT_HUMAN).astype(int), (b >= PRESENT_HUMAN).astype(int)
        return [wkappa(a, b, ww), wspearman(a.astype(float), b.astype(float), ww),
                float((ww * (a == b)).sum() / ww.sum()), float((ww * (np.abs(a - b) <= 1)).sum() / ww.sum()),
                wkappa(pr[0], pr[1], ww, K=2, quadratic=False)]
    return f


rowsia = []
for name, g in groups:
    pt, M = boot_matrix(ia_fn(g))
    lo, hi = np.nanpercentile(M, 2.5, axis=0), np.nanpercentile(M, 97.5, axis=0)
    r = {"group": name}
    for i, m in enumerate(["quadratic weighted kappa (4 levels)", "Spearman", "exact agreement", "agreement within 1 point",
                           "Cohen kappa for presence (>=2)"]):
        r[m] = pt[i]; r[m + " CI"] = f"[{fmt(lo[i])}, {fmt(hi[i])}]"
    rowsia.append(r)
ia = pd.DataFrame(rowsia)
cnt = pd.DataFrame({
    "group": CODES + ["pooled A1-A9"],
    "unsure annotator 1": [int(unsure[0].str.contains(c).sum()) for c in CODES] + [int((unsure[0].str.strip() != "").sum())],
    "unsure annotator 2": [int(unsure[1].str.contains(c).sum()) for c in CODES] + [int((unsure[1].str.strip() != "").sum())]})
result["inter_annotator"] = ia.merge(cnt, on="group", how="left")

present = (Hc >= PRESENT_HUMAN)
prev = pd.DataFrame({"criterion": CODES,
                     "present in sample (n users)": present.sum(0),
                     "weighted prevalence in the 500 users": [(w * present[:, k]).sum() / w.sum() for k in range(9)]})
result["criteria_present"] = prev

#  2. criterion level: judges and fusion vs human
print("criterion-level agreement with humans ...")
sysC = {}
for j, jd in enumerate(judges):
    sysC[f"judge {jd}"] = Xs[:, :, j]
F1 = fuse(Xs, "F1_mean")[0][0]; F2 = fuse(Xs, "F2_median")[0][0]; F3 = fuse(Xs, "F3_share")[0][0]
sysC["F1 mean"], sysC["F2 median"], sysC["F3 majority share"] = F1, F2, F3
namesC = list(sysC)


def crit_fn(g):
    def f(u):
        out = []
        for s in namesC:
            sc = sysC[s]
            if g is None:
                r = rows(u); a, h, ww = sc.reshape(-1)[r], Hc.reshape(-1)[r], np.repeat(w[u], 9)
            else:
                a, h, ww = sc[u, g], Hc[u, g], w[u]
            ok = np.isfinite(a) & np.isfinite(h); ww = ww * ok
            a, h = np.where(ok, a, 0.0), np.where(ok, h, 0.0)
            out += [wspearman(a, h, ww), wauc(a, h >= PRESENT_HUMAN, ww), wkappa(bin_llm(a), rhu(h), ww)]
        return out
    return f


MET = ["Spearman", "AUC presence", "weighted kappa (4 levels)"]
tabC, pooledM = [], None
for name, g in [(c, k) for k, c in enumerate(CODES)] + [("pooled A1-A9", None)]:
    pt, M = boot_matrix(crit_fn(g))
    lo, hi = np.nanpercentile(M, 2.5, axis=0), np.nanpercentile(M, 97.5, axis=0)
    for si, s in enumerate(namesC):
        r = {"group": name, "system": s}
        for mi, m in enumerate(MET):
            q = si * 3 + mi
            r[m] = pt[q]; r[m + " CI"] = f"[{fmt(lo[q])}, {fmt(hi[q])}]"
        tabC.append(r)
    if g is None:
        pooledM = (pt, M)
result["criterion_vs_human"] = pd.DataFrame(tabC)

# threshold-based metrics (point estimates)
tb = []
for name, g in [(c, k) for k, c in enumerate(CODES)] + [("pooled A1-A9", None)]:
    for s in namesC:
        sc = sysC[s]
        a, h, ww = (sc.reshape(-1), Hc.reshape(-1), np.repeat(w, 9)) if g is None else (sc[:, g], Hc[:, g], w)
        ok = np.isfinite(a) & np.isfinite(h)
        for t in LLM_THRS:
            tb.append({"group": name, "system": s, "LLM threshold": t,
                       **bin_metrics(np.where(ok, a, 0.0), np.where(ok, h, 0.0), ww * ok, t)})
result["criterion_thresholds"] = pd.DataFrame(tb)

# paired differences, pooled (does fusion improve agreement with humans?)
pt, M = pooledM
names3 = [f"{s}|{m}" for s in namesC for m in MET]
fus = ["F1 mean", "F2 median", "F3 majority share"]
dd = []
for mi, m in enumerate(MET):
    sub = [f"{s}|{m}" for s in namesC]
    ptm = np.array([pt[namesC.index(s) * 3 + mi] for s in namesC])
    Mm = np.stack([M[:, namesC.index(s) * 3 + mi] for s in namesC], axis=1)
    pairs = [(a, b) for i, a in enumerate(fus) for b in fus[i + 1:]] + [(a, f"judge {jd}") for a in fus for jd in judges]
    dd += diff_table(ptm, Mm, namesC, pairs, m)
result["fusion_vs_judges_pooled"] = pd.DataFrame(dd)


# 2b. each annotator separately (like-for-like with the agreement between the two annotators)
print("each annotator separately ...")


def single_fn(u):
    out = []
    r, ww = rows(u), np.repeat(w[u], 9)
    for sname in namesC:
        a = sysC[sname].reshape(-1)[r]
        vals = []
        for Hk in (H1, H2):
            h = Hk.reshape(-1)[r]; ok = np.isfinite(a) & np.isfinite(h); wk = ww * ok
            aa, hh = np.where(ok, a, 0.0), np.where(ok, h, 0.0)
            vals += [wspearman(aa, hh, wk), wauc(aa, hh >= PRESENT_HUMAN, wk)]
        out += vals + [(vals[0] + vals[2]) / 2, (vals[1] + vals[3]) / 2]
    # reference: one annotator predicting the other
    h1, h2 = H1.reshape(-1)[r], H2.reshape(-1)[r]
    ok = np.isfinite(h1) & np.isfinite(h2); wk = ww * ok
    h1, h2 = np.where(ok, h1, 0.0), np.where(ok, h2, 0.0)
    sp = wspearman(h1, h2, wk)
    a12, a21 = wauc(h1, h2 >= PRESENT_HUMAN, wk), wauc(h2, h1 >= PRESENT_HUMAN, wk)
    out += [sp, a12, sp, a21, sp, (a12 + a21) / 2]
    return out


pts, Ms = boot_matrix(single_fn)
los, his = np.nanpercentile(Ms, 2.5, axis=0), np.nanpercentile(Ms, 97.5, axis=0)
lab6 = ["Spearman annotator 1", "AUC annotator 1", "Spearman annotator 2", "AUC annotator 2",
        "Spearman mean of the two annotators", "AUC mean of the two annotators"]
tabS = []
for si, sname in enumerate(namesC + ["annotator 1 vs annotator 2 (reference)"]):
    r = {"system": sname}
    for mi, m in enumerate(lab6):
        q = si * 6 + mi
        r[m] = pts[q]; r[m + " CI"] = f"[{fmt(los[q])}, {fmt(his[q])}]"
    tabS.append(r)
result["criterion_vs_each_annotator"] = pd.DataFrame(tabS)

# how each annotator used the scale
dist = []
for nm, Hk, Ok in (("annotator 1", H1, O1), ("annotator 2", H2, O2)):
    row = {"annotator": nm}
    for v in range(4):
        row[f"A1-A9 score {v} (n)"] = int(np.nansum(Hk == v))
    for v in range(4):
        row[f"OVERALL {v} (n)"] = int(np.nansum(Ok == v))
    cnt_k = np.nansum(Hk >= PRESENT_HUMAN, axis=1)
    row["users with >=4 criteria present"] = int((cnt_k >= DSM_MIN).sum())
    row["mean criteria present per user"] = float(cnt_k.mean())
    dist.append(row)
result["annotator_scale_use"] = pd.DataFrame(dist)

#  3. user level: humans vs GT1, framework, zero-shot
print("user-level comparison (framework and zero-shot) ...")
count = (Hc >= PRESENT_HUMAN).sum(1).astype(float)
dsm = count >= DSM_MIN
ov_pos = Oc >= PRESENT_HUMAN
count1, count2 = (H1 >= PRESENT_HUMAN).sum(1).astype(float), (H2 >= PRESENT_HUMAN).sum(1).astype(float)

userSys = {"GT1 (reference)": y.astype(float)}
userSys["F1 + mean"] = run(Xs, "F1_mean", "mean")
userSys["F2 + mean (A1)"] = run(Xs, "F2_median", "mean")
userSys["F2 + crisp OWA (A2)"] = run(Xs, "F2_median", "owa")
userSys["F4 + T1OWA Yager (A3)"] = run(Xs, "F4_fuzzy", "owa")
zs_names = []
if os.path.exists(ZEROSHOT_FILE):
    zs = pd.read_excel(ZEROSHOT_FILE, sheet_name="predictions")
    zs["User"] = _uid(zs["User"]); zs = zs.drop_duplicates("User").set_index("User")
    for c in ["mean_score", "yager_score", "llama370", "llama3", "deepseek", "qwen3", "gemma3"]:
        if c in zs.columns:
            v = pd.to_numeric(zs[c], errors="coerce").reindex(users).values
            userSys[f"zero-shot {c}"] = v; zs_names.append(f"zero-shot {c}")
else:
    warn.append("zero-shot file not found: zero-shot comparison skipped")
namesU = list(userSys)
targets = {"human count of present criteria": count, "human OVERALL score": Oc}
posT = {"DSM-5 presence (count >= 4)": dsm, "OVERALL >= 2": ov_pos}


def user_fn(u):
    out = []
    for s in namesU:
        a = userSys[s][u]; ok = np.isfinite(a); ww = w[u] * ok; a = np.where(ok, a, 0.0)
        for t in targets.values():
            out.append(wspearman(a, t[u], ww))
        for t in posT.values():
            out.append(wauc(a, t[u], ww))
    return out


UM = list(targets) + [f"AUC {k}" for k in posT]
ptU, MU = boot_matrix(user_fn)
loU, hiU = np.nanpercentile(MU, 2.5, axis=0), np.nanpercentile(MU, 97.5, axis=0)
tabU = []
for si, s in enumerate(namesU):
    r = {"system": s}
    for mi, m in enumerate(["Spearman with " + k for k in targets] + ["AUC for " + k for k in posT]):
        q = si * 4 + mi
        r[m] = ptU[q]; r[m + " CI"] = f"[{fmt(loU[q])}, {fmt(hiU[q])}]"
    tabU.append(r)
# human ceiling (annotator 1 predicts annotator 2)
def ceil_fn(u):
    return [wspearman(count1[u], count2[u], w[u]), wspearman(O1[u], O2[u], w[u]),
            wauc(count1[u], count2[u] >= DSM_MIN, w[u]), wauc(O1[u], O2[u] >= PRESENT_HUMAN, w[u])]
ptc, Mc = boot_matrix(ceil_fn); loc, hic = np.nanpercentile(Mc, 2.5, axis=0), np.nanpercentile(Mc, 97.5, axis=0)
r = {"system": "human ceiling (annotator 1 vs annotator 2)"}
for mi, m in enumerate(["Spearman with " + k for k in targets] + ["AUC for " + k for k in posT]):
    r[m] = ptc[mi]; r[m + " CI"] = f"[{fmt(loc[mi])}, {fmt(hic[mi])}]"
tabU.append(r)
result["user_level_vs_human"] = pd.DataFrame(tabU)

fw = ["F2 + mean (A1)", "F2 + crisp OWA (A2)", "F4 + T1OWA Yager (A3)"]
dU = []
for mi, m in enumerate(["Spearman with " + k for k in targets] + ["AUC for " + k for k in posT]):
    ptm = np.array([ptU[si * 4 + mi] for si in range(len(namesU))])
    Mm = np.stack([MU[:, si * 4 + mi] for si in range(len(namesU))], axis=1)
    pairs = [(z, f) for z in zs_names if z in ("zero-shot mean_score", "zero-shot yager_score") for f in fw]
    pairs += [("F4 + T1OWA Yager (A3)", "F2 + crisp OWA (A2)"), ("F2 + crisp OWA (A2)", "F2 + mean (A1)"),
              ("F4 + T1OWA Yager (A3)", "F2 + mean (A1)"), ("F4 + T1OWA Yager (A3)", "GT1 (reference)"),
              ("F2 + crisp OWA (A2)", "GT1 (reference)"), ("zero-shot mean_score", "GT1 (reference)")]
    dU += diff_table(ptm, Mm, namesU, pairs, m)
result["zero_shot_vs_framework"] = pd.DataFrame(dU)


#  3b. user level, each annotator separately
def user_single_fn(u):
    out = []
    for sname in namesU:
        a = userSys[sname][u]; ok = np.isfinite(a); aa = np.where(ok, a, 0.0)
        v = []
        for T in (count1, count2, O1, O2):
            wk = w[u] * ok * np.isfinite(T[u])
            v.append(wspearman(aa, np.where(np.isfinite(T[u]), T[u], 0.0), wk))
        out += v + [(v[0] + v[1]) / 2, (v[2] + v[3]) / 2]
    cs = wspearman(count1[u], count2[u], w[u]); osp = wspearman(O1[u], O2[u], w[u])
    out += [cs, cs, osp, osp, cs, osp]
    return out


ptu, Mu = boot_matrix(user_single_fn)
lou, hiu = np.nanpercentile(Mu, 2.5, axis=0), np.nanpercentile(Mu, 97.5, axis=0)
lab6u = ["Spearman with count, annotator 1", "Spearman with count, annotator 2", "Spearman with OVERALL, annotator 1",
         "Spearman with OVERALL, annotator 2", "Spearman with count, average of the two", "Spearman with OVERALL, average of the two"]
tabUS = []
for si, sname in enumerate(namesU + ["annotator 1 vs annotator 2 (reference)"]):
    r = {"system": sname}
    for mi, m in enumerate(lab6u):
        q = si * 6 + mi
        r[m] = ptu[q]; r[m + " CI"] = f"[{fmt(lou[q])}, {fmt(hiu[q])}]"
    tabUS.append(r)
result["user_level_each_annotator"] = pd.DataFrame(tabUS)

# 4. humans vs GT1
print("humans vs GT1 ...")


def gt_fn(u):
    return [wspearman(Oc[u], y[u].astype(float), w[u]), wspearman(count[u], y[u].astype(float), w[u]),
            wkappa(rhu(Oc[u]), y[u], w[u]),
            wauc(count[u], y[u] == 3, w[u]), wauc(Oc[u], y[u] == 3, w[u])]


ptg, Mg = boot_matrix(gt_fn); log, hig = np.nanpercentile(Mg, 2.5, axis=0), np.nanpercentile(Mg, 97.5, axis=0)
labg = ["Spearman: OVERALL vs GT1", "Spearman: count of present criteria vs GT1", "quadratic weighted kappa: OVERALL (rounded) vs GT1",
        "AUC: count present, GT1 high vs rest", "AUC: OVERALL, GT1 high vs rest"]
result["humans_vs_gt1"] = pd.DataFrame({"measure": labg, "estimate": ptg,
                                        "ci_low": log, "ci_high": hig})
cm = pd.crosstab(pd.Series(rhu(Oc), name="human OVERALL (rounded)"), pd.Series(y, name="GT1 level"))
cm = cm.reindex(index=range(4), columns=range(4), fill_value=0)
cm.index = [f"human {i}" for i in range(4)]; cm.columns = [f"GT1 {LEVELS[i]}" for i in range(4)]
result["confusion_overall_gt1"] = cm.reset_index()
bys = pd.DataFrame({"GT1 level": LEVELS, "n users": [int((y == i).sum()) for i in range(4)],
                    "mean count present": [count[y == i].mean() for i in range(4)],
                    "share with count >= 4": [dsm[y == i].mean() for i in range(4)],
                    "mean OVERALL": [Oc[y == i].mean() for i in range(4)]})
result["human_by_gt1_level"] = bys

# 5. representativeness of the sample
print("representativeness ...")
N_all = len(raw)
score_all = run(Xall, "F2_median", "owa")
sd_all = np.nanmean(np.nanstd(Xall, axis=2), axis=1)
vars_ = {"F2 + crisp OWA score": score_all, "mean judge SD across criteria": sd_all}
if os.path.exists(TEXTS_FILE):
    t = pd.read_csv(TEXTS_FILE)
    tid = [c for c in t.columns if str(c).lower() in ("user_id", "main_user", "user", "id")][0]
    tx = max([c for c in t.columns if c != tid and not pd.api.types.is_numeric_dtype(t[c])],
             key=lambda c: t[c].astype(str).str.len().mean())
    t["u"] = _uid(t[tid]); ln = t.groupby("u")[tx].apply(lambda x: x.astype(str).str.len().sum())
    vars_["text length (characters)"] = ln.reindex(raw.index).values.astype(float)
in_s = np.zeros(N_all, bool); in_s[[pos_of[u] for u in users]] = True
reprow = []
for name, v in vars_.items():
    a, b = v[in_s], v[~in_s]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    wm = np.nansum(w * v[[pos_of[u] for u in users]]) / w.sum()
    reprow.append({"variable": name, "mean sample (unweighted)": a.mean(), "mean other users": b.mean(),
                   "standardised mean difference": (a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                   "KS p-value": ks_2samp(a, b).pvalue, "weighted sample mean": wm,
                   "mean of all 500 users": np.nanmean(v)})
result["representativeness"] = pd.DataFrame(reprow)

print("ranking agreement between GT1 and human reference ...")
from scipy.stats import kendalltau

rank_sys = {
    "F1 + crisp OWA":        run(Xs, "F1_mean", "owa"),
    "F2 + mean":             run(Xs, "F2_median", "mean"),
    "F2 + crisp OWA":        run(Xs, "F2_median", "owa"),
    "F3 (vote share) + mean": run(Xs, "F3_share", "mean"),
    "F4 + T1OWA Yager":      run(Xs, "F4_fuzzy", "owa"),
    "F4 + T1OWA equal":      run(Xs, "F4_fuzzy", "mean"),
}
rank_names = list(rank_sys)
rank_refs = {"human count of present criteria": count, "human OVERALL score": Oc}


def rank_fn(u):
    out = []
    for ref in rank_refs.values():
        rg = np.array([wspearman(rank_sys[s][u], y.astype(float)[u], w[u]) for s in rank_names])   # pipeline vs GT1
        rh = np.array([wspearman(rank_sys[s][u], ref[u], w[u]) for s in rank_names])               # pipeline vs human
        tau = kendalltau(rg, rh)[0] if np.ptp(rg) > 0 and np.ptp(rh) > 0 else np.nan
        out.append(tau)
    return out


ptR, MR = boot_matrix(rank_fn)
rows_r = []
for i, ref in enumerate(rank_refs):
    lo, hi = np.nanpercentile(MR[:, i], 2.5), np.nanpercentile(MR[:, i], 97.5)
    rg = [wspearman(rank_sys[s][ALL], y.astype(float), w) for s in rank_names]
    rh = [wspearman(rank_sys[s][ALL], rank_refs[ref], w) for s in rank_names]
    rows_r.append({"human reference": ref, "Kendall tau between the two rankings": ptR[i],
                   "ci_low": lo, "ci_high": hi, "n pipelines": len(rank_names),
                   "valid resamples": int(np.isfinite(MR[:, i]).sum()),
                   "ranking by GT1": " > ".join(np.array(rank_names)[np.argsort(rg)[::-1]]),
                   "ranking by human": " > ".join(np.array(rank_names)[np.argsort(rh)[::-1]])})
result["pipeline_ranking_agreement"] = pd.DataFrame(rows_r)
detail = pd.DataFrame({"pipeline": rank_names,
                       "Spearman with GT1 (100 users)": [wspearman(rank_sys[s][ALL], y.astype(float), w) for s in rank_names],
                       "Spearman with human count": [wspearman(rank_sys[s][ALL], count, w) for s in rank_names],
                       "Spearman with human OVERALL": [wspearman(rank_sys[s][ALL], Oc, w) for s in rank_names]})
result["pipeline_ranking_detail"] = detail
print(result["pipeline_ranking_agreement"][["human reference", "Kendall tau between the two rankings", "ci_low", "ci_high"]].to_string(index=False))


#  write
meta = pd.DataFrame({"item": ["users annotated", "bootstrap resamples", "seed", "consensus", "human presence", "DSM-5 presence",
                               "judge-score bins (4 levels)", "warnings"],
                     "value": [n, N_BOOT, SEED, consensus_src, f"score >= {PRESENT_HUMAN}", f"count >= {DSM_MIN}",
                               "0 | 0.10-0.30 | 0.50 | >=0.70 (cuts 0.05/0.40/0.60)", "; ".join(warn) or "none"]})
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUTPUT_FILE) as wr:
    meta.to_excel(wr, sheet_name="settings", index=False)
    for k, v in result.items():
        v.round(4).to_excel(wr, sheet_name=k[:31], index=False)

# console summary
p = result["inter_annotator"]
print("\n== inter-annotator (pooled A1-A9): kappa %.3f %s | Spearman %.3f" % (
    p.loc[p.group == "pooled A1-A9", "quadratic weighted kappa (4 levels)"].iloc[0],
    p.loc[p.group == "pooled A1-A9", "quadratic weighted kappa (4 levels) CI"].iloc[0],
    p.loc[p.group == "pooled A1-A9", "Spearman"].iloc[0]))
c = result["criterion_vs_human"]; c = c[c.group == "pooled A1-A9"]
print("== pooled criterion level vs human (Spearman | AUC presence | kappa):")
for _, r in c.iterrows():
    print(f"   {r['system']:<28} {r['Spearman']:.3f} {r['Spearman CI']:<18} {r['AUC presence']:.3f}  {r['weighted kappa (4 levels)']:.3f}")
print("== user level (Spearman with human count | AUC DSM presence):")
for _, r in result["user_level_vs_human"].iterrows():
    print(f"   {r['system']:<42} {r['Spearman with human count of present criteria']:.3f}  {r['AUC for DSM-5 presence (count >= 4)']:.3f}")
print("saved", OUTPUT_FILE, "| warnings:", warn or "none")

