# Additional analyses for the revision.
# Reads the judge scores and GT1 labels.
# Input: data/judge_scores_nine_criteria.xlsx, data/user_labels_gt1.xlsx, results/baseline_llm_zeroshot_all.xlsx, code/type1owa_zhou.py
# Output: result/revision_analyses.xlsx.
import os, re, sys, itertools, warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

USER_SCORES   = "data/judge_scores_nine_criteria.xlsx"
PROFILES_FILE = "data/user_labels_gt1.xlsx"
ZEROSHOT_FILE = "results/baseline_llm_zeroshot_all.xlsx"
OUTPUT_FILE   = "result/revision_analyses.xlsx"
T1OWA_DIR     = os.path.dirname(os.path.abspath(__file__))

PRESENT_THR = 0.50
MIN_JUDGES  = 2
YAGER_Q     = 0.5
N_BOOT, SEED = 2000, 42
GRID_STEP, N_ALPHA = 101, 50
THRESHOLDS  = [0.3, 0.4, 0.5, 0.6, 0.7]
MARGINS     = [0.05, 0.10]
KEEP_FRACS  = [1.0, 0.8, 0.6, 0.5]
CV_REPEATS  = 5
SCALE_GRID  = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

sys.path.insert(0, T1OWA_DIR)
from type1owa_zhou import aggregate_triangular_t1owa
domainX = np.round(np.linspace(0.0, 1.0, GRID_STEP), 2)
alphas = np.linspace(0.0, 1.0, N_ALPHA)
GT1 = {"none": 0, "low": 1, "moderate": 2, "high": 3}
LEVELS = list(GT1)

# ---------- data ----------
raw = pd.read_excel(USER_SCORES, index_col=0)
raw.index = raw.index.astype(str).str.strip()
pat = re.compile(r"^A([1-9])_(.+)$")
judges = []
for c in raw.columns:
    m = pat.match(c)
    if m and m.group(2) != "n_llm" and m.group(2) not in judges:
        judges.append(m.group(2))
J = len(judges)
X = np.full((len(raw), 9, J), np.nan)
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
N = len(y)
print(f"judges: {judges} | users: {N} | label counts: {np.bincount(y).tolist()}")

# ---------- fusion and aggregation (same definitions as experiments_1_3.py) ----------
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
    elif strategy == "F3_majority": out = ((share > 0.5).astype(float),)
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

def triangles(abc, assessed, i, skip=None):
    a, b, c = abc
    P = []
    for k in np.where(assessed[i])[0]:
        if k == skip: continue
        bb = round(float(b[i, k]), 2); aa = min(round(float(a[i, k]), 2), bb); cc = max(round(float(c[i, k]), 2), bb)
        if bb == 0.0 and cc - aa < 1e-3: cc = 1e-3
        P.append((aa, bb, cc, 1.0))
    return P

def t1owa(P, q=YAGER_Q, equal=False):
    w = None if equal else yager_w(len(P), q)
    try:
        agg, _, cen = aggregate_triangular_t1owa(P, domain_x=domainX, alpha_levels=alphas, weights=w)
    except Exception:
        return None, np.mean([p[1] for p in P])
    if not np.isfinite(cen):
        return None, np.mean([p[1] for p in P])
    return agg, cen

def agg_fuzzy(abc, assessed, equal=False, q=YAGER_Q):
    out = np.zeros(assessed.shape[0])
    for i in range(len(out)):
        P = triangles(abc, assessed, i)
        if P: out[i] = t1owa(P, q, equal)[1]
    return out

def run(Xs, fusion, op, q=YAGER_Q):
    f, assessed = fuse(Xs, fusion)
    if fusion == "F4_fuzzy":
        return agg_fuzzy(tuple(f), assessed, equal=(op == "mean"), q=q)
    return agg_crisp(f[0], assessed, op, q)

# ---------- metrics and paired bootstrap ----------
def auc(s, pos):
    pos = np.asarray(pos, bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auc_nh(s, yy):
    m = (yy == 0) | (yy == 3); return auc(s[m], yy[m] == 3)

def sp(s, yy):
    r = spearmanr(s, yy)[0]; return 0.0 if np.isnan(r) else r

METRICS = {"spearman": sp, "auc_none_vs_high": auc_nh}
BOOT = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))

def bstats(s):
    return {m: (fn(s, y), np.array([fn(s[b], y[b]) for b in BOOT])) for m, fn in METRICS.items()}

def fmt(v, bs):
    bs = bs[~np.isnan(bs)]
    return f"{v:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"

def row_of(name, s, st=None):
    st = st or bstats(s)
    r = {"method": name}
    for m in METRICS: r[m] = fmt(*st[m])
    return r

def diff_cell(st1, st2, m):
    d0 = st1[m][0] - st2[m][0]; d = st1[m][1] - st2[m][1]; d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{d0:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if lo > 0 or hi < 0 else "")

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run_max = 0.0
    for rank, idx in enumerate(order):
        run_max = max(run_max, (m - rank) * p[idx]); adj[idx] = min(1.0, run_max)
    return adj

# ---------- baseline pipelines ----------
t0 = time.time()
S = {}
for f in ("F1_mean", "F2_median", "F3_majority", "F4_fuzzy"):
    for op in ("mean", "owa"):
        S[(f, op)] = run(X, f, op)
print(f"baseline pipelines done in {time.time() - t0:.0f}s")
ST = {k: bstats(v) for k, v in S.items()}
PIPE_LABEL = {("F1_mean", "mean"): "F1 + mean", ("F1_mean", "owa"): "F1 + OWA", ("F2_median", "mean"): "F2 + mean (A1)",
              ("F2_median", "owa"): "F2 + crisp OWA (A2)", ("F3_majority", "mean"): "F3 + mean", ("F3_majority", "owa"): "F3 + OWA",
              ("F4_fuzzy", "mean"): "F4 + T1OWA equal (A4)", ("F4_fuzzy", "owa"): "F4 + T1OWA Yager (A3)"}
A2, A3 = S[("F2_median", "owa")], S[("F4_fuzzy", "owa")]
print(f"check F2 + crisp OWA: {fmt(*ST[('F2_median', 'owa')]['spearman'])}")

# ---------- R2.5: score inventory ----------
inv = []
for j, jd in enumerate(judges):
    v = X[:, :, j]; m = np.isfinite(v); vv = v[m]
    off = np.min(np.abs(vv[:, None] - np.array(SCALE_GRID)[None, :]), axis=1) > 1e-6
    inv.append({"judge": jd, "valid scores": int(m.sum()), "missing": int((~m).sum()), "min": float(vv.min()), "max": float(vv.max()),
                "mean": round(float(vv.mean()), 4), "outside 0-1": int(((vv < 0) | (vv > 1)).sum()),
                "not on the prompt scale (0, .1, .3, .5, .7, .9, 1)": int(off.sum())})
inv = pd.DataFrame(inv)

# ---------- R1.6: threshold sensitivity of majority voting ----------
rates, perf = [], []
for thr in THRESHOLDS:
    f3, ass = fuse(X, "F3_majority", thr)
    rates.append({"threshold": thr, **{f"A{k+1}": round(float(f3[0][:, k][ass[:, k]].mean() * 100), 1) for k in range(9)}})
    cand = {f"F3 majority + mean (thr {thr})": agg_crisp(f3[0], ass, "mean"),
            f"F3 majority + OWA (thr {thr})": agg_crisp(f3[0], ass, "owa")}
    sh, ass2 = fuse(X, "F3_share", thr)
    cand[f"F3 vote share + mean (thr {thr})"] = agg_crisp(sh[0], ass2, "mean")
    cand[f"F3 vote share + OWA (thr {thr})"] = agg_crisp(sh[0], ass2, "owa")
    for nm, s in cand.items():
        st = bstats(s); r = row_of(nm, s, st)
        r["diff vs F2 + crisp OWA (spearman)"] = diff_cell(st, ST[("F2_median", "owa")], "spearman")
        r["diff vs F2 + crisp OWA (auc_none_vs_high)"] = diff_cell(st, ST[("F2_median", "owa")], "auc_none_vs_high")
        perf.append(r)
rates = pd.DataFrame(rates).rename(columns={"threshold": "judge vote threshold: % of users with criterion present"})
perf = pd.DataFrame(perf)

# ---------- R1.4: judge normalisation ----------
def normalise(Xs, method):
    Xn = Xs.copy(); pooled = Xs[np.isfinite(Xs)]; pool_sorted = np.sort(pooled)
    for j in range(Xs.shape[2]):
        v = Xs[:, :, j]; m = np.isfinite(v)
        if method == "quantile":
            u = (rankdata(v[m], method="average") - 0.5) / m.sum()
            Xn[:, :, j][m] = np.quantile(pool_sorted, u)
        else:
            Xn[:, :, j][m] = np.clip(v[m] - v[m].mean() + pooled.mean(), 0.0, 1.0)
    return Xn

def extremes(Xs):
    n_valid = np.isfinite(Xs).sum(axis=2)
    with np.errstate(all="ignore"):
        mx = np.nanmax(Xs, axis=2, keepdims=True); mn = np.nanmin(Xs, axis=2, keepdims=True)
    dis = (n_valid >= 2) & (mx[:, :, 0] > mn[:, :, 0] + 1e-9)
    out = []
    for j, jd in enumerate(judges):
        v = Xs[:, :, j]
        out.append({"judge": jd, "pairs with disagreement": int(dis.sum()),
                    "attains max": round(float((np.abs(v - mx[:, :, 0]) < 1e-9)[dis].mean()), 3),
                    "attains min": round(float((np.abs(v - mn[:, :, 0]) < 1e-9)[dis].mean()), 3)})
    return out

norm_rows, ext_rows, wid_rows = [], [], []
variants = {"original scores": X, "quantile-normalised per judge": normalise(X, "quantile"), "mean-shifted per judge": normalise(X, "shift")}
for vn, Xv in variants.items():
    for f, op, lab in (("F1_mean", "mean", "F1 + mean"), ("F2_median", "mean", "F2 + mean (A1)"),
                       ("F2_median", "owa", "F2 + crisp OWA (A2)"), ("F4_fuzzy", "owa", "F4 + T1OWA Yager (A3)")):
        s = S[(f, op)] if vn == "original scores" else run(Xv, f, op)
        st = ST[(f, op)] if vn == "original scores" else bstats(s)
        r = row_of(f"{lab} | {vn}", s, st)
        r["diff vs original (spearman)"] = "" if vn == "original scores" else diff_cell(st, ST[(f, op)], "spearman")
        norm_rows.append(r)
    for e in extremes(Xv): ext_rows.append({"scores": vn, **e})
    f4v, assv = fuse(Xv, "F4_fuzzy")
    wv = np.nanmean(np.where(assv, f4v[2] - f4v[0], np.nan), axis=1); sdv = np.nanmean(np.nanstd(Xv, axis=2), axis=1)
    wid_rows.append({"scores": vn, **{f"mean width, GT1 {lv}": round(float(np.nanmean(wv[y == i])), 4) for i, lv in enumerate(LEVELS)},
                     "spearman(width, GT1)": round(sp(wv, y), 3), "spearman(judge SD, GT1)": round(sp(sdv, y), 3),
                     "spearman(width, judge SD)": round(float(spearmanr(wv, sdv)[0]), 3)})
norm_perf, ext_tab, wid_tab = pd.DataFrame(norm_rows), pd.DataFrame(ext_rows), pd.DataFrame(wid_rows)
print(f"normalisation done in {time.time() - t0:.0f}s")

# ---------- R1.5: properties of the aggregated fuzzy sets (T1OWA Yager) ----------
f4, ass4 = fuse(X, "F4_fuzzy")
cols = {k: np.full(N, np.nan) for k in ("centroid", "support_width", "width_alpha05", "asymmetry", "loco_max", "loco_mean", "input_width", "judge_sd")}
for i in range(N):
    P = triangles(f4, ass4, i)
    if not P: continue
    agg, cen = t1owa(P)
    cols["centroid"][i] = cen
    if agg is not None:
        mf, dm = agg.mf_value, agg.domain
        sup = dm[mf > 0]; cols["support_width"][i] = (sup.max() - sup.min()) if sup.size else 0.0
        core = dm[mf >= 0.5]
        if core.size:
            L, R = core.min(), core.max(); cols["width_alpha05"][i] = R - L
            cols["asymmetry"][i] = ((R - cen) - (cen - L)) / (R - L) if R > L else 0.0
    if len(P) > 1:
        ch = []
        for k in np.where(ass4[i])[0]:
            Q = triangles(f4, ass4, i, skip=k)
            ch.append(abs(t1owa(Q)[1] - cen))
        cols["loco_max"][i], cols["loco_mean"][i] = max(ch), float(np.mean(ch))
    cols["input_width"][i] = np.nanmean(np.where(ass4[i], f4[2][i] - f4[0][i], np.nan))
    cols["judge_sd"][i] = np.nanmean(np.nanstd(X[i], axis=1))
FZ = pd.DataFrame(cols)
print(f"fuzzy-set analysis done in {time.time() - t0:.0f}s | max |centroid - A3| = {np.nanmax(np.abs(FZ['centroid'].values - A3)):.4f}")

pr_y = (rankdata(y) - 0.5) / N
pr_s = (rankdata(A3) - 0.5) / N
err = np.abs(pr_s - pr_y)
desc = []
for nm in ("centroid", "support_width", "width_alpha05", "asymmetry", "loco_max", "loco_mean", "input_width", "judge_sd"):
    v = FZ[nm].values
    desc.append({"quantity (per user)": nm, **{lv: round(float(np.nanmean(v[y == i])), 4) for i, lv in enumerate(LEVELS)},
                 "spearman with GT1": round(sp(np.nan_to_num(v), y), 3),
                 "spearman with absolute ranking error": round(float(spearmanr(np.nan_to_num(v), err)[0]), 3),
                 "spearman with judge SD": round(float(spearmanr(np.nan_to_num(v), np.nan_to_num(FZ["judge_sd"].values))[0]), 3)})
fz_desc = pd.DataFrame(desc)

def sel_stat(unc, frac, idx, score, yy):
    w = unc[idx]; k = w <= np.quantile(w, frac)
    s_, y_ = score[idx][k], yy[idx][k]
    return sp(s_, y_), auc_nh(s_, y_)

rng = np.random.default_rng(SEED + 1)
sel = []
for nm in ("width_alpha05", "support_width", "input_width", "judge_sd", "loco_max"):
    unc = np.nan_to_num(FZ[nm].values)
    for frac in KEEP_FRACS:
        pt = sel_stat(unc, frac, np.arange(N), A3, y)
        bs = np.array([sel_stat(unc, frac, b, A3, y) for b in BOOT[:500]])  
        sel.append({"uncertainty measure": nm, "kept (least uncertain)": frac, "users kept": int((unc <= np.quantile(unc, frac)).sum()),
                    "spearman": f"{pt[0]:.3f} [{np.percentile(bs[:, 0], 2.5):.3f}, {np.percentile(bs[:, 0], 97.5):.3f}]",
                    "auc_none_vs_high": f"{pt[1]:.3f} [{np.nanpercentile(bs[:, 1], 2.5):.3f}, {np.nanpercentile(bs[:, 1], 97.5):.3f}]"})
rand = []
for frac in KEEP_FRACS[1:]:
    vals = []
    for _ in range(300):
        keep_idx = rng.choice(N, int(round(frac * N)), replace=False)
        vals.append((sp(A3[keep_idx], y[keep_idx]), auc_nh(A3[keep_idx], y[keep_idx])))
    vals = np.array(vals)
    rand.append({"uncertainty measure": "random subset (300 draws)", "kept (least uncertain)": frac, "users kept": int(round(frac * N)),
                 "spearman": f"{vals[:, 0].mean():.3f} (draw SD {vals[:, 0].std():.3f})",
                 "auc_none_vs_high": f"{np.nanmean(vals[:, 1]):.3f} (draw SD {np.nanstd(vals[:, 1]):.3f})"})
sel = pd.DataFrame(sel + rand)

# ---------- R2.2: differences, equivalence (TOST) and Holm correction ----------
pairs = []
FUS = ["F1_mean", "F2_median", "F3_majority", "F4_fuzzy"]
for op in ("mean", "owa"):
    for a, b in itertools.combinations(FUS, 2): pairs.append(((a, op), (b, op)))
extra = [(("F2_median", "mean"), ("F2_median", "owa")), (("F2_median", "mean"), ("F4_fuzzy", "owa")),
         (("F2_median", "owa"), ("F4_fuzzy", "mean")), (("F4_fuzzy", "owa"), ("F4_fuzzy", "mean"))]
pairs += extra
eq_rows = []
for m in METRICS:
    block = []
    for p1, p2 in pairs:
        d = ST[p1][m][1] - ST[p2][m][1]; d = d[~np.isnan(d)]; d0 = ST[p1][m][0] - ST[p2][m][0]; B = len(d)
        p_diff = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (B + 1))
        row = {"metric": m, "comparison": f"{PIPE_LABEL[p1]} - {PIPE_LABEL[p2]}", "difference": round(d0, 4),
               "95% interval": f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]", "p (bootstrap)": round(p_diff, 4)}
        lo90, hi90 = np.percentile(d, [5, 95])
        for dl in MARGINS:
            row[f"TOST p (margin {dl})"] = max(((d <= -dl).sum() + 1) / (B + 1), ((d >= dl).sum() + 1) / (B + 1))
            row[f"90% interval inside +-{dl}"] = bool(lo90 > -dl and hi90 < dl)
        block.append(row)
    ph = holm([r["p (bootstrap)"] for r in block])
    for r, a in zip(block, ph): r["p Holm (16 comparisons)"] = round(a, 4)
    for dl in MARGINS:
        th = holm([r[f"TOST p (margin {dl})"] for r in block])
        for r, a in zip(block, th):
            r[f"TOST p Holm (margin {dl})"] = round(a, 4)
            r[f"equivalent at {dl} (Holm p<0.05)"] = bool(a < 0.05)
            r[f"TOST p (margin {dl})"] = round(r[f"TOST p (margin {dl})"], 4)
    eq_rows += block
eq = pd.DataFrame(eq_rows)

# ---------- R2.4: zero-shot comparison and supervised aggregation ----------
zs_rows = []
try:
    zs = pd.read_excel(ZEROSHOT_FILE, sheet_name="predictions")
    zs["User"] = zs["User"].astype(str).str.strip()
    zs = zs.set_index("User").reindex(users)
    assert (zs["y"].values == y).all(), "label mismatch between zero-shot file and GT1"
    zcand = {"Zero-shot Yager OWA of the five models": zs["yager_score"].values, "Zero-shot mean of the five models": zs["mean_score"].values}
    zst = {k: bstats(np.asarray(v, float)) for k, v in zcand.items()}
    for zn, st in zst.items():
        for lab, ref in (("F2 + crisp OWA (A2)", ST[("F2_median", "owa")]), ("F4 + T1OWA Yager (A3)", ST[("F4_fuzzy", "owa")])):
            zs_rows.append({"comparison": f"{zn} - {lab}", **{m: diff_cell(st, ref, m) for m in METRICS}})
except Exception as e:
    zs_rows.append({"comparison": f"zero-shot file not used: {e}"})
zs_tab = pd.DataFrame(zs_rows)

def cv_scores(F, make):
    acc = np.zeros(N)
    for r in range(CV_REPEATS):
        oof = np.zeros(N)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(F, y):
            oof[te] = make().fit(F[tr], y[tr]).predict(F[te])
        acc += oof
    return acc / CV_REPEATS

sup_rows, w_rows, o_rows = [], [], []
for fname, fkey in (("F2 (median)", "F2_median"), ("F1 (mean)", "F1_mean")):
    Fv = fuse(X, fkey)[0][0]; Fs = -np.sort(-Fv, axis=1)
    models = {"ridge on the nine criteria": (Fv, lambda: RidgeCV(alphas=np.logspace(-3, 3, 13))),
              "non-negative weighted mean (learned criterion weights)": (Fv, lambda: LinearRegression(positive=True)),
              "learned OWA (non-negative position weights)": (Fs, lambda: LinearRegression(positive=True))}
    for mn, (F, make) in models.items():
        s = cv_scores(F, make); st = bstats(s)
        r = row_of(f"{mn} | {fname}, 5-fold CV x{CV_REPEATS}", s, st)
        r["diff vs F2 + crisp OWA (spearman)"] = diff_cell(st, ST[("F2_median", "owa")], "spearman")
        r["diff vs F2 + crisp OWA (auc_none_vs_high)"] = diff_cell(st, ST[("F2_median", "owa")], "auc_none_vs_high")
        sup_rows.append(r)
        mdl = make().fit(F, y)
        coef = np.asarray(mdl.coef_, float); coef = coef / coef.sum() if coef.sum() > 0 else coef
        if "OWA" in mn: o_rows.append({"model": f"{mn} | {fname}", **{f"position {i+1}": round(float(c), 3) for i, c in enumerate(coef)}})
        elif "weighted mean" in mn: w_rows.append({"model": f"{mn} | {fname}", **{f"A{i+1}": round(float(c), 3) for i, c in enumerate(coef)}})
sup = pd.DataFrame(sup_rows); wtab = pd.DataFrame(w_rows); otab = pd.DataFrame(o_rows)

# ---------- save ----------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUTPUT_FILE) as w:
    inv.to_excel(w, sheet_name="r25_score_inventory", index=False)
    rates.to_excel(w, sheet_name="r16_f3_rates", index=False)
    perf.to_excel(w, sheet_name="r16_f3_thresholds", index=False)
    norm_perf.to_excel(w, sheet_name="r14_normalisation", index=False)
    ext_tab.to_excel(w, sheet_name="r14_extremes", index=False)
    wid_tab.to_excel(w, sheet_name="r14_width", index=False)
    fz_desc.to_excel(w, sheet_name="r15_fuzzy_outputs", index=False)
    sel.to_excel(w, sheet_name="r15_selective", index=False)
    FZ.insert(0, "User", users); FZ.to_excel(w, sheet_name="r15_per_user", index=False)
    eq.to_excel(w, sheet_name="r22_equivalence", index=False)
    zs_tab.to_excel(w, sheet_name="r24_zero_shot_paired", index=False)
    sup.to_excel(w, sheet_name="r24_supervised_agg", index=False)
    wtab.to_excel(w, sheet_name="r24_learned_weights", index=False)
    otab.to_excel(w, sheet_name="r24_learned_owa_weights", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
for nm, t in (("R2.5 score inventory", inv), ("R1.6 F3 present (%)", rates), ("R1.6 F3 thresholds", perf), ("R1.4 normalisation", norm_perf),
              ("R1.4 extremes", ext_tab), ("R1.4 width", wid_tab), ("R1.5 fuzzy outputs", fz_desc), ("R1.5 selective", sel),
              ("R2.4 zero-shot", zs_tab), ("R2.4 supervised aggregation", sup), ("R2.4 learned criterion weights", wtab), ("R2.4 learned OWA weights", otab)):
    print(f"\n--- {nm} ---\n{t.to_string(index=False)}")
print(f"\nSaved -> {OUTPUT_FILE} ({time.time() - t0:.0f}s)")