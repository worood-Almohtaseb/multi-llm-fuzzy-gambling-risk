#   python code/reviewer_checks.py
import sys, time, numpy as np, pandas as pd
src = open("code/experiments_1_3.py").read().split("\n")
ns = {"__name__": "ec"}
exec("\n".join(src[:146]), ns)                       
X, y, assessed_fn = ns["X"], ns["y"], ns["fuse"]
run, sp, auc_nh, auc_mh, yager_w = ns["run"], ns["sp"], ns["auc_nh"], ns["auc_mh"], ns["yager_w"]
judges = ns["judges"]; print(judges, X.shape)
out = {}

def pipelines(Xs):
    P = {"F1 + mean": run(Xs, "F1_mean", "mean"), "F2 + mean": run(Xs, "F2_median", "mean"),
         "F2 + crisp OWA": run(Xs, "F2_median", "owa"), "F4 + T1OWA Yager": run(Xs, "F4_fuzzy", "owa"),
         "F4 + T1OWA equal": run(Xs, "F4_fuzzy", "mean")}
    ns["PRESENT_THR"] = 0.5; P["F3 majority (0.5)"] = run(Xs, "F3_majority", "mean")
    ns["PRESENT_THR"] = 0.7; P["F3 majority (0.7)"] = run(Xs, "F3_majority", "mean"); ns["PRESENT_THR"] = 0.5
    return P

t0 = time.time(); S = pipelines(X); print("pipelines done", round(time.time() - t0), "s", flush=True)
base = {k: round(sp(v, y), 3) for k, v in S.items()}; print(base)

# ---- A. sample sensitivity: stratified bootstrap to the risk distribution of the full corpus ----
corpus = np.array([397, 493, 714, 496]); size = np.round(500 * corpus / corpus.sum()).astype(int); size[2] += 500 - size.sum()
rng = np.random.default_rng(42); idx = [np.where(y == h)[0] for h in range(4)]
B = np.array([np.concatenate([rng.choice(idx[h], size[h], replace=True) for h in range(4)]) for _ in range(2000)])
rows = []
for name, s in S.items():
    r = {"method": name}
    for mn, fn in (("spearman", sp), ("auc_none_vs_high", auc_nh)):
        v = np.array([fn(s[b], y[b]) for b in B]); r[mn + "_corpus_weighted"] = f"{v.mean():.3f} [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]"
        r[mn + "_sample"] = f"{fn(s, y):.3f}"
    rows.append(r)
pairs = [("F1 + mean", "F2 + crisp OWA"), ("F2 + mean", "F2 + crisp OWA"), ("F4 + T1OWA Yager", "F2 + crisp OWA"),
         ("F4 + T1OWA Yager", "F4 + T1OWA equal"), ("F3 majority (0.5)", "F2 + crisp OWA"), ("F3 majority (0.7)", "F2 + crisp OWA")]
prow = []
for a, b in pairs:
    d = np.array([sp(S[a][i], y[i]) - sp(S[b][i], y[i]) for i in B])
    prow.append({"comparison": f"{a} - {b}", "spearman_diff_corpus_weighted": f"{d.mean():+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]",
                 "spearman_diff_sample": f"{sp(S[a], y) - sp(S[b], y):+.3f}"})
out["A_corpus_weighted"] = pd.DataFrame(rows); out["A_paired_diff"] = pd.DataFrame(prow)
print(out["A_corpus_weighted"].to_string(index=False)); print(out["A_paired_diff"].to_string(index=False), flush=True)

# ---- B. off-scale scores of Qwen3 32B ----
scale = np.array([0, .1, .3, .5, .7, .9, 1.0]); q = judges.index("qwen3_32b")
v = X[:, :, q]; off = np.isfinite(v) & (np.abs(v[..., None] - scale).min(-1) > 1e-9)
print("off-scale Qwen3 values:", int(off.sum()), sorted(set(np.round(v[off], 3))), "users:", int(off.any(1).sum()))
Xa = X.copy(); Xa[:, :, q] = np.where(off, np.nan, v)                                   # dropped
Xb = X.copy(); Xb[:, :, q] = np.where(off, scale[np.abs(v[..., None] - scale).argmin(-1)], v)   # nearest level
Sa, Sb = pipelines(Xa), pipelines(Xb)
out["B_offscale"] = pd.DataFrame([{"method": k, "original": base[k], "off-scale dropped": round(sp(Sa[k], y), 3),
        "snapped to nearest level": round(sp(Sb[k], y), 3),
        "max_abs_change": round(max(abs(sp(Sa[k], y) - sp(S[k], y)), abs(sp(Sb[k], y) - sp(S[k], y))), 4)} for k in S])
print(out["B_offscale"].to_string(index=False), flush=True)

# ----  aggregated T1OWA sets: width against crisp OWA of the minimum and maximum; finer grid ----
a_, b_, c_ = [np.asarray(z) for z in assessed_fn(X, "F4_fuzzy")[0]]; asm = assessed_fn(X, "F4_fuzzy")[1]
agg = ns["aggregate_triangular_t1owa"]
def t1(grid, nal, weights_kind, dec):
    dom = np.round(np.linspace(0, 1, grid), dec); al = np.linspace(0, 1, nal); cen = np.zeros(len(y)); wid = np.zeros(len(y)); 
    for i in range(len(y)):
        m = asm[i]; P = []
        for k in np.where(m)[0]:
            bb = round(float(b_[i, k]), 2); aa = min(round(float(a_[i, k]), 2), bb); cc = max(round(float(c_[i, k]), 2), bb)
            if bb == 0.0 and cc - aa < 1e-3: cc = 1e-3
            P.append((aa, bb, cc, 1.0))
        w = None if weights_kind == "equal" else yager_w(len(P), 0.5)
        fs, _, ce = agg(P, domain_x=dom, alpha_levels=al, weights=w)
        sup = fs.domain[np.asarray(fs.mf_value) > 1e-9]
        cen[i] = ce if np.isfinite(ce) else np.mean([p[1] for p in P]); wid[i] = (sup.max() - sup.min()) if len(sup) else np.nan
    return cen, wid
t0 = time.time(); c1, w1 = t1(101, 50, "yager", 2); print("A3 base", round(time.time() - t0), "s", flush=True)
c1e, _ = t1(101, 50, "equal", 2)
c2, w2 = t1(201, 100, "yager", 3); c2e, _ = t1(201, 100, "equal", 3)
out["E_grid"] = pd.DataFrame([
    {"operator": "T1OWA Yager", "max_abs_diff_centroid": float(np.nanmax(np.abs(c1 - c2))), "mean_abs_diff": float(np.nanmean(np.abs(c1 - c2))),
     "spearman_between_grids": float(pd.Series(c1).corr(pd.Series(c2), method="spearman")), "spearman_GT1_101pts_50a": round(sp(c1, y), 4), "spearman_GT1_201pts_100a": round(sp(c2, y), 4)},
    {"operator": "T1OWA equal weights", "max_abs_diff_centroid": float(np.nanmax(np.abs(c1e - c2e))), "mean_abs_diff": float(np.nanmean(np.abs(c1e - c2e))),
     "spearman_between_grids": float(pd.Series(c1e).corr(pd.Series(c2e), method="spearman")), "spearman_GT1_101pts_50a": round(sp(c1e, y), 4), "spearman_GT1_201pts_100a": round(sp(c2e, y), 4)}])
print(out["E_grid"].to_string(index=False), flush=True)
# crisp OWA of the minima and the maxima (interval OWA with crisp weights)
lo = np.zeros(len(y)); hi = np.zeros(len(y))
for i in range(len(y)):
    m = asm[i]; w = yager_w(int(m.sum()), 0.5)
    lo[i] = np.dot(w, np.sort(a_[i][m])[::-1]); hi[i] = np.dot(w, np.sort(c_[i][m])[::-1])
cw = hi - lo; ok = np.isfinite(w1)
out["D_width"] = pd.DataFrame([{"corr(T1OWA support width, crisp OWA(max)-OWA(min))": float(np.corrcoef(w1[ok], cw[ok])[0, 1]),
    "max_abs_diff": float(np.max(np.abs(w1[ok] - cw[ok]))), "mean_abs_diff": float(np.mean(np.abs(w1[ok] - cw[ok]))),
    "spearman_width_GT1_T1OWA": round(sp(w1[ok], y[ok]), 3), "spearman_width_GT1_crisp": round(sp(cw[ok], y[ok]), 3),
    "spearman(centroid T1OWA, crisp OWA of medians)": float(pd.Series(c1).corr(pd.Series(S["F2 + crisp OWA"]), method="spearman"))}])
print(out["D_width"].T.to_string(header=False), flush=True)
with pd.ExcelWriter("results/reviewer_checks.xlsx") as w:
    for k, d in out.items(): d.to_excel(w, sheet_name=k, index=False)
print("saved results/reviewer_checks.xlsx")
