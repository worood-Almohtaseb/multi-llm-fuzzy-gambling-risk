# Zero-shot LLM baseline: evaluates the direct risk-level answers of the five LLMs
# (data/zero_shot/direct_<model>.xlsx, written by zero_shot_llm_run.py) against the GT1 labels.
# It scores each model on its own and four combinations of the five answers
# (majority vote, median, mean and Yager OWA; mean and OWA use cross-validated cut-points).
# Output: results/baseline_llm_zeroshot_all.xlsx (sheet "metrics" with CIs, sheet "predictions" per user).
import os, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from metrics_utils import GT1_MAP, evaluate

IN_DIR = "data/zero_shot"
SHOT_LABEL = "zero-shot"
MODELS = {"llama370": "Llama 3.1 70B", "llama3": "Llama 3.3 70B", "deepseek": "deepseek-r1:70b",
          "qwen3": "Qwen3 32B", "gemma3": "Gemma3 27B"}
MIN_VALID = 3
YAGER_ALPHA = 0.5
N_SPLITS, SEED = 5, 42
OUT_FILE = "results/baseline_llm_zeroshot_all.xlsx"

os.makedirs("results", exist_ok=True)
P = None
for key in MODELS:
    path = os.path.join(IN_DIR, f"direct_{key}.xlsx")
    if not os.path.exists(path):
        print(f"missing (skipped): {path}"); continue
    d = pd.read_excel(path)[["User", "true_level", f"{key}_risk_level"]]
    d["User"] = d["User"].astype(str).str.strip()
    P = d if P is None else P.merge(d.drop(columns="true_level"), on="User", how="outer")
keys = [k for k in MODELS if f"{k}_risk_level" in P.columns]
print("models found:", keys)
P["y"] = P["true_level"].astype(str).str.strip().str.lower().map(GT1_MAP)
P = P.dropna(subset=["y"]).reset_index(drop=True); P["y"] = P["y"].astype(int)
num = pd.DataFrame({k: P[f"{k}_risk_level"].astype(str).str.strip().str.lower().map(GT1_MAP) for k in keys})
y = P["y"].values

def yager_w(n, a=YAGER_ALPHA):
    return np.array([(k / n) ** a - ((k - 1) / n) ** a for k in range(1, n + 1)])

def agg_row(v):
    v = np.array([x for x in v if not np.isnan(x)], float)
    if len(v) < MIN_VALID:
        return pd.Series({"vote": np.nan, "median": np.nan, "mean": np.nan, "yager": np.nan})
    c = Counter(v); top = max(c.values())
    vote = int(np.ceil(np.median([k for k, n in c.items() if n == top])))
    return pd.Series({"vote": vote, "median": int(np.ceil(np.median(v))), "mean": v.mean(),
                      "yager": float(np.dot(yager_w(len(v)), np.sort(v)[::-1]))})
A = num.apply(agg_row, axis=1)

def youden_cut(s, pos):
    fpr, tpr, thr = roc_curve(pos, s); j = tpr - fpr
    return thr[np.argmax(j[1:]) + 1]
def cv_classes(s, yy):
    out = np.full(len(s), -1)
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(s, yy):
        cuts = np.sort([youden_cut(s[tr], yy[tr] >= k) for k in (1, 2, 3)])
        out[te] = (s[te][:, None] >= cuts[None, :]).sum(axis=1)
    return out

rows, preds = [], P[["User", "y"]].copy()
for k in keys:
    ok = num[k].notna().values
    r = evaluate(f"{MODELS[k]} ({SHOT_LABEL})", y[ok], pred=num[k][ok].astype(int).values, score=num[k][ok].values)
    r["n_error"] = int((~ok).sum()); rows.append(r); preds[k] = num[k]

def add(name, score, cls):
    ok = ~np.isnan(score)
    r = evaluate(name, y[ok], pred=cls[ok].astype(int), score=score[ok]); r["n_error"] = int((~ok).sum()); rows.append(r)

ok = A["vote"].notna().values
cls = A["vote"].values;   add(f"Majority vote ({SHOT_LABEL})", np.where(ok, A["mean"].values, np.nan), cls)
cls = A["median"].values; add(f"Median of models ({SHOT_LABEL})", A["median"].values, cls)
for col, name in (("mean", "Mean of models"), ("yager", "Yager OWA of models")):
    s = A[col].values; okk = ~np.isnan(s)
    c = np.full(len(s), np.nan); c[okk] = cv_classes(s[okk], y[okk])
    add(f"{name} + CV cut-points ({SHOT_LABEL})", s, c)
    preds[col + "_score"] = s; preds[col + "_class"] = c
preds["vote_class"] = A["vote"]; preds["median_class"] = A["median"]

with pd.ExcelWriter(OUT_FILE) as w:
    pd.DataFrame(rows).to_excel(w, sheet_name="metrics", index=False)
    preds.to_excel(w, sheet_name="predictions", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).set_index("method")[["n", "n_error", "spearman", "accuracy", "macro_F1", "macro_precision", "macro_recall",
      "nonelow_vs_modhigh_precision", "nonelow_vs_modhigh_recall", "nonelow_vs_modhigh_F1"]].to_string())
print(f"\nYager weights (n=5): {np.round(yager_w(5), 3)}")
print(f"Saved -> {OUT_FILE}")