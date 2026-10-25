"""
Shared evaluation for all baselines: label coding and metrics with 95% bootstrap CIs (2,000 resamples, seed 42).
Metrics: Spearman, AUC (none vs high; none+low vs moderate+high), accuracy, macro precision/recall/F1, binary precision/recall/F1.
Imported by tfidf_svm_baseline.py, bert_baseline.py and zero_shot_llm_aggregate.py (keep this file in code/).
"""
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

LEVELS = ["none", "low", "moderate", "high"]
GT1_MAP = {l: i for i, l in enumerate(LEVELS)}
N_BOOT, SEED = 2000, 42

def auc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive).astype(bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def spear(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if np.isnan(r) else r

def auc1(score, y):                       # none vs high, middle classes dropped
    m = (y == 0) | (y == 3)
    return auc(np.asarray(score)[m], (y == 3)[m])

def auc2(score, y):                       # none+low vs moderate+high
    return auc(score, y >= 2)

def bin_prf(y, pred, task):
    """precision / recall / F1 of the positive class for the two binary tasks."""
    y, pred = np.asarray(y), np.asarray(pred)
    if task == 1:
        m = (y == 0) | (y == 3); yt, pt = (y[m] == 3), (pred[m] >= 2)   # a 'moderate' prediction counts as positive
    else:
        yt, pt = (y >= 2), (pred >= 2)
    p, r, f, _ = precision_recall_fscore_support(yt, pt, average="binary", zero_division=0)
    return p, r, f

def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return np.percentile(a, 2.5), np.percentile(a, 97.5)

def evaluate(name, y, pred=None, score=None, n_boot=N_BOOT):
    """y: int labels 0..3; pred: predicted class (0..3) or None; score: continuous score or None."""
    y = np.asarray(y, int); n = len(y)
    idx = np.random.default_rng(SEED).integers(0, n, size=(n_boot, n))
    row = {"method": name, "n": n}
    def put(key, fn):
        v = fn(np.arange(n)); bs = [fn(i) for i in idx]; lo, hi = ci(bs)
        row[key] = f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    if score is not None:
        s = np.asarray(score, float)
        put("spearman", lambda i: spear(s[i], y[i]))
        put("auc_none_vs_high", lambda i: auc1(s[i], y[i]))
        put("auc_nonelow_vs_modhigh", lambda i: auc2(s[i], y[i]))
    if pred is not None:
        p = np.asarray(pred, int)
        put("accuracy", lambda i: accuracy_score(y[i], p[i]))
        put("macro_F1", lambda i: f1_score(y[i], p[i], labels=[0, 1, 2, 3], average="macro", zero_division=0))
        put("macro_precision", lambda i: precision_recall_fscore_support(y[i], p[i], labels=[0, 1, 2, 3], average="macro", zero_division=0)[0])
        put("macro_recall", lambda i: precision_recall_fscore_support(y[i], p[i], labels=[0, 1, 2, 3], average="macro", zero_division=0)[1])
        for t, nm in ((1, "none_vs_high"), (2, "nonelow_vs_modhigh")):
            for k, lab in enumerate(("precision", "recall", "F1")):
                put(f"{nm}_{lab}", lambda i, t=t, k=k: bin_prf(y[i], p[i], t)[k])
    return row