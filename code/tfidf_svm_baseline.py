"""
TF-IDF + SVM baseline (supervised). 
Protocol: TEST = the 500 users scored by the LLM judges (read from t1owa_results.xlsx);
          TRAIN = every other user with a label. C is chosen by 5-fold CV on TRAIN only.
Outputs: baseline_tfidf_svm.csv (metrics with bootstrap CIs) and preds_tfidf_svm.csv (per-user predictions).
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from metrics_utils import GT1_MAP, evaluate

INPUT_FILE      = "data/user_texts_concatenated.csv"
PROFILES_FILE   = "data/user_labels_gt1.csv"
TEST_USERS_FILE = "data/test_users_500.xlsx"
OUTPUT_DIR      = "results"
SEED = 42

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_METRICS, OUT_PREDS = os.path.join(OUTPUT_DIR, "baseline_tfidf_svm.csv"), os.path.join(OUTPUT_DIR, "preds_tfidf_svm.csv")

def read_any(path):
    return pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)

raw = read_any(INPUT_FILE)
ucol = next(c for c in ("main_user", "User", "user", "user_id") if c in raw.columns)
raw[ucol] = raw[ucol].astype(str).str.strip()
txt = (raw.dropna(subset=["Monologue"]).groupby(ucol)["Monologue"]
         .apply(lambda s: " ".join(str(t).strip() for t in s)).reset_index().rename(columns={ucol: "User"}))

prof = read_any(PROFILES_FILE)
pcol = next(c for c in ("user_id", "main_user", "User", "user", "id") if c in prof.columns)
prof = prof[[pcol, "risk_level"]].rename(columns={pcol: "User"})
prof["User"] = prof["User"].astype(str).str.strip()
prof["y"] = prof["risk_level"].astype(str).str.strip().str.lower().map(GT1_MAP)

df = txt.merge(prof[["User", "y"]], on="User").dropna(subset=["y"])
df["y"] = df["y"].astype(int)

test_ids = set(pd.read_excel(TEST_USERS_FILE, index_col="main_user").index.astype(str).str.strip())
te = df[df["User"].isin(test_ids)].reset_index(drop=True)
tr = df[~df["User"].isin(test_ids)].reset_index(drop=True)
assert not (set(te["User"]) & set(tr["User"])), "train/test overlap"
print(f"train users: {len(tr)} | test users: {len(te)} | test labels: {te['y'].value_counts().sort_index().to_dict()}")

def pipe(model):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=20000)),
                     ("m", model)])
grid = {"m__C": [0.03, 0.1, 0.3, 1, 3, 10]}

clf = GridSearchCV(pipe(LinearSVC(class_weight="balanced", random_state=SEED, max_iter=20000)), grid,
                   scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
clf.fit(tr["Monologue"], tr["y"])
pred = clf.predict(te["Monologue"])

reg = GridSearchCV(pipe(LinearSVR(random_state=SEED, max_iter=20000)), grid,
                   scoring="neg_mean_absolute_error", cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
reg.fit(tr["Monologue"], tr["y"].astype(float))
score = reg.predict(te["Monologue"])
print("best C  classifier:", clf.best_params_, "| regressor:", reg.best_params_)

row = evaluate("TF-IDF + SVM", te["y"].values, pred=pred, score=score)
pd.DataFrame([row]).to_csv(OUT_METRICS, index=False)
pd.DataFrame({"User": te["User"], "y": te["y"], "svm_class": pred, "svr_score": score}).to_csv(OUT_PREDS, index=False)
print(pd.Series(row).to_string())
