import os
import numpy as np
import pandas as pd

LABELS_FILE = "data/user_labels_gt1.xlsx"
SCORES_FILE = "data/judge_scores_nine_criteria.xlsx"   
TEXTS_FILE = "data/user_texts_concatenated.csv"
OUT_DIR = "annotation"
PER_LEVEL = 25         
N_PRACTICE = 0          
SEED = 42
LEVELS = ["none", "low", "moderate", "high"]

CRITERIA = {
    "A1": "Tolerance (needs to gamble with increasing amounts)",
    "A2": "Withdrawal / irritability when trying to cut down",
    "A3": "Loss of control (repeated unsuccessful attempts to stop or reduce)",
    "A4": "Preoccupation (constant thoughts about gambling)",
    "A5": "Escape / mood relief (gambles when distressed)",
    "A6": "Chasing losses (returns to win back money lost)",
    "A7": "Secrecy / lying (hides or lies about gambling)",
    "A8": "Jeopardized relationships / work (risks job, relationships, studies)",
    "A9": "Bailout / financial rescue (relies on others to pay gambling debts)",
    "OVERALL": "OVERALL gambling-risk level of this user as a whole (score AFTER A1-A9): 0 none, 1 low, 2 moderate, 3 high",
}
SCALE = [
    (0, "Absent", "Nothing in the text supports it, or the text denies it."),
    (1, "Weak or possible", "Only a hint, or only gambling in general is described."),
    (2, "Clear", "The text clearly describes the criterion once or twice."),
    (3, "Strong", "Explicit and repeated, or severe."),
]
# For the OVERALL row the same 0-3 values mean: 0 none, 1 low, 2 moderate, 3 high risk.
RULES = [
    "One row per user. Read the text, then fill A1-A9 and then OVERALL (0-3).",
    "OVERALL = one holistic risk level for the whole text (0 none, 1 low, 2 moderate, 3 high). Do not just count criteria. Score it last and do not go back to change A1-A9.",
    "Work alone. Do not look at any model output or at the other annotator's scores.",
    "Follow the order of the column 'order'. Practice users (part = practice) are scored together in the calibration round.",
    "Negation is not evidence ('I never hide my gambling' = 0 for A7).",
    "Generic gambling talk is not evidence for any criterion.",
    "Score each criterion independently of the others.",
    "Score only what the text says or clearly implies.",
    "If you hesitate between two scores, take the lower one and write the criterion in the column 'unsure' (e.g. A3, A6).",
    "Use the column 'note' only when something needs explaining (optional).",
    "Do not change a score after seeing the other annotator's scores.",
]

os.makedirs(OUT_DIR, exist_ok=True)
lab = pd.read_excel(LABELS_FILE)
idc = [c for c in lab.columns if c.lower() in ("user_id", "main_user", "user", "id")][0]
lab["user"] = lab[idc].astype(str).str.strip()
lab["gt1"] = lab["risk_level"].astype(str).str.strip().str.lower()
lab = lab[lab["gt1"].isin(LEVELS)].drop_duplicates("user").reset_index(drop=True)
# population = the users that were scored by the judges (judge scores file); texts and labels are matched by user id
_s = pd.read_excel(SCORES_FILE)
_sid = [c for c in _s.columns if str(c).lower() in ("user_id", "main_user", "user", "id")]
scored = list(dict.fromkeys(_s[_sid[0] if _sid else _s.columns[0]].astype(str).str.strip()))
print(f"scored users (judge scores file): {len(scored)} | {scored[0]} ... {scored[-1]}")
_t = pd.read_csv(TEXTS_FILE)
_tid = [c for c in _t.columns if str(c).lower() in ("user_id", "main_user", "user", "id")][0]
no_text = set(scored) - set(_t[_tid].astype(str).str.strip())
no_label = set(scored) - set(lab["user"])
assert not no_text, f"{len(no_text)} scored users have no text, e.g. {sorted(no_text)[:5]}"
assert not no_label, f"{len(no_label)} scored users have no GT1 label, e.g. {sorted(no_label)[:5]}"
lab = lab[lab["user"].isin(scored)].reset_index(drop=True)
print("users:", len(lab), lab["gt1"].value_counts().reindex(LEVELS).to_dict())

rng = np.random.default_rng(SEED)
sample = []
for lv in LEVELS:
    pool = lab[lab["gt1"] == lv]["user"].tolist()
    assert len(pool) >= PER_LEVEL, f"not enough users in {lv}"
    sample += [(u, lv, len(pool)) for u in rng.choice(pool, PER_LEVEL, replace=False)]
S = pd.DataFrame(sample, columns=["user", "gt1", "stratum_size"])
S["n_sampled"] = PER_LEVEL
S["weight"] = S["stratum_size"] / S["n_sampled"]       # users represented by each sampled user

rest = lab[~lab["user"].isin(S["user"])]["user"].tolist()
practice = list(rng.choice(rest, N_PRACTICE, replace=False))

order = list(rng.permutation(len(S)))                  # blind order (mixes the GT1 levels)
S["order"] = np.argsort(order) + 1
S = S.sort_values("order").reset_index(drop=True)

# representativeness helper: GT1 counts of sample vs population
chk = pd.DataFrame({"population": lab["gt1"].value_counts().reindex(LEVELS),
                    "sampled": S["gt1"].value_counts().reindex(LEVELS)})
chk["weight_per_user"] = chk["population"] / chk["sampled"]

with pd.ExcelWriter(f"{OUT_DIR}/sample_private.xlsx") as w:
    S.to_excel(w, sheet_name="sample", index=False)
    pd.DataFrame({"user": practice}).to_excel(w, sheet_name="practice_users", index=False)
    chk.to_excel(w, sheet_name="strata")
    pd.DataFrame({"parameter": ["seed", "per_level", "n_practice"], "value": [SEED, PER_LEVEL, N_PRACTICE]}).to_excel(
        w, sheet_name="settings", index=False)

# texts (optional)
texts = None
if os.path.exists(TEXTS_FILE):
    t = pd.read_csv(TEXTS_FILE)
    print("columns in texts file:", list(t.columns))
    tid = [c for c in t.columns if c.lower() in ("user_id", "main_user", "user", "id")][0]
    # text column = the string column with the longest average length (works for any column name)
    cand = [c for c in t.columns if c != tid and not pd.api.types.is_numeric_dtype(t[c])]
    tx = max(cand, key=lambda c: t[c].astype(str).str.len().mean())
    print("text column used:", tx)
    t["user"] = t[tid].astype(str).str.strip()
    rows_per_user = t.groupby("user").size()
    print("rows per user in texts file: min", rows_per_user.min(), "max", rows_per_user.max())
    if rows_per_user.max() > 1:   # several rows per user (e.g. one per topic): join them in file order
        topic = [c for c in t.columns if c not in (tid, tx, "user") and not pd.api.types.is_numeric_dtype(t[c])]
        if topic:
            t[tx] = "[" + t[topic[0]].astype(str) + "] " + t[tx].astype(str)
        texts = t.groupby("user", sort=False)[tx].apply(lambda x: "\n\n".join(x.astype(str)))
    else:
        texts = t.set_index("user")[tx].astype(str)
else:
    print("WARNING: texts file not found; text column left empty (add it before sending).")


def build(path, who):
    users = [(u, "practice", 0) for u in practice] + [(r.user, "main", r.order) for r in S.itertuples()]
    codes = list(CRITERIA)                      # A1..A9, OVERALL
    df = pd.DataFrame({
        "order": [o for _, _, o in users], "part": [p for _, p, _ in users], "user": [u for u, _, _ in users],
        "text": [texts.get(u, "") if texts is not None else "" for u, _, _ in users]})
    for c in codes:
        df[c] = None
    df["unsure"] = ""
    df["note"] = ""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"Annotation task": [who]}).to_excel(w, sheet_name="instructions", index=False)
        ws = w.sheets["instructions"]
        r = 4
        ws.cell(r, 1, "Criteria"); r += 1
        for k, d in CRITERIA.items():
            ws.cell(r, 1, k); ws.cell(r, 2, d); r += 1
        r += 1; ws.cell(r, 1, "Scale (A1-A9)"); r += 1
        for sc_, n, d in SCALE:
            ws.cell(r, 1, sc_); ws.cell(r, 2, n); ws.cell(r, 3, d); r += 1
        ws.cell(r, 1, "OVERALL"); ws.cell(r, 2, "0 none, 1 low, 2 moderate, 3 high risk (whole text)"); r += 2
        ws.cell(r, 1, "Rules"); r += 1
        for i, x in enumerate(RULES, 1):
            ws.cell(r, 1, i); ws.cell(r, 2, x); r += 1
        ws.column_dimensions["A"].width = 14; ws.column_dimensions["B"].width = 70; ws.column_dimensions["C"].width = 70
        df.to_excel(w, sheet_name="scoring", index=False)
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.styles import Alignment
        from openpyxl.utils import get_column_letter as L
        sc = w.sheets["scoring"]
        n = len(df) + 1
        dv = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True)
        sc.add_data_validation(dv)
        dv.add(f"E2:{L(4 + len(codes))}{n}")
        widths = [7, 10, 9, 90] + [6] * 9 + [9, 14, 30]
        for i, wd in enumerate(widths, 1):
            sc.column_dimensions[L(i)].width = wd
        for row in sc.iter_rows(min_row=2, min_col=4, max_col=4):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
        for row in sc.iter_rows(min_row=2, min_col=5, max_col=len(df.columns)):
            for c in row:
                c.alignment = Alignment(vertical="top")
        sc.freeze_panes = "D2"


build(f"{OUT_DIR}/annotator_1.xlsx", "Annotator 1")
build(f"{OUT_DIR}/annotator_2.xlsx", "Annotator 2")
print(chk.to_string())
print("saved in", OUT_DIR, "| first 5 sampled:", S["user"].head(5).tolist(), "| practice:", practice)