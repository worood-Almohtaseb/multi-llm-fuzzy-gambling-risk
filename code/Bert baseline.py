"""
BERT / RoBERTa fine-tuning baseline (supervised), same protocol as TF-IDF + SVM:
  TEST  = the 500 users scored by the LLM judges (index main_user of t1owa_results.xlsx)
  TRAIN = all other users; 10% of them (stratified) are held out as VALIDATION to choose the best epoch.
The test users are never used for training, model selection or early stopping.
Each user's monologues are joined into one text (about 400-550 tokens, so max_length = 512 covers almost all users).
Outputs: baseline_bert.csv (metrics with bootstrap CIs) and preds_bert.csv (per-user class, probabilities, score).

Install once:  pip install torch transformers scikit-learn scipy pandas openpyxl
Run:           python bert_baseline.py
"""
import os, random, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from metrics_utils import GT1_MAP, evaluate

INPUT_FILE      = "data/user_texts_concatenated.csv"
PROFILES_FILE   = "data/user_labels_gt1.csv"
TEST_USERS_FILE = "data/test_users_500.xlsx"
OUTPUT_DIR      = "results"

MODEL_NAME = "roberta-base"
MAX_LEN    = 512
EPOCHS     = 4
BATCH      = 16
LR         = 2e-5
WARMUP     = 0.1
VAL_FRAC   = 0.10
SEED       = 42
FORCE_DEVICE = None
LIMIT      = None

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
device = FORCE_DEVICE or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"device: {device} | model: {MODEL_NAME}")
if device == "cpu":
    print("WARNING: no GPU found - training will be slow (use the server with the GPU).")

os.makedirs(OUTPUT_DIR, exist_ok=True)
tag = MODEL_NAME.replace("/", "_")
OUT_METRICS = os.path.join(OUTPUT_DIR, f"baseline_{tag}.csv")
OUT_PREDS   = os.path.join(OUTPUT_DIR, f"preds_{tag}.csv")

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
df = txt.merge(prof[["User", "y"]], on="User").dropna(subset=["y"]); df["y"] = df["y"].astype(int)

test_ids = set(pd.read_excel(TEST_USERS_FILE, index_col="main_user").index.astype(str).str.strip())
te = df[df["User"].isin(test_ids)].reset_index(drop=True)
trall = df[~df["User"].isin(test_ids)].reset_index(drop=True)
assert not (set(te["User"]) & set(trall["User"])), "train/test overlap"
if LIMIT:
    trall = trall.sample(LIMIT, random_state=SEED).reset_index(drop=True)
tr, va = train_test_split(trall, test_size=VAL_FRAC, stratify=trall["y"], random_state=SEED)
tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
print(f"train: {len(tr)} | validation: {len(va)} | test: {len(te)} | test labels: {te['y'].value_counts().sort_index().to_dict()}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def encode(frame):
    enc = tok(frame["Monologue"].tolist(), truncation=True, max_length=MAX_LEN, padding=False)
    return enc
def n_truncated(frame):
    lens = [len(x) for x in tok(frame["Monologue"].tolist(), truncation=False)["input_ids"]]
    return sum(l > MAX_LEN for l in lens), int(np.max(lens))
nt, mx = n_truncated(te)
print(f"test texts longer than {MAX_LEN} tokens: {nt} (longest {mx} tokens)")

def make_loader(frame, shuffle):
    enc = encode(frame)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(frame["y"].iloc[i])}
             for i in range(len(frame))]
    def collate(b):
        out = tok.pad([{k: x[k] for k in ("input_ids", "attention_mask")} for x in b], return_tensors="pt")
        out["labels"] = torch.tensor([x["labels"] for x in b])
        return out
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(items, batch_size=BATCH, shuffle=shuffle, collate_fn=collate, generator=g if shuffle else None)

train_dl, val_dl, test_dl = make_loader(tr, True), make_loader(va, False), make_loader(te, False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4, attn_implementation="eager").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, foreach=False)
total = EPOCHS * len(train_dl)
sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total), total)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

@torch.no_grad()
def predict(dl):
    model.eval(); P = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(**b).logits
        P.append(torch.softmax(logits.float(), -1).cpu().numpy())
    P = np.vstack(P)
    if not np.isfinite(P).all():
        raise RuntimeError("NaN in model outputs - stop and check the device / attention setting.")
    return P

best_f1, best_state, n_skipped = -1, None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run = 0
    for step, b in enumerate(train_dl, start=1):
        b = {k: v.to(device) for k, v in b.items()}
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(**b).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {ep}, step {step}: {loss.item()}")
        if ep == 1 and step <= 5:
            print(f"  step {step}: loss {loss.item():.4f}", flush=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        if ep == 1 and step <= 5:
            print(f"  step {step}: grad norm {float(gn):.3f}", flush=True)
        if not torch.isfinite(gn):
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"  WARNING: non-finite gradient at epoch {ep}, step {step}; skipped. First bad params: {bad[:3]}", flush=True)
            opt.zero_grad(); sched.step(); n_skipped += 1
            if n_skipped >= 5:
                raise RuntimeError("Too many non-finite gradients. Try FORCE_DEVICE = 'cpu' or run on the server GPU.")
            continue
        scaler.step(opt); scaler.update(); sched.step(); run += loss.item()
    pv = predict(val_dl)
    f1v = f1_score(va["y"], pv.argmax(1), average="macro", labels=[0, 1, 2, 3], zero_division=0)
    print(f"epoch {ep}/{EPOCHS} | train loss {run / len(train_dl):.4f} | val macro-F1 {f1v:.3f} | {time.time() - t0:.0f}s")
    if f1v > best_f1:
        best_f1 = f1v; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
print(f"best validation macro-F1: {best_f1:.3f}")

proba = predict(test_dl)
pred = proba.argmax(1)
score = proba @ np.arange(4)
row = evaluate(f"{MODEL_NAME} (fine-tuned)", te["y"].values, pred=pred, score=score)
row["best_val_macro_F1"] = round(best_f1, 3); row["epochs"] = EPOCHS; row["lr"] = LR; row["max_len"] = MAX_LEN
pd.DataFrame([row]).to_csv(OUT_METRICS, index=False)
out = pd.DataFrame({"User": te["User"], "y": te["y"], "bert_class": pred, "bert_score": score})
for i, l in enumerate(["none", "low", "moderate", "high"]):
    out[f"p_{l}"] = proba[:, i]
out.to_csv(OUT_PREDS, index=False)
print(pd.Series(row).to_string())
print(f"\nSaved -> {OUT_METRICS}\n         {OUT_PREDS}")