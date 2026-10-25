# Judge scoring script: one local LLM judge (run through Ollama) scores the nine DSM-5 criteria (0-1) with an evidence span for every monologue.
# Set MODEL_KEY and MODEL_NAME for the judge, then run once per judge from the repository root:  python code/judges/score_with_llm_judge.py
# Output: one Excel file per judge in data/judge_scores_with_evidence_nine_criteria/.


import pandas as pd
import ollama
import json
import os
import re
import time
import zipfile
import hashlib
from collections import defaultdict
from datetime import datetime

MODEL_KEY  = "llama370"           #  llama3 | gemma3 | deepseek | qwen3 | llama370
MODEL_NAME = "llama3.1:70b"    # | llama3.3:70b , gemma3:27b, deepseek-r1:70b, qwen3:32b ,llama3.1:70b

# input: one monologue per row (columns main_user, Monologue)
INPUT_FILE  = "data/user_texts_concatenated.csv"
OUTPUT_FILE = f"data/judge_scores_with_evidence_nine_criteria/{MODEL_KEY}.xlsx"


# crash safety: journal (one line per finished row), temp file and backup of the Excel output
JOURNAL_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_journal.jsonl"

TMP_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_tmp.xlsx"

BAK_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_backup.xlsx"

# number of rows to score; None = all rows
MAX_ROWS = None

# deterministic decoding
TEMPERATURE = 0.0
OLLAMA_SEED = 42


CHECKPOINT_EVERY = 5
SLEEP_BETWEEN_CALLS = 0.0
MAX_POST_LENGTH = 2000
MAX_RETRIES = 3


# the nine criterion definitions shown to the judge
CRITERIA = {
    "A1": "Tolerance: needing increasing amounts of gambling money or intensity to obtain the desired excitement.",
    "A2": "Withdrawal/Irritability: restlessness or irritability when attempting to cut down or stop gambling.",
    "A3": "Loss of Control: repeated unsuccessful efforts to control, cut back, or stop gambling.",
    "A4": "Preoccupation: persistent gambling-related thoughts, reliving past gambling, planning future gambling, or thinking about obtaining money to gamble.",
    "A5": "Escape/Mood Relief: gambling when distressed, such as when feeling helpless, guilty, anxious, or depressed.",
    "A6": "Chasing Losses: after losing money gambling, returning later to try to win back or get even for the losses.",
    "A7": "Secrecy/Lying: lying or concealing the extent of gambling involvement.",
    "A8": "Jeopardized Relationships/Work: jeopardizing or losing a significant relationship, job, educational opportunity, or career opportunity because of gambling.",
    "A9": "Bailout/Financial Rescue: relying on other people to provide money to relieve serious financial problems caused by gambling."
}

SCORE_COLUMNS = [f"{MODEL_KEY}_{c}_score" for c in CRITERIA]
EVIDENCE_COLUMNS = [f"{MODEL_KEY}_{c}_evidence" for c in CRITERIA]

STATUS_COL = f"{MODEL_KEY}_all9_status"
ERROR_COL = f"{MODEL_KEY}_all9_error"
HASH_COL = "_row_hash"
PILOT_COL = "_all9_pilot_selected"


def make_row_hash(row):
    key = "||".join([
        str(row.get("main_user", "")),
        str(row.get("Monologue", ""))
    ])
    return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()

# judge prompt: scoring scale, rules and required JSON format
def build_prompt(post):
    criteria_block = "\n".join([f"{k}: {v}" for k, v in CRITERIA.items()])

    return f"""
You are evaluating criterion-level linguistic evidence in ONE synthetic social-media monologue.

IMPORTANT RULES:
- This is NOT a clinical diagnosis.
- Evaluate only what is explicitly stated or reasonably implied by THIS text.
- Do not assume a criterion merely because the post concerns gambling.
- Do not invent missing history, repeated behavior, failed quit attempts,
  relationship consequences, or financial rescue.
- Distinguish closely related criteria carefully.
- Negated behavior is NOT positive evidence.
- Generic gambling, losing money, distress, excitement, regret, or betting alone
  is insufficient unless it specifically supports the criterion.
- Each criterion must be scored independently.
- A criterion may correctly receive 0.00.
- Evaluate ALL NINE criteria, whether or not they were present in any previous mapping.

CRITERIA:
{criteria_block}

SCORING SCALE:
0.00 = no evidence
0.10 = extremely weak / highly ambiguous cue
0.30 = weak but plausible evidence
0.50 = moderate evidence
0.70 = strong evidence
0.90 = very strong / explicit evidence
1.00 = exceptionally explicit and unambiguous evidence

For each criterion return:
1. "score": a number between 0.00 and 1.00.
2. "evidence": an EXACT short span copied from the monologue, preferably 12 words
   or fewer, that best supports the score. Use null when score is 0.00.
   Do not provide explanations or chain-of-thought.

MONOLOGUE:
{post}

Return ONLY valid JSON with exactly these nine keys:
{{
  "A1": {{"score": 0.00, "evidence": null}},
  "A2": {{"score": 0.00, "evidence": null}},
  "A3": {{"score": 0.00, "evidence": null}},
  "A4": {{"score": 0.00, "evidence": null}},
  "A5": {{"score": 0.00, "evidence": null}},
  "A6": {{"score": 0.00, "evidence": null}},
  "A7": {{"score": 0.00, "evidence": null}},
  "A8": {{"score": 0.00, "evidence": null}},
  "A9": {{"score": 0.00, "evidence": null}}
}}
""".strip()

# parse the JSON answer, recovering the outer object if extra text is present
def extract_json(text):
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))

# require all nine criteria and scores in [0, 1]; invalid answers count as errors, never as 0
def validate_result(obj):
    if not isinstance(obj, dict):
        raise ValueError("Top-level response is not a JSON object.")

    missing = [c for c in CRITERIA if c not in obj]
    if missing:
        raise ValueError(f"Missing criteria: {missing}")

    out = {}

    for criterion in CRITERIA:
        item = obj[criterion]

        if not isinstance(item, dict):
            raise ValueError(f"{criterion} is not an object.")

        if "score" not in item:
            raise ValueError(f"Missing score for {criterion}")

        score = float(item["score"])

        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Out-of-range score for {criterion}: {score}")

        evidence = item.get("evidence", None)

        if evidence is None:
            evidence_clean = None
        else:
            evidence_clean = str(evidence).strip()
            if not evidence_clean:
                evidence_clean = None
            elif len(evidence_clean) > 300:
                evidence_clean = evidence_clean[:300]

        if score == 0.0:
            evidence_clean = None

        out[criterion] = {
            "score": round(score, 4),
            "evidence": evidence_clean
        }

    return out

# call the judge with up to MAX_RETRIES attempts
def call_model(post):
    prompt = build_prompt(post)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": TEMPERATURE,
                    "seed": OLLAMA_SEED,
                    "num_ctx": 4096
                }
            )

            try:
                content = response["message"]["content"]
            except Exception:
                content = response.message.content

            parsed = extract_json(content)
            return validate_result(parsed), None

        except Exception as e:
            last_error = (
                f"attempt {attempt}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))

    return None, last_error


def is_valid_xlsx(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False

# safe save: write temp file, keep backup, then replace the output
def save_checkpoint(df):
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_excel(TMP_FILE, index=False, engine="openpyxl")

    if not zipfile.is_zipfile(TMP_FILE):
        print("  WARNING: temp save looks corrupt; keeping previous output.")
        return

    for attempt in range(1, 6):
        try:
            if os.path.exists(OUTPUT_FILE):
                os.replace(OUTPUT_FILE, BAK_FILE)
            os.replace(TMP_FILE, OUTPUT_FILE)
            return
        except PermissionError:

            print(f"  output file is locked (open in Excel?) - retry {attempt}/5")
            time.sleep(3)

    print(
        f"  WARNING: could not replace {OUTPUT_FILE}. Latest data is safe in "
        f"{TMP_FILE} and in the journal; it will be picked up on resume."
    )

# append one finished row to the journal and flush it to disk
def journal_append(row_hash, status, result=None, error=None):
    record = {
        "hash": row_hash,
        "model": MODEL_KEY,
        "status": status,
        "result": result,
        "error": error,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# find the newest readable earlier output for resuming
def load_previous_output():
    candidates = [OUTPUT_FILE, TMP_FILE, BAK_FILE]
    candidates = [p for p in candidates if os.path.exists(p)]
    candidates.sort(key=os.path.getmtime, reverse=True)

    for path in candidates:
        if not is_valid_xlsx(path):
            print(f"  skipping damaged file: {path}")
            continue
        try:
            old = pd.read_excel(path)
        except Exception as e:
            print(f"  could not read {path}: {type(e).__name__}: {e}")
            continue
        if HASH_COL not in old.columns:
            print(f"  skipping {path}: no stable row hash column")
            continue
        print(f"Resuming from: {path}")
        return old

    return None

# restore rows finished after the last checkpoint
def apply_journal(df):
    if not os.path.exists(JOURNAL_FILE):
        return 0

    hash_to_idx = defaultdict(list)
    for idx, h in df[HASH_COL].items():
        hash_to_idx[h].append(idx)

    restored = 0
    with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:

                continue
            if rec.get("model") != MODEL_KEY:
                continue

            for idx in hash_to_idx.get(rec.get("hash"), []):
                current = str(df.at[idx, STATUS_COL]).strip().upper()

                if rec["status"] == "OK" and rec.get("result"):
                    for c in CRITERIA:
                        r = rec["result"].get(c, {})
                        df.at[idx, f"{MODEL_KEY}_{c}_score"] = r.get("score")
                        df.at[idx, f"{MODEL_KEY}_{c}_evidence"] = r.get("evidence")
                    df.at[idx, STATUS_COL] = "OK"
                    df.at[idx, ERROR_COL] = pd.NA
                    df.at[idx, PILOT_COL] = True
                    if current != "OK":
                        restored += 1
                elif current != "OK":

                    df.at[idx, STATUS_COL] = "ERROR"
                    df.at[idx, ERROR_COL] = rec.get("error")

    return restored

# choose the rows to score (first MAX_ROWS rows, or all)
def choose_and_freeze_pilot(df):
    all_idx = df.index.tolist()
    existing = df[PILOT_COL].fillna(False).astype(bool)
    existing_idx = df.index[existing].tolist()

    if MAX_ROWS is None:
        df.loc[all_idx, PILOT_COL] = True
        return all_idx

    if len(existing_idx) >= MAX_ROWS:
        return existing_idx

    if len(all_idx) <= MAX_ROWS:
        df.loc[all_idx, PILOT_COL] = True
        return all_idx

    selected = all_idx[:MAX_ROWS]
    df.loc[selected, PILOT_COL] = True
    return selected


# load input and create result columns
print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

required_cols = {"Monologue"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df[HASH_COL] = df.apply(make_row_hash, axis=1)

for col in SCORE_COLUMNS:
    if col not in df.columns:
        df[col] = pd.NA

for col in EVIDENCE_COLUMNS:
    if col not in df.columns:
        df[col] = pd.NA

if STATUS_COL not in df.columns:
    df[STATUS_COL] = "NOT_EVALUATED"

if ERROR_COL not in df.columns:
    df[ERROR_COL] = pd.NA

if PILOT_COL not in df.columns:
    df[PILOT_COL] = False


# resume from earlier output and journal
old = load_previous_output()

if old is not None:
    old = old.drop_duplicates(subset=[HASH_COL], keep="last").set_index(HASH_COL)

    copy_cols = (
        SCORE_COLUMNS
        + EVIDENCE_COLUMNS
        + [STATUS_COL, ERROR_COL, PILOT_COL]
    )

    for idx, row_hash in df[HASH_COL].items():
        if row_hash in old.index:
            for col in copy_cols:
                if col in old.columns:
                    df.at[idx, col] = old.at[row_hash, col]
elif any(os.path.exists(p) for p in [OUTPUT_FILE, TMP_FILE]):
    print("WARNING: previous output exists but none was readable; relying on journal.")

restored = apply_journal(df)
if restored:
    print(f"Restored {restored:,} finished rows from journal: {JOURNAL_FILE}")


selected_rows = choose_and_freeze_pilot(df)

save_checkpoint(df)

eligible = [
    idx for idx in selected_rows
    if str(df.at[idx, STATUS_COL]).strip().upper() != "OK"
]

already_done = len(selected_rows) - len(eligible)

print(f"Rows selected: {len(selected_rows):,}")
print(f"Rows already done: {already_done:,}")
print(f"Rows still needing evaluation: {len(eligible):,}")
if eligible:
    print(f"Resuming at row index: {eligible[0]} (Excel row {eligible[0] + 2})")
print(f"Model: {MODEL_NAME}")
print("Scoring all 9 criteria independently in one model call.\n")


# main loop: score each remaining row, journal it, checkpoint every CHECKPOINT_EVERY rows
completed_this_run = 0

try:
    for pos, idx in enumerate(eligible, start=1):
        raw_post = df.at[idx, "Monologue"]
        row_hash = df.at[idx, HASH_COL]

        if pd.isna(raw_post) or not str(raw_post).strip():
            df.at[idx, STATUS_COL] = "ERROR"
            df.at[idx, ERROR_COL] = "Empty Monologue"
            journal_append(row_hash, "ERROR", error="Empty main_post")
            continue

        post = str(raw_post)[:MAX_POST_LENGTH]

        print(f"[{already_done + pos:,}/{len(selected_rows):,}] row={idx}")

        result, error = call_model(post)

        if result is None:
            df.at[idx, STATUS_COL] = "ERROR"
            df.at[idx, ERROR_COL] = error
            journal_append(row_hash, "ERROR", error=error)
            print(f"  ERROR: {error}")
        else:
            for criterion in CRITERIA:
                df.at[idx, f"{MODEL_KEY}_{criterion}_score"] = result[criterion]["score"]
                df.at[idx, f"{MODEL_KEY}_{criterion}_evidence"] = result[criterion]["evidence"]

            df.at[idx, STATUS_COL] = "OK"
            df.at[idx, ERROR_COL] = pd.NA

            journal_append(row_hash, "OK", result=result)
            completed_this_run += 1

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

        if pos % CHECKPOINT_EVERY == 0:
            save_checkpoint(df)
            print(
                f"  checkpoint saved at "
                f"{datetime.now().isoformat(timespec='seconds')}"
            )

except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C) - saving progress...")
    raise

finally:

    save_checkpoint(df)
    print(f"Final save done. Journal: {JOURNAL_FILE}")


# summary of scores per criterion
pilot_mask = df[PILOT_COL].fillna(False).astype(bool)
ok_mask = pilot_mask & (df[STATUS_COL] == "OK")
err_mask = pilot_mask & (df[STATUS_COL] == "ERROR")

print("\n============================================================")
print("RUN COMPLETE")
print("============================================================")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows selected: {int(pilot_mask.sum()):,}")
print(f"OK rows: {int(ok_mask.sum()):,}")
print(f"ERROR rows: {int(err_mask.sum()):,}")
print(f"Completed in this run: {completed_this_run:,}")

if ok_mask.any():
    print("\nEvidence-score summary among successfully evaluated rows:")
    for criterion in CRITERIA:
        col = f"{MODEL_KEY}_{criterion}_score"
        vals = pd.to_numeric(df.loc[ok_mask, col], errors="coerce").dropna()

        print(
            f"{criterion}: "
            f"n={len(vals):,} | "
            f"mean={vals.mean():.3f} | "
            f"median={vals.median():.3f} | "
            f">=0.50={(vals >= 0.50).mean()*100:.1f}% | "
            f"zero={(vals == 0).mean()*100:.1f}%"
        )