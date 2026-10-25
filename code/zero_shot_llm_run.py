# Zero-shot LLM baseline: one local LLM (via Ollama) reads all monologues of a user and returns ONE overall risk level (none / low / moderate / high).
# Runs on the 500 scored test users only; run once per model (set MODEL_KEY / MODEL_NAME), from the repository root: python code/zero_shot_llm_run.py
# Output: data/zero_shot/direct_<MODEL_KEY>.xlsx. Re-running resumes from the journal. Then run zero_shot_llm_aggregate.py.
import pandas as pd
import ollama
import json
import os
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_KEY  = "deepseek"          #  llama3        | gemma3    | deepseek        | qwen3    | llama370
MODEL_NAME = "deepseek-r1:70b"   # | llama3.3:70b , gemma3:27b, deepseek-r1:70b, qwen3:32b ,llama3.1:70b


INPUT_FILE      = "data/user_texts_concatenated.csv"          # columns: main_user (or User), Monologue
PROFILES_FILE   = "data/user_labels_gt1.csv" # ground truth risk_level (used ONLY for scoring)
TEST_USERS_FILE = "data/test_users_500.xlsx" # index main_user = the 500 test users
OUTPUT_DIR      = "data/zero_shot"

OUTPUT_FILE  = os.path.join(OUTPUT_DIR, f"direct_{MODEL_KEY}.xlsx")
JOURNAL_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_journal.jsonl"
TMP_FILE     = os.path.splitext(OUTPUT_FILE)[0] + "_tmp.xlsx"

MAX_ROWS = None           # None = all 500 test users; a number runs only the first N
ONE_SHOT = False          # True = one-shot (one in-prompt example); False = zero-shot

TEMPERATURE = 0.0
OLLAMA_SEED = 42
NUM_CTX = 8192 if MODEL_KEY == "deepseek" else (4096 if MODEL_KEY == "qwen3" else 2048)   # reasoning models need room to think
MAX_USER_CHARS = 8000   # safety cut of the joined text (about 4,000 tokens); check n_truncated in the summary
CHECKPOINT_EVERY = 10
# --- speed settings ---
KEEP_ALIVE  = "2h"        # keep the model loaded in GPU memory between calls
DISABLE_THINK = MODEL_KEY in ("qwen3", "deepseek")   # skip the <think> block (needs ollama python >= 0.5)
NUM_PREDICT = 1024 if DISABLE_THINK else 120         # answer is ~50 tokens; reasoning models get a ceiling that cannot cut the JSON
NUM_PREDICT_REASONING = 3000                         # deepseek fallback: let it reason, then answer
N_WORKERS = 1             # >1 sends users in parallel; first start the server with OLLAMA_NUM_PARALLEL=<same number>
MAX_RETRIES = 3

LEVELS = ["none", "low", "moderate", "high"]

# =============================================================================
# PROMPT
# =============================================================================

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

# Short criterion names (the judges received the full definitions; here they only guide the overall judgement).
CRITERIA_SHORT = ("tolerance, withdrawal/irritability, loss of control, preoccupation, escape from distress, "
                  "chasing losses, lying/secrecy, jeopardised relationships or work, financial bailout")

EXAMPLE_TEXT = ("Post 1: Said no, but felt lonely after a €50 ATM withdrawal at 2 am. "
                "Post 2: Felt a bit too much during late-night games, ATM trips; €50, €75, €20. "
                "Post 3: Late-night ATM, awkward first time discussing salaries, felt like I'm drowning in secrets. "
                "Post 4: Late-night ATM, just checked account, €50 gone again... Tempted to cancel, but €20 still in LuckySpin.")
EXAMPLE_ANSWER = '{"evidence": "felt like I\'m drowning in secrets", "risk_level": "moderate"}'

def build_prompt(post):
    example = f"\nEXAMPLE\n{EXAMPLE_TEXT}\n{EXAMPLE_ANSWER}\n" if ONE_SHOT else ""
    return f"""
Rate the overall severity of gambling-related problems in the monologues of ONE person (synthetic data).
Assess the text, not the person; this is not a diagnosis. Use only what the texts state or clearly imply;
do not invent history, and negated behaviour is not evidence.
Guide: the DSM-5 criteria ({CRITERIA_SHORT}).

LEVELS (choose one):
none = no gambling, or gambling without signs of problems
low = at most one or two mild or uncertain signs
moderate = several clear signs, some loss of control or harm
high = many strong signs, marked loss of control and harm
{example}
"evidence" = an exact span of at most 12 words from the texts supporting your choice (null for none). No explanations.

MONOLOGUES:
{post}

Return ONLY JSON: {{"evidence": null, "risk_level": "none"}}
""".strip()


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", (text or ""), flags=re.DOTALL).strip()   # deepseek-r1 / qwen3
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found. Raw start: {text[:150]!r}")
    return json.loads(match.group(0))


def validate_result(obj):
    if not isinstance(obj, dict) or "risk_level" not in obj:
        raise ValueError("Missing risk_level.")
    level = str(obj["risk_level"]).strip().lower()
    if level not in LEVELS:
        raise ValueError(f"Invalid risk_level: {level!r}")
    ev = obj.get("evidence", None)
    ev = None if ev is None or not str(ev).strip() else str(ev).strip()[:300]
    return {"risk_level": level, "evidence": ev}


def call_model(post):
    prompt = build_prompt(post) + ("\n/no_think" if MODEL_KEY == "qwen3" else "")   # qwen3 soft switch: no reasoning block
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # deepseek: attempt 1 = no thinking + forced JSON (fast); if that returns nothing, attempts 2-3 let it
            # reason freely (no forced JSON, larger token limit) and the JSON is read after the </think> block.
            reasoning_mode = (MODEL_KEY == "deepseek" and attempt >= 2)
            kw = dict(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], keep_alive=KEEP_ALIVE,
                      options={"temperature": TEMPERATURE, "seed": OLLAMA_SEED, "num_ctx": NUM_CTX,
                               "num_predict": NUM_PREDICT_REASONING if reasoning_mode else NUM_PREDICT})
            if not reasoning_mode:
                kw["format"] = "json"
            if DISABLE_THINK and not reasoning_mode:
                try:
                    response = ollama.chat(**kw, think=False)
                except (TypeError, ollama.ResponseError):      # old client / model without the option
                    response = ollama.chat(**kw)
            else:
                response = ollama.chat(**kw)
            try:
                content = response["message"]["content"]
            except Exception:
                content = response.message.content
            return validate_result(extract_json(content)), None
        except Exception as e:
            last_error = f"attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))
    # IMPORTANT: failures return None, never 'none'.
    return None, last_error

# =============================================================================
# PERSISTENCE
# =============================================================================

def journal_append(user, status, result=None, error=None):
    rec = {"user": user, "model": MODEL_KEY, "status": status,
           "risk_level": (result or {}).get("risk_level"), "evidence": (result or {}).get("evidence"),
           "error": error, "time": datetime.now().isoformat(timespec="seconds")}
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_state():
    state = {}
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # half-written last line after a crash
                if rec.get("model") != MODEL_KEY:
                    continue
                if state.get(rec["user"], {}).get("status") == "OK":
                    continue                      # an ERROR never overwrites a finished user
                state[rec["user"]] = rec
    return state


def save_output(users, y_true, state):
    rows = []
    for u in users:
        r = state.get(u, {})
        rows.append({"User": u, "true_level": y_true.get(u), f"{MODEL_KEY}_risk_level": r.get("risk_level"),
                     f"{MODEL_KEY}_evidence": r.get("evidence"),
                     f"{MODEL_KEY}_status": r.get("status", "NOT_EVALUATED"), f"{MODEL_KEY}_error": r.get("error")})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_excel(TMP_FILE, index=False, engine="openpyxl")
    for attempt in range(1, 6):
        try:
            os.replace(TMP_FILE, OUTPUT_FILE)
            return
        except PermissionError:
            print(f"  output file is locked (open in Excel?) - retry {attempt}/5")
            time.sleep(3)
    print(f"  WARNING: could not replace {OUTPUT_FILE}; data is safe in the journal.")

# =============================================================================
# LOAD DATA
# =============================================================================

def read_any(path):
    return pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)

print(f"Loading: {INPUT_FILE}")
raw = read_any(INPUT_FILE)
ucol = next(c for c in ("main_user", "User", "user", "user_id") if c in raw.columns)
raw[ucol] = raw[ucol].astype(str).str.strip()
texts = (raw.dropna(subset=["Monologue"]).groupby(ucol)["Monologue"]
         .apply(lambda s: "\n\n".join(f"Post {i + 1}: {str(t).strip()}" for i, t in enumerate(s))).to_dict())

prof = read_any(PROFILES_FILE)
pcol = next(c for c in ("user_id", "main_user", "User", "user", "id") if c in prof.columns)
prof[pcol] = prof[pcol].astype(str).str.strip()
y_true = prof.set_index(pcol)["risk_level"].astype(str).str.strip().str.lower().to_dict()

test_users = sorted(pd.read_excel(TEST_USERS_FILE, index_col="main_user").index.astype(str).str.strip())
users = [u for u in test_users if u in texts]
if MAX_ROWS is not None:
    users = users[:MAX_ROWS]
missing = len(test_users) - len([u for u in test_users if u in texts])
print(f"Test users: {len(test_users)} | with text: {len(test_users) - missing} | this run: {len(users)}")

state = load_state()
eligible = [u for u in users if state.get(u, {}).get("status") != "OK"]
print(f"Model: {MODEL_NAME} | {'one-shot' if ONE_SHOT else 'zero-shot'}")
print(f"Already done: {len(users) - len(eligible)} | still needing evaluation: {len(eligible)}\n")

# =============================================================================
# RUN
# =============================================================================

n_trunc = sum(len(texts[u]) > MAX_USER_CHARS for u in users)
done = 0
n_before = len(users) - len(eligible)

def record(u, result, error, pos):
    global done
    print(f"[{n_before + pos}/{len(users)}] user={u}")
    if result is None:
        state[u] = {"user": u, "status": "ERROR", "error": error}
        journal_append(u, "ERROR", error=error)
        print(f"  ERROR: {error}")
    else:
        state[u] = {"user": u, "status": "OK", **result, "error": None}
        journal_append(u, "OK", result=result)
        done += 1
    if pos % CHECKPOINT_EVERY == 0:
        save_output(users, y_true, state)
        print(f"  checkpoint saved at {datetime.now().isoformat(timespec='seconds')}")

try:
    if N_WORKERS <= 1:
        for pos, u in enumerate(eligible, start=1):
            result, error = call_model(texts[u][:MAX_USER_CHARS])
            record(u, result, error, pos)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(call_model, texts[u][:MAX_USER_CHARS]): u for u in eligible}
            for pos, fut in enumerate(as_completed(futs), start=1):
                result, error = fut.result()
                record(futs[fut], result, error, pos)      # journal/state written from the main thread only
except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C) - saving progress...")
    raise
finally:
    save_output(users, y_true, state)
    print(f"Final save done. Journal: {JOURNAL_FILE}")

# =============================================================================
# SUMMARY
# =============================================================================

ok = [u for u in users if state.get(u, {}).get("status") == "OK"]
err = [u for u in users if state.get(u, {}).get("status") == "ERROR"]
print("\n============================================================\nRUN COMPLETE\n============================================================")
print(f"Output: {OUTPUT_FILE}")
print(f"Users: {len(users)} | OK: {len(ok)} | ERROR: {len(err)} | completed this run: {done}")
print(f"Texts cut at MAX_USER_CHARS ({MAX_USER_CHARS}): {n_trunc}")
if ok:
    print("Predicted levels:", pd.Series([state[u]['risk_level'] for u in ok]).value_counts().reindex(LEVELS).fillna(0).astype(int).to_dict())