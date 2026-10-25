# Multi-LLM judgments to fuzzy evidence profiles: gambling risk assessment

Data, results and code for the paper *"From Multi-LLM Judgments to Fuzzy Evidence Profiles: A Comparative Study of Ensemble Fusion and Multicriteria Aggregation for Gambling Risk Assessment"* (MDPI *Electronics*).

Five local LLM judges score nine DSM-5 gambling criteria (0–1) in the texts of 500 synthetic users. Stage 1 fuses the five judgments per criterion (mean, median, majority vote, triangular fuzzy set). Stage 2 aggregates the nine criteria into one user score (mean, crisp OWA, T1OWA with Yager weights, T1OWA with equal weights). The scores are compared with an external risk label (GT1) and with supervised and zero-shot baselines.

The users and texts are synthetic. The scores measure criterion-level textual evidence; they are not diagnoses. GT1 is derived from the synthetic user profiles and was not validated by clinicians.

## Repository structure

```
README.md
LICENSE
requirements.txt
code/
  judge_scores_nine_criteria.py      judge scoring with Ollama (one run per judge)
  experiments_1_3.py                 Experiments 1-3 (fusion, operators, leave-one-judge-out)
  type1owa_zhou.py                   T1OWA via alpha-cuts (port of the R package Type1OWA 1.0)
  metrics_utils.py                   metrics and bootstrap intervals (used by the baselines)
  tfidf_svm_baseline.py              supervised baseline: TF-IDF + SVM
  bert_baseline.py                   supervised baseline: fine-tuned RoBERTa
  zero_shot_llm_run.py               zero-shot baseline: direct risk level from one LLM
  zero_shot_llm_aggregate.py         zero-shot baseline: evaluation and combinations
data/
  Source/                            source monologues
  user_texts_concatenated.csv        all monologues of each user (columns main_user, Monologue, risk_level)
  user_labels_gt1.xlsx               GT1 label (risk_level) per user
  judge_scores_with_evidence_nine_criteria/   raw outputs of the five judges (scores and evidence spans)
  judge_scores_nine_criteria.xlsx    scores of the five judges for the 500 scored users (also defines the test set)
  zero_shot/                         direct_<model>.xlsx from the zero-shot baseline
results/
  experiments_1_3.xlsx               Experiments 1-3
  baseline_tfidf_svm.csv, preds_tfidf_svm.csv
  baseline_roberta-base.csv, preds_roberta-base.csv
  baseline_llm_zeroshot_all.xlsx
```

## Judges

| Run key | Ollama model | Column suffix in the score file |
|---|---|---|
| llama370 | llama3.1:70b | `llama3.1:70b` |
| llama3 | llama3.3:70b | `llama3_3_70b` |
| deepseek | deepseek-r1:70b | `deepseek_r1` |
| qwen3 | qwen3:32b | `qwen3_32b` |
| gemma3 | gemma3:27b | `gemma3:27b` |

`data/judge_scores_nine_criteria.xlsx` has one row per user (`User`) and the columns `A1_<judge>` … `A9_<judge>`, which hold the score of criterion A1–A9 from each judge. The `A<k>_n_llm` columns give the number of valid judge scores.

## Setup

Python 3.9 or later.

```
pip install -r requirements.txt
```

The judge and zero-shot runs need [Ollama](https://ollama.com) with the models above. The RoBERTa baseline runs on a GPU.

## Reproducing the results

Run all commands from the repository root.

**Experiments 1–3** (needs only the score and label files):

```
python code/experiments_1_3.py
```

It writes `results/experiments_1_3.xlsx`. Reference check: criterion median fusion with crisp OWA (F2 + A2) gives Spearman 0.292 [0.210, 0.372].

**Baselines:**

```
python code/tfidf_svm_baseline.py
python code/bert_baseline.py
python code/zero_shot_llm_run.py
python code/zero_shot_llm_aggregate.py
```

Run `zero_shot_llm_run.py` once per model after setting `MODEL_KEY` and `MODEL_NAME` at the top of the file. `zero_shot_llm_aggregate.py` then reads the five `direct_<model>.xlsx` files from `data/zero_shot/`.

**Judge scoring** (to repeat the scoring from the texts):

```
python code/judge_scores_nine_criteria.py
```

Set `MODEL_KEY` and `MODEL_NAME` at the top of the file and run once per judge. The output goes to `data/judge_scores_with_evidence_nine_criteria/`. Local LLM output can differ between hardware and Ollama versions, so a new run may not reproduce the stored scores exactly. The stored scores are the ones used in the paper.

All bootstrap intervals use 2,000 resamples and seed 42.

## Licence

- Code: MIT (see `LICENSE`), except `code/type1owa_zhou.py`, which is a Python port of the GPL-licensed R package Type1OWA 1.0 and keeps the original licence terms.
- Data and results: derived from a public dataset released under CC BY-NC 4.0, so the same non-commercial terms apply. Source dataset: see the paper.

## Citation

If you use this repository, please cite the paper (reference to be added after publication) and the archived release: DOI to be added (Zenodo).
