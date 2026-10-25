**Multi-LLM judgments to fuzzy evidence profiles: gambling risk assessment**
Data, results, and code for the paper "From Multi-LLM Judgments to Fuzzy Evidence Profiles: A Comparative Study of Ensemble Fusion and Multicriteria Aggregation for Gambling Risk Assessment" (MDPI Electronics).
Five local LLM judges score nine DSM-5 gambling criteria (0–1) in the texts of 500 synthetic users.
**Stage 1 fuses the five judgments per criterion** (mean, median, majority vote, triangular fuzzy set).
**Stage 2 aggregates the nine criteria into one user score** (mean, crisp OWA, T1OWA with Yager weights, T1OWA with equal weights).
The scores are compared with an external risk label (GT1) and with supervised and zero-shot baselines.
The users and texts are synthetic. The scores are a measure of criterion-level textual evidence; they are not diagnoses. GT1 comes from the synthetic user profiles and has not been validated by clinicians.
