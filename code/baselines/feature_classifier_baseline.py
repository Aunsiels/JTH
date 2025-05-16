#!/usr/bin/env python3
"""
feature_classifier_baseline.py
==============================
Classifier‑based ranking baseline using similarity & temporal features – **now
with automatic missing‑value imputation**.

* For every *categorical/string* column (`skills`, `contract_type`,
  `expertise_area`, `years_experience`) **missing or empty values are replaced
  by the most frequent non‑empty value in that column**.
* For `create_date` we continue to impute with the mean timestamp.

Features per (candidate, job, timestamp) pair
--------------------------------------------
1. `n_common_skills`     – overlap size (after “;” split)
2. `contract_match`      – job contract ∈ candidate contract list?
3. `expertise_match`     – equal expertise area?
4. `years_match`         – equal experience bucket?
5. `entity_delta_days`   – |candidate_date − job_date|
6. `rec_delta_days`      – |target_date − interaction_timestamp|
7. `existed_before`      – target_date ≤ timestamp ? 1 : 0

CLI unchanged – see `--help`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classifier-based recommendation baseline with temporal features and missing-value imputation.")
    p.add_argument("candidates", type=Path)
    p.add_argument("jobs", type=Path)
    p.add_argument("train", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    p.add_argument("--neg_ratio", type=float, default=1.0)
    p.add_argument("--model", choices=["logreg", "tree", "forest"], default="forest")
    p.add_argument("--use_temporal_features", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def impute_mean_datetime(series: pd.Series) -> pd.Series:
    if series.isna().all():
        raise ValueError("All create_date values missing; cannot impute mean timestamp.")
    mean_ns = series.dropna().astype("int64").mean()
    return series.fillna(pd.to_datetime(int(mean_ns)))


def fill_with_mode(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column where NaN / empty strings replaced with most frequent non‑empty value."""
    col_ser = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")
    mode_val = col_ser[col_ser.str.strip() != ""].mode(dropna=True)
    mode_val = mode_val.iloc[0] if not mode_val.empty else ""
    filled = col_ser.replace("", mode_val)
    return filled


def parse_skills(s: str) -> set[str]:
    if not s:
        return set()
    return {tok.strip().lower() for tok in s.split(";") if tok.strip()}


def parse_contract_list(s: str) -> set[str]:
    if not s:
        return set()
    try:
        data = json.loads(s.replace("'", "\""))
        if isinstance(data, list):
            return {str(x).strip().lower() for x in data if str(x).strip()}
    except Exception:
        pass
    return {tok.strip().lower() for tok in s.replace(";", ",").split(",") if tok.strip()}


# ---------------------------------------------------------------------------
# Entity loaders with mode imputation
# ---------------------------------------------------------------------------

def load_candidates(path: Path) -> Dict[str, dict]:
    df = pd.read_csv(path, dtype=str)
    req = ["candidate_id", "skills", "contract_type",
           "expertise_area", "job_category", "years_experience", "create_date"]
    missing = set(req) - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {', '.join(sorted(missing))}")

    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_date"] = impute_mean_datetime(df["create_date"])

    for col in ["skills", "contract_type", "expertise_area", "job_category", "years_experience", "technical_skills", "soft_skills"]:
        if col in df.columns:
            df[col] = fill_with_mode(df, col)

    store = {}
    for r in df.itertuples(index=False):
        skills = []
        if r.skills:
            skills.append(r.skills)
        if "technical_skills" in df.columns and r.technical_skills:
            skills.append(r.technical_skills)
        if "soft_skills" in df.columns and r.soft_skills:
            skills.append(r.soft_skills)
        store[str(r.candidate_id)] = {
            "skills": parse_skills(";".join(skills)),
            "contract": parse_skills(r.contract_type),
            "expertise": parse_skills(r.expertise_area),
            "job_category": parse_skills(r.job_category),
            "years": r.years_experience.strip().lower(),
            "date": r.create_date,
        }
    return store


def load_jobs(path: Path) -> Dict[str, dict]:
    df = pd.read_csv(path, dtype=str)
    req = ["job_id", "skills", "contract_type", "expertise_area", "job_category", "years_experience", "create_date"]
    missing = set(req) - set(df.columns)
    if missing:
        raise ValueError(f"Jobs CSV missing columns: {', '.join(sorted(missing))}")

    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_date"] = impute_mean_datetime(df["create_date"])

    for col in ["skills", "contract_type", "expertise_area", "job_category", "years_experience"]:
        df[col] = fill_with_mode(df, col)

    store = {}
    for r in df.itertuples(index=False):
        store[str(r.job_id)] = {
            "skills": parse_skills(r.skills),
            "contract": parse_skills(r.contract_type),
            "expertise": parse_skills(r.expertise_area),
            "job_category": parse_skills(r.job_category),
            "years": r.years_experience.strip().lower(),
            "date": r.create_date,
        }
    return store


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(cand: dict, job: dict, rec_ts: pd.Timestamp, mode: str, temporal_features: bool):
    if job["skills"]:
        n_common_skills = len(cand["skills"] & job["skills"]) / len(job["skills"])
    else:
        n_common_skills = 0
    if job["contract"]:
        contract_match = len(cand["contract"] & job["contract"]) / len(job["contract"])
    else:
        contract_match = 0
    if job["expertise"]:
        expertise_match = len(cand["expertise"] & job["expertise"]) / len(job["expertise"])
    else:
        expertise_match = 0
    if job["job_category"]:
        job_match = len(cand["job_category"] & job["job_category"]) / len(job["job_category"])
    else:
        job_match = 0
    years_match = int(cand["years"] == job["years"])

    if temporal_features:
        entity_delta = abs((cand["date"] - job["date"]).days)
        target_date = job["date"] if mode == "c2j" else cand["date"]
        rec_delta = abs((rec_ts - target_date).days)
        existed_before = int(target_date <= rec_ts)
    else:
        entity_delta = rec_delta = existed_before = 0

    return (n_common_skills, contract_match, expertise_match, job_match, years_match,
            entity_delta, rec_delta, existed_before)


# ---------------------------------------------------------------------------
# Negative sampling (unchanged)
# ---------------------------------------------------------------------------

def sample_negatives_per_row(c_id: str, j_id: str, ts: pd.Timestamp, cand_ids: List[str], job_ids: List[str],
                             mode: str, n_needed: int, rng: random.Random):
    negs = []
    used_c = {c_id}
    used_j = {j_id}
    while len(negs) < n_needed:
        if mode == "c2j":
            neg_job = rng.choice(job_ids)
            if neg_job in used_j:
                continue
            used_j.add(neg_job)
            negs.append((c_id, neg_job, ts))
        else:
            neg_cand = rng.choice(cand_ids)
            if neg_cand in used_c:
                continue
            used_c.add(neg_cand)
            negs.append((neg_cand, j_id, ts))
    return negs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    rng = random.Random(args.seed)

    candidates = load_candidates(args.candidates)
    jobs = load_jobs(args.jobs)
    cand_ids, job_ids = list(candidates.keys()), list(jobs.keys())

    # -------------------- Training data --------------------
    train_df = pd.read_csv(args.train, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    if train_df["timestamp"].isna().any():
        raise ValueError("Training data contains invalid timestamps.")

    X, y = [], []
    for row in train_df.itertuples(index=False):
        c_id, j_id, ts = str(row.candidate_id), str(row.job_id), row.timestamp
        X.append(build_features(candidates[c_id], jobs[j_id], ts, args.mode, args.use_temporal_features))
        y.append(1)
        for nc, nj, nts in sample_negatives_per_row(c_id, j_id, ts, cand_ids, job_ids, args.mode, int(args.neg_ratio),
                                                    rng):
            X.append(build_features(candidates[nc], jobs[nj], nts, args.mode, args.use_temporal_features))
            y.append(0)

    X_arr, y_arr = np.asarray(X, float), np.asarray(y)

    # -------------------- Train model --------------------
    if args.model == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    elif args.model == "tree":
        clf = DecisionTreeClassifier(random_state=args.seed)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    clf.fit(X_arr, y_arr)

    # -------------------- Inference --------------------
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")
    if test_df["timestamp"].isna().any():
        raise ValueError("Test data contains invalid timestamps.")

    target_ids = job_ids if args.mode == "c2j" else cand_ids
    output: Dict[str, List[str]] = {}

    for row in test_df.itertuples(index=False):
        c_id, j_id, ts = str(row.candidate_id), str(row.job_id), row.timestamp
        if args.mode == "c2j":
            feats = [build_features(candidates[c_id], jobs[j], ts, "c2j", args.use_temporal_features)
                     for j in target_ids]
            preds = clf.predict_proba(np.asarray(feats, float))[:, 1]
            ranking = [tid for _, tid in sorted(zip(preds, target_ids), key=lambda t: -t[0])]
            query_id = c_id
        else:
            feats = [build_features(candidates[c], jobs[j_id], ts, "j2c", args.use_temporal_features)
                     for c in target_ids]
            preds = clf.predict_proba(np.asarray(feats, float))[:, 1]
            ranking = [tid for _, tid in sorted(zip(preds, target_ids), key=lambda t: -t[0])]
            query_id = j_id

        output[json.dumps([query_id, ts.strftime("%Y-%m-%d")])] = ranking

    # -------------------- Save --------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Classifier baseline with temporal features written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
