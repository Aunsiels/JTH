#!/usr/bin/env python3
"""
popularity_recency_baseline.py
==============================
Online **popularity × recency** baseline.

Score(t, now)  =  popularity(t) * exp(-λ · age_days)

* `popularity(t)`   incremental count of interactions up to *now*.
* `age_days`        max(0, (now − create_date(t)).days).
* Items created **after** `now` get score 0 and therefore sink to the end.

The ranking list is recomputed **once per timestamp batch** (exact same update
cycle as the online-popularity baseline).

Usage
-----
```bash
python popularity_recency_baseline.py TRAIN.csv TEST.csv CANDIDATES.csv JOBS.csv \
       OUT.json --mode c2j --lambda 0.02
```

Arguments
~~~~~~~~~
TRAIN.csv     Interactions with cols `candidate_id, job_id, timestamp`
TEST.csv      Same columns (evaluation set)
CANDIDATES.csv, JOBS.csv  Metadata with `*_id, create_date`
OUT.json      Predictions in evaluator JSON format
--mode        `c2j` (jobs→candidates, default) or `j2c`
--lambda      Decay rate λ in the exponential term (default 0.02)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Popularity×Recency baseline (online, timestamp batches).")
    ap.add_argument("train", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("candidates", type=Path)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    ap.add_argument("--lambda", dest="decay", type=float, default=0.02, help="Recency decay λ (per day)")
    return ap.parse_args(argv)


def to_str(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("")


def load_create_dates(path: Path, id_col: str) -> Dict[str, pd.Timestamp]:
    df = pd.read_csv(path, dtype=str, usecols=[id_col, "create_date"])
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    # Impute mean date if missing
    mean_ts = pd.to_datetime(df["create_date"].dropna().astype("int64").mean()) if df["create_date"].notna().any() else pd.Timestamp("1970-01-01")
    df["create_date"] = df["create_date"].fillna(mean_ts)
    return dict(zip(df[id_col].astype(str), df["create_date"]))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    # Load metadata
    cand_dates = load_create_dates(args.candidates, "candidate_id")
    job_dates = load_create_dates(args.jobs, "job_id")

    # Load interactions
    train_df = pd.read_csv(args.train, dtype=str)
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

    # Popularity counts
    tgt_col = "job_id" if args.mode == "c2j" else "candidate_id"
    qry_col = "candidate_id" if args.mode == "c2j" else "job_id"

    pop: Dict[str, int] = train_df[tgt_col].astype(str).value_counts().to_dict()

    # Pre-load create dates for all targets in train/test to ensure presence
    if args.mode == "c2j":
        for jid in pd.concat([train_df[tgt_col], test_df[tgt_col]]).astype(str).unique():
            pop.setdefault(jid, 0)
            job_dates.setdefault(jid, pd.Timestamp("1970-01-01"))
    else:
        for cid in pd.concat([train_df[tgt_col], test_df[tgt_col]]).astype(str).unique():
            pop.setdefault(cid, 0)
            cand_dates.setdefault(cid, pd.Timestamp("1970-01-01"))

    decay = args.decay

    output: Dict[str, List[str]] = {}

    for ts, grp in test_df.sort_values("timestamp").groupby("timestamp"):
        now = ts
        # compute scores for all targets
        scores: Dict[str, float] = {}
        if args.mode == "c2j":
            for jid in pop.keys():
                age = max(0, (now - job_dates[jid]).days)
                scores[jid] = pop[jid] * math.exp(-decay * age)
        else:
            for cid in pop.keys():
                age = max(0, (now - cand_dates[cid]).days)
                scores[cid] = pop[cid] * math.exp(-decay * age)
        ranked = sorted(scores.keys(), key=lambda i: (-scores[i], i))

        for qid in grp[qry_col].astype(str):
            output[json.dumps([qid, now.strftime("%Y-%m-%d")])] = ranked

        # update popularity counts after predictions
        for tgt in grp[tgt_col].astype(str):
            pop[tgt] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Popularity×Recency baseline written: {args.out} (queries: {len(output)})")


if __name__ == "__main__":
    main()
