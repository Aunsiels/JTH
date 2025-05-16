#!/usr/bin/env python3
"""
random_baseline.py
------------------
Generate a random‑ranking baseline JSON compatible with *evaluate_recsys.py*.

Usage
-----
$ python random_baseline.py CANDIDATES.csv JOBS.csv TEST.csv OUT.json \
      --mode c2j [--seed 42]

Arguments
~~~~~~~~~
CANDIDATES.csv   File containing at least ``candidate_id``.
JOBS.csv         File containing at least ``job_id``.
TEST.csv         The temporal test split (needs ``candidate_id``, ``job_id``,
                 and ``timestamp``).
OUT.json         Path to write the generated recommendations.
--mode           "c2j" for *jobs recommended to candidates* (default) or
                 "j2c" for *candidates recommended to jobs*.
--seed           RNG seed for repeatability.

Output format matches the evaluator: each key is a JSON list string
``[id, timestamp]`` and the value is a fully‑shuffled list of targets.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a random ranking baseline.")
    p.add_argument("candidates", type=Path, help="Candidates CSV (candidate_id column)")
    p.add_argument("jobs", type=Path, help="Jobs CSV (job_id column)")
    p.add_argument("test", type=Path, help="Test set CSV (needs timestamps)")
    p.add_argument("out", type=Path, help="Output JSON file")
    p.add_argument(
        "--mode", choices=["c2j", "j2c"], default="c2j",
        help="Recommendation direction: 'c2j' jobs to candidates (default) or 'j2c' candidates to jobs",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    return p.parse_args(argv)


def load_column(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path, dtype=str, usecols=[col])
    if col not in df.columns:
        raise ValueError(f"{path} missing required column '{col}'")
    return df[col].dropna().astype(str).unique().tolist()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    candidates = load_column(args.candidates, "candidate_id")
    jobs = load_column(args.jobs, "job_id")

    # Load test set ----------------------------------------------------------
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})
    if "timestamp" not in test_df.columns:
        raise ValueError("Test CSV must have a 'timestamp' column")

    required = {"candidate_id", "job_id"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Test CSV missing columns: {', '.join(sorted(missing))}")

    output: Dict[str, List[str]] = {}

    if args.mode == "c2j":
        target_pool = jobs
        query_col = "candidate_id"
    else:
        target_pool = candidates
        query_col = "job_id"

    for row in test_df.itertuples(index=False):
        query_id = getattr(row, query_col)
        ts = getattr(row, "timestamp")
        key = json.dumps([query_id, ts])
        ranked_list = target_pool.copy()
        random.shuffle(ranked_list)
        output[key] = ranked_list

    # Save -------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Random baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
