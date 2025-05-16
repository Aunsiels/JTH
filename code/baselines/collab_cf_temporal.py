#!/usr/bin/env python3
"""
collab_cf_temporal.py
=====================
Online collaborative‑filtering baseline that incrementally retrains as it
walks through the test set in timestamp order.

Algorithm
~~~~~~~~~
* **Initial state**: all interactions from TRAIN CSV (columns
  `candidate_id, job_id, timestamp`).
* At evaluation time:
  1. Take unique timestamps from TEST CSV, sort ascending.
  2. For each timestamp *t*:
     • Generate recommendations for every test row at *t* using the current
       CF model (item–item or user–user similarity).
     • Append those interactions to the training pool (**online update**).

Similarity model
----------------
We use simple co‑occurrence based cosine similarity:

For **candidate→job** (``--mode c2j``)
::
    sim(j1,j2) = |users(j1) ∩ users(j2)| / sqrt(pop(j1) * pop(j2))

Score for a candidate *c* and unseen job *j* is::

    score(c, j) = Σ_{j'∈Jobs(c)} sim(j, j')

For **job→candidate** the formulation is symmetrical on the candidate side.

All jobs/candidates are returned, ordered by descending score; ties broken by
popularity, then ID.

Usage
-----
```bash
python collab_cf_temporal.py TRAIN.csv TEST.csv OUT.json --mode c2j --seed 42
```
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# CLI and helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Online CF baseline with temporal walk.")
    ap.add_argument("train", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Data structures for incremental CF
# ---------------------------------------------------------------------------
class CFState:
    """Holds interaction sets, popularity counts, and co‑occurrence tables."""

    def __init__(self):
        # candidate → set(jobs)
        self.cand_jobs: Dict[str, set[str]] = defaultdict(set)
        # job → set(candidates)
        self.job_cands: Dict[str, set[str]] = defaultdict(set)
        # popularity
        self.job_pop: Dict[str, int] = defaultdict(int)
        self.cand_pop: Dict[str, int] = defaultdict(int)
        # co-occurrence counts (sparse symmetric)
        self.job_co: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cand_co: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # -------------------- building / updating --------------------
    def add_interaction(self, cand: str, job: str):
        """Add (candidate, job) positive interaction; idempotent."""
        if job in self.cand_jobs[cand]:
            return  # already seen
        # Update co‑occurrence: job side
        for j_prev in self.cand_jobs[cand]:
            self.job_co[j_prev][job] += 1
            self.job_co[job][j_prev] += 1
        # Update co‑occurrence: candidate side
        for c_prev in self.job_cands[job]:
            self.cand_co[c_prev][cand] += 1
            self.cand_co[cand][c_prev] += 1
        # Register sets & pops
        self.cand_jobs[cand].add(job)
        self.job_cands[job].add(cand)
        self.job_pop[job] += 1
        self.cand_pop[cand] += 1

    # -------------------- ranking --------------------
    def recommend_jobs_for(self, cand: str, all_jobs: List[str]) -> List[str]:
        seen = self.cand_jobs.get(cand, set())
        scores: Dict[str, float] = defaultdict(float)
        for j_prev in seen:
            pop_prev = self.job_pop[j_prev]
            for j, co in self.job_co[j_prev].items():
                if j in seen:
                    continue
                pop_j = self.job_pop[j]
                if pop_prev and pop_j:
                    scores[j] += co / math.sqrt(pop_prev * pop_j)

        # Fallback: unseen items with zero score
        def sort_key(j):
            return (-scores.get(j, 0.0), -self.job_pop[j], j)

        ranked = sorted(all_jobs, key=sort_key)
        return ranked

    def recommend_cands_for(self, job: str, all_cands: List[str]) -> List[str]:
        seen = self.job_cands.get(job, set())
        scores: Dict[str, float] = defaultdict(float)
        for c_prev in seen:
            pop_prev = self.cand_pop[c_prev]
            for c, co in self.cand_co[c_prev].items():
                if c in seen:
                    continue
                pop_c = self.cand_pop[c]
                if pop_prev and pop_c:
                    scores[c] += co / math.sqrt(pop_prev * pop_c)

        def sort_key(c):
            return -scores.get(c, 0.0), -self.cand_pop[c], c

        ranked = sorted(all_cands, key=sort_key)
        return ranked


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    rng = random.Random(args.seed)

    # Load datasets
    train_df = pd.read_csv(args.train, dtype=str)
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    # Build state from training interactions
    state = CFState()
    for c, j in zip(train_df["candidate_id"].astype(str), train_df["job_id"].astype(str)):
        state.add_interaction(c, j)

    all_cands = list(state.cand_jobs.keys())
    all_jobs = list(state.job_cands.keys())

    # Prepare timestamp order
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")
    test_df = test_df.sort_values("timestamp")
    ts_groups = test_df.groupby("timestamp")

    output: Dict[str, List[str]] = {}

    for ts, group in ts_groups:
        # --- make predictions using current model ---
        if args.mode == "c2j":
            for cand, job in zip(group["candidate_id"].astype(str), group["job_id"].astype(str)):
                ranking = state.recommend_jobs_for(cand, all_jobs)
                output[json.dumps([cand, ts.strftime("%Y-%m-%d")])] = ranking
        else:  # j2c
            for cand, job in zip(group["candidate_id"].astype(str), group["job_id"].astype(str)):
                ranking = state.recommend_cands_for(job, all_cands)
                output[json.dumps([job, ts.strftime("%Y-%m-%d")])] = ranking
        # --- update state with these interactions ---
        for cand, job in zip(group["candidate_id"].astype(str), group["job_id"].astype(str)):
            state.add_interaction(cand, job)
            if job not in all_jobs:
                all_jobs.append(job)
            if cand not in all_cands:
                all_cands.append(cand)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Collaborative CF baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
