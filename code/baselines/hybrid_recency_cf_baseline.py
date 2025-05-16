#!/usr/bin/env python3
"""
hybrid_recency_cf_baseline.py
=============================
*Item‑item / user‑user* cosine collaborative filtering **filtered by recency**.

Pipeline (online, timestamp batches)
------------------------------------
1. Build initial CF model from TRAIN interactions (same as `collab_cf_temporal`).
2. For each timestamp batch in TEST (oldest → newest):
   * **Filter targets** to those that
        • existed before the timestamp and
        • are no older than *max_age* days.
   * Rank the filtered set with the cosine CF scores.
   * Append the *stale* or *future* items afterwards, ordered by original CF
     score (future items with score 0 at the very end).
   * After emitting predictions, ingest the batch interactions to update the
     CF state.

Usage
~~~~~
```bash
python hybrid_recency_cf_baseline.py TRAIN.csv TEST.csv CANDIDATES.csv JOBS.csv \
       OUT.json --mode c2j --max_age 180
```

Parameters
~~~~~~~~~~
--mode      `c2j` (default) or `j2c`
--max_age   Maximum age in *days* for an item to be considered “fresh” (default 180).

Metadata files are needed for `create_date` so age can be computed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# CF state (re‑used from previous cosine implementation)
# ---------------------------------------------------------------------------
class CFState:
    def __init__(self):
        self.cand_jobs: Dict[str, set[str]] = defaultdict(set)
        self.job_cands: Dict[str, set[str]] = defaultdict(set)
        self.job_pop: Dict[str, int] = defaultdict(int)
        self.cand_pop: Dict[str, int] = defaultdict(int)
        self.job_co: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cand_co: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, cand: str, job: str):
        if job in self.cand_jobs[cand]:
            return
        for j_prev in self.cand_jobs[cand]:
            self.job_co[j_prev][job] += 1
            self.job_co[job][j_prev] += 1
        for c_prev in self.job_cands[job]:
            self.cand_co[c_prev][cand] += 1
            self.cand_co[cand][c_prev] += 1
        self.cand_jobs[cand].add(job)
        self.job_cands[job].add(cand)
        self.job_pop[job] += 1
        self.cand_pop[cand] += 1

    def sim_jobs(self, job: str, seen: set[str]):
        scores = defaultdict(float)
        pop_j = self.job_pop[job]
        for j2, co in self.job_co[job].items():
            if j2 in seen:
                continue
            pop2 = self.job_pop[j2]
            if pop_j and pop2:
                scores[j2] += co / math.sqrt(pop_j * pop2)
        return scores

    def sim_cands(self, cand: str, seen: set[str]):
        scores = defaultdict(float)
        pop_c = self.cand_pop[cand]
        for c2, co in self.cand_co[cand].items():
            if c2 in seen:
                continue
            pop2 = self.cand_pop[c2]
            if pop_c and pop2:
                scores[c2] += co / math.sqrt(pop_c * pop2)
        return scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Hybrid recency‑filtered CF baseline.")
    ap.add_argument("train", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("candidates", type=Path)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    ap.add_argument("--max_age", type=int, default=180, help="Max age (days) for freshness filter")
    return ap.parse_args(argv)


def load_dates(path: Path, id_col: str) -> Dict[str, pd.Timestamp]:
    df = pd.read_csv(path, dtype=str, usecols=[id_col, "create_date"])
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    # impute with mean if missing
    if df["create_date"].isna().all():
        df["create_date"] = pd.Timestamp("1970-01-01")
    else:
        mean_ts = pd.to_datetime(df["create_date"].dropna().astype("int64").mean())
        df["create_date"] = df["create_date"].fillna(mean_ts)
    return dict(zip(df[id_col].astype(str), df["create_date"]))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    max_age = pd.Timedelta(days=args.max_age)

    cand_dates = load_dates(args.candidates, "candidate_id")
    job_dates  = load_dates(args.jobs, "job_id")

    train_df = pd.read_csv(args.train, dtype=str)
    test_df  = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"], errors="coerce")

    state = CFState()
    for c, j in zip(train_df["candidate_id"].astype(str), train_df["job_id"].astype(str)):
        state.add(c, j)

    all_cands = set(list(state.cand_jobs.keys()) + list(test_df["candidate_id"].astype(str).unique()))
    all_jobs  = set(list(state.job_cands.keys()) + list(test_df["job_id"].astype(str).unique()))

    output: Dict[str, List[str]] = {}

    for ts, grp in test_df.sort_values("timestamp").groupby("timestamp"):
        now = ts
        if args.mode == "c2j":
            for cand in grp["candidate_id"].astype(str):
                seen = state.cand_jobs.get(cand, set())
                sim_scores = defaultdict(float)
                for j_prev in seen:
                    sim_scores.update(state.sim_jobs(j_prev, seen))
                # ensure all jobs in dict
                for jid in all_jobs:
                    sim_scores.setdefault(jid, 0.0)
                # freshness filtering
                fresh, stale = [], []
                for jid, score in sim_scores.items():
                    created = job_dates.get(jid, pd.Timestamp("1970-01-01"))
                    if created <= now and (now - created) <= max_age:
                        fresh.append((score, jid))
                    else:
                        stale.append((score, jid))
                fresh_sorted = [j for _, j in sorted(fresh, key=lambda t: (-t[0], j))]
                stale_sorted = [j for _, j in sorted(stale, key=lambda t: (-t[0], j))]
                output[json.dumps([cand, now.strftime("%Y-%m-%d")])] = fresh_sorted + stale_sorted
        else:  # j2c
            for job in grp["job_id"].astype(str):
                seen = state.job_cands.get(job, set())
                sim_scores = defaultdict(float)
                for c_prev in seen:
                    sim_scores.update(state.sim_cands(c_prev, seen))
                for cid in all_cands:
                    sim_scores.setdefault(cid, 0.0)
                fresh, stale = [], []
                for cid, score in sim_scores.items():
                    created = cand_dates.get(cid, pd.Timestamp("1970-01-01"))
                    if created <= now and (now - created) <= max_age:
                        fresh.append((score, cid))
                    else:
                        stale.append((score, cid))
                fresh_sorted = [c for _, c in sorted(fresh, key=lambda t: (-t[0], t[1]))]
                stale_sorted = [c for _, c in sorted(stale, key=lambda t: (-t[0], t[1]))]
                output[json.dumps([job, now.strftime("%Y-%m-%d")])] = fresh_sorted + stale_sorted

        # update CF state with this batch
        for c, j in zip(grp["candidate_id"].astype(str), grp["job_id"].astype(str)):
            state.add(c, j)
            all_cands.add(c)
            all_jobs.add(j)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Hybrid recency‑filtered CF written: {args.out} (queries: {len(output)})")


if __name__ == "__main__":
    main()
