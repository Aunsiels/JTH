#!/usr/bin/env python3
"""
jaccard_baseline.py
===================
Rank by **Jaccard similarity of skills** only.

Given a query entity (candidate or job), compute the Jaccard score against every
item on the opposite side and return the descending list.

Jaccard(c, j) = |skills(c) ∩ skills(j)| / |skills(c) ∪ skills(j)|

• Empty skill sets → score 0.0.
• Ties are broken by (−|union|, id) so more specific profiles come first.

Usage
-----
```bash
python jaccard_baseline.py CANDIDATES.csv JOBS.csv TEST.csv OUT.json \
       --mode c2j
```

CLI arguments
~~~~~~~~~~~~~
* CANDIDATES.csv – columns `candidate_id, skills`
* JOBS.csv       – columns `job_id, skills`
* TEST.csv       – `candidate_id, job_id, timestamp` (or `job_offer_opening_date`)
* OUT.json       – evaluator‑ready predictions
* --mode         – `c2j` (default) or `j2c`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Jaccard‑similarity (skills) baseline.")
    ap.add_argument("candidates", type=Path)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    return ap.parse_args(argv)


def parse_skills_column(series: pd.Series) -> pd.Series:
    def to_set(val):
        if pd.isna(val) or val is None or str(val).strip() == "":
            return set()
        return {tok.strip().lower() for tok in str(val).split(";") if tok.strip()}
    return series.apply(to_set)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    cand_df = pd.read_csv(args.candidates, dtype=str, usecols=["candidate_id", "skills"])
    job_df  = pd.read_csv(args.jobs, dtype=str, usecols=["job_id", "skills"])

    cand_df["skills"] = parse_skills_column(cand_df["skills"])
    job_df["skills"]  = parse_skills_column(job_df["skills"])

    cand_skills: Dict[str, set[str]] = dict(zip(cand_df["candidate_id"].astype(str), cand_df["skills"]))
    job_skills: Dict[str, set[str]]  = dict(zip(job_df["job_id"].astype(str),  job_df["skills"]))

    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    output: Dict[str, List[str]] = {}

    if args.mode == "c2j":
        all_jobs = list(job_skills.keys())
        for cand, ts in zip(test_df["candidate_id"].astype(str), test_df["timestamp"].astype(str)):
            c_sk = cand_skills.get(cand, set())
            scores = []
            for j in all_jobs:
                j_sk = job_skills[j]
                if not c_sk and not j_sk:
                    sim = 0.0
                else:
                    sim = len(c_sk & j_sk) / len(c_sk | j_sk)
                scores.append(sim)
            # tie-break: more skills (union size) first, then id
            ranking = [j for _, j in sorted(zip(scores, all_jobs), key=lambda t: (-t[0], -len(c_sk | job_skills[t[1]]), t[1]))]
            output[json.dumps([cand, ts])] = ranking
    else:
        all_cands = list(cand_skills.keys())
        for job, ts in zip(test_df["job_id"].astype(str), test_df["timestamp"].astype(str)):
            j_sk = job_skills.get(job, set())
            scores = []
            for c in all_cands:
                c_sk = cand_skills[c]
                if not c_sk and not j_sk:
                    sim = 0.0
                else:
                    sim = len(c_sk & j_sk) / len(c_sk | j_sk)
                scores.append(sim)
            ranking = [c for _, c in sorted(zip(scores, all_cands), key=lambda t: (-t[0], -len(job_skills[job] | cand_skills[t[1]]), t[1]))]
            output[json.dumps([job, ts])] = ranking

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Jaccard baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
