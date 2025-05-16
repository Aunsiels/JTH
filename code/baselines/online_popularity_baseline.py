#!/usr/bin/env python3
"""
online_popularity_baseline.py
=============================
*Incremental* global‑popularity baseline that walks through the test set
chronologically and updates popularity counts after each timestamp batch.

How it works
------------
1. **Init**: counts of job (or candidate) occurrences from *train.csv*.
2. **Loop** through unique timestamps in *test.csv* (old → new):
   • Build the popularity ranking with current counts.
   • Emit that same list for every query at this timestamp.
   • **Update counts** with the just‑revealed interactions.

Thus the list can evolve over time, capturing newly popular items.

Usage
~~~~~
```bash
python online_popularity_baseline.py TRAIN.csv TEST.csv OUT.json \
       --mode c2j
```

`--mode c2j`  (default)  : recommend **jobs** to each candidate.
`--mode j2c`            : recommend **candidates** to each job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Online global popularity baseline.")
    ap.add_argument("train", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    return ap.parse_args(argv)


def to_str(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("")


def build_ranked_list(pop_counts: Dict[str, int]) -> List[str]:
    return sorted(pop_counts.keys(), key=lambda i: (-pop_counts[i], i))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    train_df = pd.read_csv(args.train, dtype=str)
    test_df = pd.read_csv(args.test, dtype=str)

    # Determine target and query columns depending on mode
    tgt_col = "job_id" if args.mode == "c2j" else "candidate_id"
    qry_col = "candidate_id" if args.mode == "c2j" else "job_id"

    # Initial popularity counts from TRAIN ----------------------------------
    pop: Dict[str, int] = train_df[tgt_col].astype(str).value_counts().to_dict()

    # Ensure timestamps are usable ------------------------------------------
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

    # Output container ------------------------------------------------------
    output: Dict[str, List[str]] = {}

    # Walk through timestamps ----------------------------------------------
    for ts, grp in test_df.sort_values("timestamp").groupby("timestamp"):
        # Include unseen targets in pop dict with count 0 so they appear in list
        new_targets = grp[tgt_col].astype(str).unique()
        for t in new_targets:
            pop.setdefault(t, 0)

        ranked = build_ranked_list(pop)
        # Emit same list for every query in this timestamp batch
        for qid in grp[qry_col].astype(str):
            output[json.dumps([qid, ts.strftime("%Y-%m-%dT%H:%M:%S")])] = ranked

        # Update popularity with this batch interactions
        for tgt in grp[tgt_col].astype(str):
            pop[tgt] += 1

    # Write JSON ------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Online popularity baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
