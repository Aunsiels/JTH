#!/usr/bin/env python3
"""
past_temporal_baseline.py
=========================
Rank targets by *temporal proximity in the past* relative to the query’s
**timestamp** (taken from the test-set key).

Rule
----
1. Keep all targets whose `create_date` ≤ query timestamp.
2. Sort them by increasing distance (*timestamp − create_date*).
3. Append all *future* targets (create_date > timestamp) afterwards, again
   ordered by increasing distance.

Thus, users see the most recently available jobs (or candidates) **up to the
query time** first; items that didn’t exist yet are pushed to the bottom.

Usage (mirrors other baselines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```bash
python past_temporal_baseline.py CANDIDATES.csv JOBS.csv TEST.csv OUT.json \
       --mode c2j --seed 42
```

*Missing `create_date` values are filled with the table mean.*
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal (past-only) ranking baseline.")
    p.add_argument("candidates", type=Path)
    p.add_argument("jobs", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def impute_mean_datetime(series: pd.Series) -> pd.Series:
    if series.isna().all():
        raise ValueError("All create_date values missing; cannot impute mean.")
    mean_ns = series.dropna().astype("int64").mean()
    mean_ts = pd.to_datetime(int(mean_ns))
    return series.fillna(mean_ts)


def load_entities(path: Path, id_col: str) -> Dict[str, pd.Timestamp]:
    df = pd.read_csv(path, dtype=str)
    needed = {id_col, "create_date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_date"] = impute_mean_datetime(df["create_date"])

    return dict(zip(df[id_col].astype(str), df["create_date"]))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    cand_dates = load_entities(args.candidates, "candidate_id")
    job_dates = load_entities(args.jobs, "job_id")

    if args.mode == "c2j":
        query_col = "candidate_id"
        target_pool_ids = np.array(list(job_dates.keys()))
        target_dates = np.array([job_dates[j] for j in target_pool_ids], dtype="datetime64[ns]")
    else:
        query_col = "job_id"
        target_pool_ids = np.array(list(cand_dates.keys()))
        target_dates = np.array([cand_dates[c] for c in target_pool_ids], dtype="datetime64[ns]")

    # Load test set
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    for col in [query_col, "timestamp"]:
        if col not in test_df.columns:
            raise ValueError(f"Test CSV missing required column '{col}'")

    output: Dict[str, List[str]] = {}

    # Vectorised ranking
    for qid, ts_str in zip(test_df[query_col].astype(str), test_df["timestamp"].astype(str)):
        q_timestamp = pd.to_datetime(ts_str, errors="coerce")
        if pd.isna(q_timestamp):
            raise ValueError(f"Unparseable timestamp: {ts_str}")

        # Get target distances and masks
        deltas = (q_timestamp.to_datetime64() - target_dates).astype("timedelta64[s]")
        seconds = deltas.astype(np.int64)
        past_mask = seconds >= 0  # non-negative means target created before or at timestamp

        past_indices = np.where(past_mask)[0]
        future_indices = np.where(~past_mask)[0]

        # Sort each group by ascending abs(delta)
        past_order = past_indices[np.argsort(seconds[past_indices], kind="mergesort")]
        future_order = future_indices[np.argsort(np.abs(seconds[future_indices]), kind="mergesort")]

        # Resolve ties inside groups by shuffle
        def shuffle_ties(order_idx: np.ndarray):
            out: List[int] = []
            i = 0
            while i < len(order_idx):
                j = i + 1
                while j < len(order_idx) and seconds[order_idx[j]] == seconds[order_idx[i]]:
                    j += 1
                block = order_idx[i:j].tolist()
                rng.shuffle(block)
                out.extend(block)
                i = j
            return out

        ranked_ids = [*target_pool_ids[shuffle_ties(past_order)].tolist(),
                       *target_pool_ids[shuffle_ties(future_order)].tolist()]

        output[json.dumps([qid, ts_str])] = ranked_ids

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Past-temporal baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
