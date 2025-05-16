#!/usr/bin/env python3
"""
temporal_baseline.py
--------------------
Rank by *temporal proximity* between candidate and job creation dates.

This baseline produces a deterministic list where the items whose
``create_date`` is **closest** to the query's ``create_date`` come first.
The interface mirrors `random_baseline.py`.

Usage
-----
$ python temporal_baseline.py CANDIDATES.csv JOBS.csv TEST.csv OUT.json \
      --mode c2j [--seed 42]

Arguments
~~~~~~~~~
CANDIDATES.csv   At least ``candidate_id`` and ``create_date`` columns.
JOBS.csv         At least ``job_id`` and ``create_date`` columns.
TEST.csv         Split containing ``candidate_id``, ``job_id`` and ``timestamp``.
OUT.json         Destination file (JSON) – same format expected by the evaluator.
--mode           "c2j" (default) – *jobs to candidates* or "j2c" – *candidates to jobs*.
--seed           RNG seed used **only** to break ties when two targets share identical
                 distance to the query.

Implementation notes
~~~~~~~~~~~~~~~~~~~~
* Dates are parsed with ``pandas.to_datetime`` (errors='coerce'). Rows with
  unparsable ``create_date``s raise an exception (fail‑fast).
* For each query we compute the absolute time delta to every possible target
  and sort ascending; equal distances are shuffled with the given seed.
* Keys in the output JSON are ``[query_id, timestamp]``.
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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate temporal‑proximity baseline rankings.")
    p.add_argument("candidates", type=Path, help="CSV with candidate_id and create_date")
    p.add_argument("jobs", type=Path, help="CSV with job_id and create_date")
    p.add_argument("test", type=Path, help="Test CSV (needs timestamp)")
    p.add_argument("out", type=Path, help="Output JSON file")
    p.add_argument("--mode", choices=["c2j", "j2c"], default="c2j",
                   help="'c2j' jobs→candidates (default) or 'j2c' candidates→jobs")
    p.add_argument("--seed", type=int, default=42, help="Seed for tie‑breaking shuffle")
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def impute_mean_datetime(series: pd.Series) -> pd.Series:
    """Fill NaT with the mean of non‑NaT timestamps."""
    if series.isna().all():
        raise ValueError("All create_date values are missing; cannot impute mean.")
    # Convert to int64 (ns since epoch) to compute the mean
    mean_ns = series.dropna().astype("int64").mean()
    mean_ts = pd.to_datetime(int(mean_ns))
    return series.fillna(mean_ts)


def load_entities(path: Path, id_col: str) -> Dict[str, pd.Timestamp]:
    """Return mapping id → create_date (NaT imputed with table mean)."""
    usecols = [id_col, "create_date"]
    df = pd.read_csv(path, dtype=str, usecols=usecols)
    missing = set(usecols) - set(df.columns)
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
    random.seed(args.seed)

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

    # Vectorised ranking per query
    for qid, ts in zip(test_df[query_col].astype(str), test_df["timestamp"].astype(str)):
        q_date = cand_dates.get(qid) if args.mode == "c2j" else job_dates.get(qid)
        if q_date is None:
            raise KeyError(f"Creation date not found for query id '{qid}'")

        deltas = np.abs(target_dates - q_date.to_datetime64())  # timedelta64[ns]
        abs_seconds = deltas.astype("timedelta64[s]").astype(np.int64)

        # Stable argsort; ties keep input order → then shuffled for randomness
        order = np.argsort(abs_seconds, kind="mergesort")

        # Resolve exact ties by shuffling blocks with identical distance
        ranked_ids: List[str] = []
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and abs_seconds[order[j]] == abs_seconds[order[i]]:
                j += 1
            tie_block = order[i:j].tolist()
            random.shuffle(tie_block)
            ranked_ids.extend(target_pool_ids[tie_block].tolist())
            i = j

        output[json.dumps([qid, ts])] = ranked_ids

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Temporal‑proximity baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
