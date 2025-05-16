#!/usr/bin/env python3
"""
binary_splitter.py
------------------
Temporal binary (train/test) split for tabular recommendation datasets.

Usage
-----
$ python binary_splitter.py DATA.csv OUTPUT_DIR [--split 80]

Parameters
~~~~~~~~~~
DATA.csv      CSV file containing at least the columns
              ``candidate_id``, ``job_id``, ``*_date``.
OUTPUT_DIR    Destination directory; will be created if it does not exist.
--split       Percentage of earliest records to assign to the *train* split.
              • If > 1, interpreted as a percentage (80 → 0.80).
              • If ≤ 1, interpreted directly as a fraction (0.8).
              Default: 80 (i.e., 80 %).

Process
~~~~~~~
1. Parse all dates as a datetime, take the min date and sort ascending.
2. Slice the first *split* fraction into *train*, remainder into *test*.
3. Keep only ``candidate_id``, ``job_id``, ``timestamp`` (renamed).
4. Write ``train.csv`` and ``test.csv`` inside *OUTPUT_DIR*.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"candidate_id", "job_id"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal train/test splitter.")
    p.add_argument("data", type=Path, help="Input CSV file")
    p.add_argument("out_dir", type=Path, help="Output directory for splits")
    p.add_argument(
        "--split",
        type=float,
        default=80,
        help="Percentage (or fraction) for the train split (default 80).",
    )
    return p.parse_args(argv)


def validate_fraction(split: float) -> float:
    if split <= 0 or split >= 100:
        raise ValueError("--split must be between 0 and 100 (exclusive)")
    return split / 100 if split > 1 else split


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    frac = validate_fraction(args.split)

    # Read data --------------------------------------------------------------
    df = pd.read_csv(args.data, dtype=str)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")

    for x in df.columns:
        if "_date" in x and "KO" not in x and x != "job_offer_opening_date":
            df[x] = pd.to_datetime(df[x])
    columns_date = [x for x in df.columns if "_date" in x and "KO" not in x and x != "job_offer_opening_date"]
    df_all_dates = df[["candidate_id", "job_id"] + columns_date].copy()

    df_all_dates["min_date"] = df_all_dates[columns_date].min(axis=1)

    # Convert and sort -------------------------------------------------------
    if df_all_dates["min_date"].isna().any():
        bad = df_all_dates[df_all_dates["min_date"].isna()].index[:5].tolist()
        raise ValueError(
            "Some rows have unparseable dates (showing first 5 indices): "
            + ", ".join(map(str, bad))
        )

    df_all_dates = df_all_dates.sort_values("min_date").reset_index(drop=True)

    cutoff = int(len(df_all_dates) * frac)
    train_df = df_all_dates.iloc[:cutoff, :].copy()
    test_df = df_all_dates.iloc[cutoff:, :].copy()

    # Keep desired columns & rename -----------------------------------------
    cols = ["candidate_id", "job_id", "min_date"]
    train_df = train_df[cols].rename(columns={"min_date": "timestamp"})
    test_df = test_df[cols].rename(columns={"min_date": "timestamp"})

    # Output -----------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train split: {len(train_df)} rows  -> {train_path}")
    print(f"Test  split: {len(test_df)} rows  -> {test_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
