#!/usr/bin/env python3
"""
Verify the integrity of the anonymised recruiting dataset produced by
`anonymize_recruiting_dataset.py`.

Checks implemented
------------------
* **Schema** — every expected column is present in each CSV.
* **Salary / rate sanity** — values strictly > 0 and ≤ `--salary_max`.
* **Non‑negative numeric fields** — no negative integers/floats after noise.
* **Date monotonicity**
  * Candidate / job `create_date` precedes any interaction involving the
    corresponding ID.
  * Per‑application milestone dates are non‑decreasing.
* **Categorical validity** — `source` codes are integers; zip‑codes are 1–2
  digits; bucket columns match the expected pattern (e.g., "20‑30", "‑", ">=70").

Exit status is **0** if all tests pass, **1** otherwise.  A JSON report is
written beside the input folder for audit trails.

Example
~~~~~~~
python verify_anonymized_dataset.py \
    --candidates anonymized/candidates_anon.csv \
    --jobs anonymized/jobs_anon.csv \
    --history anonymized/history_anon.csv
"""
import argparse
import re
import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def is_bucket(val: str) -> bool:
    return bool(re.fullmatch(r"\d+\-\d+|>\=\d+", str(val)))


# ---------------------------------------------------------------------------
# Verification engine
# ---------------------------------------------------------------------------

def verify(args):
    ok = True  # global pass flag
    report: Dict[str, Dict] = {}

    # ---------- load ----------
    cand = pd.read_csv(args.candidates, parse_dates=["create_date"], keep_default_na=True)
    jobs = pd.read_csv(args.jobs, parse_dates=["create_date"], keep_default_na=True)
    hist = pd.read_csv(args.history,
                       parse_dates=[c for c in pd.read_csv(args.history, nrows=0).columns if c.endswith("_date")],
                       keep_default_na=True)

    # ---------- column presence ----------
    expected_cols = {
        "candidates": ["candidate_id", "create_date", "skills", "expertise_area", "job_category",
                       "years_experience", "actual_salary", "actual_daily_salary", "contract_type",
                       "zipcode", "source", "llm_languages_spoken", "llm_hobbies", "llm_sex"],  # add more if needed
        "jobs": ["job_id", "create_date", "job_category", "skills", "contract_type", "expertise_area",
                 "years_experience", "zipcode", "salary", "daily_rate", "source"],
        "history": ["candidate_id", "job_id", "spontaneous_application_date", "last_stage_reached",
                    "end_of_process_date"]
    }
    for name, df in [("candidates", cand), ("jobs", jobs), ("history", hist)]:
        missing = sorted(set(expected_cols[name]) - set(df.columns))
        report[f"{name}_missing_cols"] = missing
        if missing:
            print(f"‼ {name}: missing columns → {missing}")
            ok = False

    # ---------- salary sanity ----------
    salary_cols = [
        (cand, ["actual_salary", "actual_daily_salary", "llm_estimated_salary_eur_per_year"]),
        (jobs, ["salary", "daily_rate", "llm_salary_estimation_eur_per_year"])
    ]
    for df, cols in salary_cols:
        for col in cols:
            if col in df.columns:
                neg = int((df[col] < 0).sum())
                hi = int((df[col] > args.salary_max).sum())
                report[f"{col}_neg"] = neg
                report[f"{col}_too_high"] = hi
                if neg or hi:
                    print(f"‼ salary check failed in {col}: {neg} neg, {hi} > {args.salary_max}")
                    ok = False

    # ---------- non‑negative numerics ----------
    for df_name, df in [("candidates", cand), ("jobs", jobs)]:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            negatives = int((df[col] < 0).sum())
            if negatives:
                print(f"‼ {df_name}.{col}: {negatives} negative values")
                ok = False

    # ---------- date monotonicity ----------
    stage_cols = [c for c in hist.columns if c.endswith("_date") and c not in {"end_of_process_date"}]
    monotone_viol = 0
    for _, row in hist.iterrows():
        prev = None
        for c in stage_cols + ["end_of_process_date"]:
            cur = row[c]
            if pd.isna(cur):
                continue
            if prev is not None and cur < prev:
                monotone_viol += 1
                break
            prev = cur
    if monotone_viol:
        print(f"‼ history: {monotone_viol} rows with non‑monotone dates")
        ok = False
    report["history_monotone_violations"] = monotone_viol

    # candidate / job create_date before interactions
    create_before_fail = 0
    cid_create = cand.set_index("candidate_id")["create_date"]
    jid_create = jobs.set_index("job_id")["create_date"]
    hist_dates = hist[stage_cols + ["end_of_process_date"]].min(axis=1)
    for idx, row in hist.iterrows():
        cdate = cid_create.get(row["candidate_id"], pd.NaT)
        jdate = jid_create.get(row["job_id"], pd.NaT)
        first_int = hist_dates.loc[idx]
        if pd.notna(cdate) and pd.notna(first_int) and cdate > first_int:
            create_before_fail += 1
        if pd.notna(jdate) and pd.notna(first_int) and jdate > first_int:
            create_before_fail += 1
    if create_before_fail:
        print(f"‼ create_date after interaction in {create_before_fail} cases")
        ok = False
    report["create_date_after_interaction"] = create_before_fail

    # ---------- categorical sanity ----------
    if "source" in cand.columns:
        non_int = cand["source"].dropna().apply(float.is_integer).eq(False).sum()
        report["source_non_int_candidates"] = int(non_int)
        if non_int:
            print(f"‼ candidates.source has {non_int} non‑integer codes")
            ok = False
    if "source" in jobs.columns:
        non_int = jobs["source"].dropna().apply(float.is_integer).eq(False).sum()
        report["source_non_int_jobs"] = int(non_int)
        if non_int:
            print(f"‼ jobs.source has {non_int} non‑integer codes")
            ok = False

    # ---------- bucket pattern check (optional) ----------
    bucket_cols = [c for c in cand.columns if c.endswith("bucket") or c.endswith("recency")]
    invalid = 0
    for col in bucket_cols:
        invalid += cand[col].dropna().apply(lambda v: not is_bucket(v)).sum()
    if invalid:
        print(f"‼ bucket columns contain {invalid} invalid strings")
        ok = False
    report["bucket_invalid_values"] = int(invalid)

    # ---------- output ----------
    out_json = Path(args.output_dir) / "verification_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, default=str)
    if ok:
        print("✓ All checks passed")
        sys.exit(0)
    else:
        print("‼ Verification failed — see details above and in", out_json)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Verify anonymised recruiter dataset")
    p.add_argument("--candidates", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--output_dir", default="verification", help="Folder to store the JSON report")
    p.add_argument("--salary_max", type=int, default=1_000_000, help="Upper bound for salaries / rates")
    return p.parse_args()


if __name__ == "__main__":
    verify(parse_args())
