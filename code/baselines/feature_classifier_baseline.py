#!/usr/bin/env python3
"""
feature_classifier_baseline.py  (v2)
====================================
Classifier‑based ranking baseline with **rich hand‑crafted features** drawn from
*all* available columns.  Two optional ablations:

* `--use_temporal_features` – keep the old temporal trio (default **off**).
* `--use_llm_features`     – include LLM‑derived columns (default **on**).

If `--use_llm_features` is **not** passed, every llm_* column is ignored and
its associated pair‑features are set to 0.

CLI
---
```bash
python feature_classifier_baseline.py CANDIDATES.csv JOBS.csv \
       TRAIN.csv TEST.csv OUT.json \
       --mode c2j --neg_ratio 1.0 --model forest \
       --use_temporal_features --use_llm_features
```
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Feature‑based recommendation baseline with optional temporal/LLM signals")
    p.add_argument("candidates", type=Path)
    p.add_argument("jobs", type=Path)
    p.add_argument("train", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    p.add_argument("--neg_ratio", type=float, default=1.0)
    p.add_argument("--model", choices=["logreg", "tree", "forest"], default="forest")
    p.add_argument("--use_temporal_features", action="store_true")
    p.add_argument("--use_llm_features", action="store_true", default=False)
    p.add_argument("--remove_human_features", action="store_true", default=False)
    p.add_argument("--pca", action="store_true", default=False)
    p.add_argument("--no_pred", action="store_true", default=False)
    p.add_argument("--shap_png", type=Path, help="Optional path to save SHAP waterfall plot")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parse_list(s: str | float | None) -> set[str]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return set()
    return {tok.strip().lower() for tok in str(s).split(";") if tok.strip()}


def parse_bool(val) -> bool:
    if isinstance(val, str):
        val = val.strip().lower()
        if val in {"true", "1", "yes"}: return True
        if val in {"false", "0", "no", "nan", ""}: return False
    try:
        return bool(int(val))
    except Exception:
        return False


def impute_mean_datetime(series: pd.Series) -> pd.Series:
    if series.isna().all():
        series = series.fillna(pd.Timestamp("1970-01-01"))
    mean_ns = series.dropna().astype("int64").mean()
    return series.fillna(pd.to_datetime(int(mean_ns)))


def fill_with_mode(df: pd.DataFrame, col: str):
    ser = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")
    mode_val = ser[ser != ""].mode(dropna=True)
    mode_val = mode_val.iloc[0] if not mode_val.empty else ""
    df[col] = ser.replace("", mode_val)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

CAND_LIST_COLS = {
    "skills", "expertise_area", "job_category", "llm_languages_spoken",
    "llm_hard_skills", "llm_soft_skills", "llm_programming_languages",
    "llm_tools_technologies", "llm_certifications", "llm_values",
    "llm_industry_domains", "llm_expertise_area", "llm_job_category",
    "technical_skills", "soft_skills", "llm_hobbies"
}
JOB_LIST_COLS = {
    "skills", "expertise_area", "job_category", "contract_type",
    "llm_required_languages_spoken", "llm_hard_skills", "llm_soft_skills",
    "llm_programming_languages", "llm_tools_technologies", "llm_certifications",
    "llm_company_values", "llm_industry_domains", "llm_expertise_area",
    "llm_job_category"
}

SENIORITY_MAP = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}
DIPLOMA_RANK = {"high school": 0, "bachelor": 1, "master": 2, "mba": 2, "phd": 3}


def load_candidates(path: Path, use_llm: bool, remove_human_features: bool) -> Dict[str, dict]:
    df = pd.read_csv(path, dtype=str)
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_date"] = impute_mean_datetime(df["create_date"])
    # mode fill for core str cols
    core_cols = ["skills", "contract_type", "expertise_area", "job_category", "years_experience"]
    for c in core_cols:
        if c in df.columns:
            fill_with_mode(df, c)
    store = {}
    for r in df.itertuples(index=False):
        rec = {
            "date": r.create_date,
            "skills": parse_list(r.skills),
            "expertise": parse_list(r.expertise_area),
            "job_cat": parse_list(r.job_category),
            "years": str(r.years_experience).lower(),
            "contract": parse_list(r.contract_type),
            "salary": _safe_int(getattr(r, "actual_salary", None)),
            "daily": _safe_int(getattr(r, "actual_daily_salary", None)),
            "zipcode": str(getattr(r, "zipcode", ""))[:2],
            "source": r.source,
            "langs": parse_list(getattr(r, "llm_languages_spoken", "")),
            "diploma": DIPLOMA_RANK.get(str(getattr(r, "llm_highest_diploma", "")).lower(), -1),
            "top_university": parse_bool(getattr(r, "llm_from_top_university", False)),
            "multi_degree": parse_bool(getattr(r, "llm_multiple_degrees", False)),
            "ongoing_education": parse_bool(getattr(r, "llm_ongoing_education", False)),
            "years_experience": parse_bool(getattr(r, "llm_years_of_work_experience", False)),
            "previous_positions": _safe_int(getattr(r, "llm_number_of_previous_positions", 0)),
            "number_of_unique_employers": _safe_int(getattr(r, "llm_number_of_unique_employers", 0)),
            "mgmt_exp": parse_bool(getattr(r, "llm_management_experience", False)),
            "years_of_management_roles": getattr(r, "llm_years_of_management_roles", ""),
            "industry": parse_list(getattr(r, "llm_industry_domains", "")),
            "entrepreneurial_experience": parse_bool(getattr(r, "llm_entrepreneurial_experience", False)),
            "startup_exp": parse_bool(getattr(r, "llm_startup_experience", False)),
            "large_exp": parse_bool(getattr(r, "llm_large_company_experience", False)),
            "freelance_exp": parse_bool(getattr(r, "llm_freelance_experience", False)),
            "contract_exp": parse_bool(getattr(r, "llm_contract_experience", False)),
            "intl_exp": parse_bool(getattr(r, "llm_international_work_experience", False)),
            "remote_exp": parse_bool(getattr(r, "llm_remote_work_experience", False)),
            "average_job_duration_months": getattr(r, "llm_average_job_duration_months", 0),
            "number_of_career_gaps": _safe_int(getattr(r, "llm_number_of_career_gaps", 0)),
            "leader_exp": parse_bool(getattr(r, "llm_leadership_experience", False)),
            "hard_skills": parse_list(getattr(r, "llm_hard_skills", "")),
            "soft_skills": parse_list(getattr(r, "llm_soft_skills", "")),
            "programming_languages": parse_list(getattr(r, "llm_programming_languages", "")),
            "tools_technologies": parse_list(getattr(r, "llm_tools_technologies", "")),
            "certifications": parse_list(getattr(r, "llm_certifications", "")),
            "volunteer_experience": parse_bool(getattr(r, "llm_volunteer_experience", False)),
            "has_publications": parse_bool(getattr(r, "llm_has_publications", False)),
            "has_patents": parse_bool(getattr(r, "llm_has_patents", False)),
            "has_portfolio_or_github_or_website": parse_bool(getattr(r, "llm_has_portfolio_or_github_or_website", False)),
            "has_linkedin": parse_bool(getattr(r, "llm_has_linkedin", False)),
            "participated_in_competitions": parse_bool(getattr(r, "llm_participated_in_competitions", False)),
            "expertise_area": parse_list(getattr(r, "llm_expertise_area", "")),
            "job_category": parse_list(getattr(r, "llm_job_category", "")),
            "estimated_salary_eur_per_year": parse_bool(getattr(r, "llm_estimated_salary_eur_per_year", False)),
            "client_role": parse_bool(getattr(r, "llm_client_facing_role", False)),
            "values": parse_list(getattr(r, "llm_values", "")),
            "senior": SENIORITY_MAP.get(str(getattr(r, "llm_seniority_level", "")).lower(), -1),
        }
        if not use_llm:
            # zero‑out llm‑only fields
            for k in [
                "industry", "values", "langs", "senior", "diploma", "remote_exp",
                "mgmt_exp", "leader_exp", "startup_exp", "large_exp", "freelance_exp",
                "contract_exp", "intl_exp", "client_role", "top_university",
                "multi_degree", "ongoing_education", "years_experience", "previous_positions",
                "number_of_unique_employers", "years_of_management_roles",
                "entrepreneurial_experience", "average_job_duration_months",
                "number_of_career_gaps", "volunteer_experience", "has_publications",
                "has_patents", "has_portfolio_or_github_or_website", "has_linkedin",
                "participated_in_competitions", "hard_skills", "soft_skills",
                "programming_languages", "tools_technologies", "certifications",
                "estimated_salary_eur_per_year", "job_category", "expertise_area"]:
                rec[k] = set() if isinstance(rec[k], set) else 0
        if remove_human_features:
            for k in [
                "skills", "expertise", "job_cat", "years", "contract", "salary", "daily", "zipcode", "source"
            ]:
                rec[k] = set() if isinstance(rec[k], set) else 0
        store[str(r.candidate_id)] = rec
    return store


def load_jobs(path: Path, use_llm: bool, remove_human_features: bool) -> Dict[str, dict]:
    df = pd.read_csv(path, dtype=str)
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_date"] = impute_mean_datetime(df["create_date"])
    for c in ["skills", "contract_type", "expertise_area", "job_category", "years_experience"]:
        if c in df.columns:
            fill_with_mode(df, c)
    store = {}
    for r in df.itertuples(index=False):
        rec = {
            "date": r.create_date,
            "job_cat": parse_list(r.job_category),
            "skills": parse_list(r.skills),
            "contract": parse_list(r.contract_type),
            "expertise": parse_list(r.expertise_area),
            "years": str(r.years_experience).lower(),
            "zipcode": str(getattr(r, "zipcode", ""))[:2],
            "salary": _safe_int(getattr(r, "salary", None)),
            "daily": _safe_int(getattr(r, "daily_rate", None)),
            "remote": parse_bool(getattr(r, "llm_remote_possible", False)),
            "industry": parse_list(getattr(r, "llm_industry_domains", "")),
            "values": parse_list(getattr(r, "llm_company_values", "")),
            "senior_req": SENIORITY_MAP.get(str(getattr(r, "llm_seniority_level", "")).lower(), -1),
            "estimated_salary": _safe_int(getattr(r, "llm_salary_estimation_eur_per_year", 0)),
            "langs_req": parse_list(getattr(r, "llm_required_languages_spoken", "")),
            "required_years_of_work_experience": parse_bool(getattr(r, "llm_required_years_of_work_experience", False)),
            "mgmt_req": parse_bool(getattr(r, "llm_required_management_experience", False)),
            "startup": parse_bool(getattr(r, "llm_is_startup", False)),
            "large": parse_bool(getattr(r, "llm_is_large_company", False)),
            "freelance_req": parse_bool(getattr(r, "llm_required_freelance_experience", False)),
            "contract_req": parse_bool(getattr(r, "llm_required_contract_experience", False)),
            "intl_req": parse_bool(getattr(r, "llm_required_international_work_experience", False)),
            "leader_req": parse_bool(getattr(r, "llm_required_leadership_experience", False)),
            "hard_skills": parse_list(getattr(r, "llm_hard_skills", "")),
            "soft_skills": parse_list(getattr(r, "llm_soft_skills", "")),
            "programming_languages": parse_list(getattr(r, "llm_programming_languages", "")),
            "tools_technologies": parse_list(getattr(r, "llm_tools_technologies", "")),
            "certifications": parse_list(getattr(r, "llm_certifications", "")),
            "expertise_area": parse_list(getattr(r, "llm_expertise_area", "")),
            "job_category": parse_list(getattr(r, "llm_job_category", "")),
            "diploma_req": DIPLOMA_RANK.get(str(getattr(r, "llm_required_lowest_diploma", "")).lower(), -1),
            "client_role": parse_bool(getattr(r, "llm_client_facing_role", False)),
        }
        if not use_llm:
            for k in [
                "industry", "values", "langs_req", "senior_req", "diploma_req", "remote",
                "mgmt_req", "leader_req", "startup", "large", "freelance_req", "contract_req",
                "intl_req", "client_role", "hard_skills", "soft_skills", "programming_languages",
                "tools_technologies", "certifications", "job_category", "expertise_area",
                "required_years_of_work_experience", "estimated_salary"
            ]:
                rec[k] = set() if isinstance(rec[k], set) else 0
        if remove_human_features:
            for k in [
                "skills", "expertise", "job_cat", "years", "contract", "salary", "daily", "zipcode",
            ]:
                rec[k] = set() if isinstance(rec[k], set) else 0
        store[str(r.job_id)] = rec
    return store


def _safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _ratio(overlap: int, base: int) -> float:
    """Safe overlap / base ratio in [0,1]."""
    return overlap / base if base else 0.0


def _salary_gap(c_val, j_val) -> float:
    """Normalised salary / estimation gap."""
    if c_val and j_val:
        return (c_val - j_val) / (abs(j_val) + 1)
    return 0.0

# bucket → numeric mid-point mappings
_YEAR_MGMT = {"0-2": 1, "2-5": 3.5, "5-10": 7.5, "10-15": 12.5,
              "15-20": 17.5, "20-30": 25, ">=30": 30}
_DURATION = {"0-6": 3, "6-12": 9, "12-24": 18, "24-36": 30,
             "36-60": 48, ">=60": 60}


def _bucket_num(val: str | float | None, mapping: dict[str, float]) -> float:
    """Convert bucket strings ('2-5', '>=30') to numeric mid-points."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return mapping.get(str(val).strip(), 0.0)


FEATURE_NAMES = [
    # overlaps
    "Skill Coverage", "Hard Skill Coverage", "Soft Skill Coverage", "Prog. Language Coverage", "Tool Coverage",
    "Certification Coverage", "Expertise Area Coverage", "Job Category Coverage",
    "Expertise Area LLM Coverage", "Job Category LLM Coverage",
    "Industry Coverage", "Values Coverage",
    "Language Coverage",
    # modality flags
    "Contract Match", "Remote Match", "Freelance Match", "With Contract Match",
    # profile fit
    "Years Bucket Match", "Management Match", "Leadership Match", "Startup Match", "Large Company Match",
    "Internation Exp. Match", "Client Role Match", "Seniority Gap", "Diploma Match",
    # compensation
    "Salary Gap", "Daily Gap", "Est. Salary Gap",
    # geo & source
    "Same Zip",
    # work‑exp req
    "Work Years Match",
    # candidate llm flags (10)
    "Top University", "Multi-degree", "Ongoing Study", "Entrepreneurial Exp.", "Volunteer",
    "Has Publications", "Has Patents", "Visible Portfolio", "Has Linkedin", "Entered Competitions",
    # numeric counts (5)
    "N. Prev. Positions", "N. Unique Employers", "Years Mgmt Roles",
    "Avg Job Duration", "N. Career Gaps",
    # temporal trio
    "Entity Delta Days", "Rec. Delta Days", "Existed Before"
]


def build_features(c: dict, j: dict, ts: pd.Timestamp, mode: str, temporal: bool) -> Tuple[float, ...]:
    """Return dense numeric vector for a candidate–job pair."""
    f: List[float] = []

    # ---------- list overlaps ----------
    f += [
        _ratio(len(c["skills"] & j["skills"]), len(j["skills"])),
        _ratio(len(c.get("hard_skills", set()) & j.get("hard_skills", set())), len(j.get("hard_skills", set()))),
        _ratio(len(c.get("soft_skills", set()) & j.get("soft_skills", set())), len(j.get("soft_skills", set()))),
        _ratio(len(c.get("programming_languages", set()) & j.get("programming_languages", set())),
               len(j.get("programming_languages", set()))),
        _ratio(len(c.get("tools_technologies", set()) & j.get("tools_technologies", set())),
               len(j.get("tools_technologies", set()))),
        _ratio(len(c.get("certifications", set()) & j.get("certifications", set())),
               len(j.get("certifications", set()))),
        _ratio(len(c["expertise"] & j["expertise"]), len(j["expertise"])),
        _ratio(len(c["job_cat"] & j["job_cat"]), len(j["job_cat"])),
        _ratio(len(c["expertise_area"] & j["expertise_area"]), len(j["expertise_area"])),
        _ratio(len(c["job_category"] & j["job_category"]), len(j["job_category"])),
        _ratio(len(c.get("industry", set()) & j.get("industry", set())), len(j.get("industry", set()))),
        _ratio(len(c.get("values", set()) & j.get("values", set())), len(j.get("values", set()))),
        _ratio(len(c.get("langs", set()) & j.get("langs_req", set())), len(j.get("langs_req", set()))),
    ]

    # ---------- modality flags ----------
    f += [
        float(bool(j["contract"] & c["contract"])),
        float(j.get("remote", False) and c.get("remote_exp", False)),
        float(j.get("freelance_req", 0) and c.get("freelance_exp", False)),
        float(j.get("contract_req", 0) and c.get("contract_exp", False)),
    ]

    # ---------- profile fit ----------
    f += [
        float(c["years"] == j["years"]),
        float(j.get("mgmt_req", 0) and c.get("mgmt_exp", 0)),
        float(j.get("leader_req", 0) and c.get("leader_exp", 0)),
        float(j.get("startup", 0) and c.get("startup_exp", 0)),
        float(j.get("large", 0) and c.get("large_exp", 0)),
        float(j.get("intl_req", 0) and c.get("intl_exp", 0)),
        float(j.get("client_role", 0) == c.get("client_role", 0)),
    ]
    senior_gap = 0.0
    if j.get("senior_req", -1) >= 0 and c.get("senior", -1) >= 0:
        senior_gap = (c["senior"] - j["senior_req"]) / 3.0
    f.append(senior_gap)
    f.append(float(c.get("diploma", -1) >= j.get("diploma_req", -1) >= 0))

    # ---------- compensation ----------
    f.append(_salary_gap(c.get("salary"), j.get("salary")))
    f.append(_salary_gap(c.get("daily"), j.get("daily")))
    # newly added: estimated‑salary gap
    f.append(_salary_gap(c.get("estimated_salary_eur_per_year"), j.get("estimated_salary")))

    # ---------- geography & source ----------
    f.append(float(c["zipcode"] == j["zipcode"]))

    # ---------- years‑of‑work‑experience requirement satisfied ----------
    f.append(float(j.get("required_years_of_work_experience", "") == c.get("years_experience", "")))

    # ---------- candidate‑only LLM flags ----------
    for flag in [
        "top_university", "multi_degree", "ongoing_education", "entrepreneurial_experience",
        "volunteer_experience", "has_publications", "has_patents",
        "has_portfolio_or_github_or_website", "has_linkedin", "participated_in_competitions"]:
        f.append(float(c.get(flag, 0)))

    # numeric counts & bucketed numerics
    f += [
        float(c.get("previous_positions", 0) or 0),
        float(c.get("number_of_unique_employers", 0) or 0),
        _bucket_num(c.get("years_of_management_roles"), _YEAR_MGMT),
        _bucket_num(c.get("average_job_duration_months"), _DURATION),
        float(c.get("number_of_career_gaps", 0) or 0),
    ]

    # ---------- temporal ----------
    if temporal:
        delta_entities = abs((c["date"] - j["date"]).days)
        tgt = j["date"] if mode == "c2j" else c["date"]
        rec_delta = abs((ts - tgt).days)
        existed = float(tgt <= ts)
    else:
        delta_entities = rec_delta = existed = 0.0
    f += [delta_entities, rec_delta, existed]

    return tuple(f)


# ---------------------------------------------------------------------------
# Negative sampling (unchanged)
# ---------------------------------------------------------------------------

def sample_negs(c_id: str, j_id: str, ts: pd.Timestamp, cand_ids: List[str], job_ids: List[str],
                mode: str, n: int, rng: random.Random):
    negs = []
    while len(negs) < n:
        if mode == "c2j":
            nj = rng.choice(job_ids)
            if nj == j_id: continue
            negs.append((c_id, nj, ts))
        else:
            nc = rng.choice(cand_ids)
            if nc == c_id: continue
            negs.append((nc, j_id, ts))
    return negs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    rng = random.Random(args.seed)
    if os.path.isfile(args.out) and not args.no_pred:
        print("Output file already exists:", args.out)
        exit(1)

    candidates = load_candidates(args.candidates, args.use_llm_features, args.remove_human_features)
    jobs = load_jobs(args.jobs, args.use_llm_features, args.remove_human_features)
    cand_ids, job_ids = list(candidates.keys()), list(jobs.keys())

    # ----------- Build training matrix -----------
    train_df = pd.read_csv(args.train, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")

    X, y = [], []
    for r in train_df.itertuples(index=False):
        cid, jid, ts = str(r.candidate_id), str(r.job_id), r.timestamp
        X.append(build_features(candidates[cid], jobs[jid], ts, args.mode, args.use_temporal_features))
        y.append(1)
        for nc, nj, nts in sample_negs(cid, jid, ts, cand_ids, job_ids, args.mode, int(args.neg_ratio), rng):
            X.append(build_features(candidates[nc], jobs[nj], nts, args.mode, args.use_temporal_features))
            y.append(0)

    X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y)
    df_feat = pd.DataFrame(X_arr, columns=FEATURE_NAMES)
    df_feat.to_csv("features.csv", index=False)

    features_to_drop = []
    desc = df_feat.describe()
    for col in desc.columns:
        if desc[col]["std"] == 0:
            features_to_drop.append(col)
    df_feat = df_feat.drop(features_to_drop, axis=1)
    print("Dropped features:", ", ".join(features_to_drop))


    # ----------- Train classifier -----------
    if args.model == "logreg":
        clf = LogisticRegression(max_iter=10000, random_state=args.seed)
    elif args.model == "tree":
        clf = DecisionTreeClassifier(random_state=args.seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed)

    scaler = StandardScaler()
    if args.pca:
        pca = PCA()
        pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", clf)])
    else:
        pipe = Pipeline(steps=[("scaler", scaler), ("logistic", clf)])
    pipe.fit(df_feat, y_arr)


    if args.shap_png is not None:
        try:
            print("Model coefficients:\n")
            coeffs = dict()
            for i in range(df_feat.shape[1]):
                coeffs[df_feat.columns[i]] = clf.coef_[0][i]
            for key, value in sorted(coeffs.items(), key=lambda x: -x[1]):
                print(key, "=", value.round(5))
            import shap, matplotlib
            matplotlib.use("Agg")  # headless save
            if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)):
                explainer = shap.TreeExplainer(pipe.predict, shap.utils.sample(df_feat, 100))
            else:
                explainer = shap.Explainer(pipe.predict, shap.utils.sample(df_feat, 100))
            sv = explainer(df_feat)
            shap.plots.waterfall(sv[20], max_display=20, show=False)
            import matplotlib.pyplot as plt
            plt.savefig(args.shap_png, bbox_inches="tight")
            plt.close()
            print(f"SHAP waterfall saved to {args.shap_png}")
            shap.plots.beeswarm(sv, max_display=20, show=False)
            plt.savefig("bee_" + str(args.shap_png), bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate SHAP plot: {e}", file=sys.stderr)
            raise e

    # ----------- Prediction -----------
    if not args.no_pred:
        test_df = pd.read_csv(args.test, dtype=str)
        if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
            test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

        tgt_ids = job_ids if args.mode == "c2j" else cand_ids
        out: Dict[str, List[str]] = {}

        for r in test_df.itertuples(index=False):
            cid, jid, ts = str(r.candidate_id), str(r.job_id), r.timestamp
            if args.mode == "c2j":
                feats = [build_features(candidates[cid], jobs[tid], ts, "c2j", args.use_temporal_features) for tid in
                         tgt_ids]
                query_id = cid
            else:
                feats = [build_features(candidates[tid], jobs[jid], ts, "j2c", args.use_temporal_features) for tid in
                         tgt_ids]
                query_id = jid
            df_test = pd.DataFrame(np.asarray(feats, float), columns=FEATURE_NAMES)
            df_test = df_test.drop(features_to_drop, axis=1)
            scores = pipe.predict_proba(df_test)[:, 1]
            ranking = [int(tid) if tid.lstrip('-').isdigit() else tid for _, tid in sorted(zip(scores, tgt_ids), key=lambda t: -t[0])]
            q_id = int(query_id) if query_id.lstrip('-').isdigit() else query_id
            out[json.dumps([q_id, ts.strftime("%Y-%m-%d")])] = ranking[:1000]

        args.out.parent.mkdir(parents=True, exist_ok=True)
        is_pickle = args.out.name.endswith(".pkl") or args.out.name.endswith(".pkl.gz")
        open_func = gzip.open if args.out.suffix == ".gz" else open
        mode = "wb" if is_pickle else "wt"
        kwargs = {} if is_pickle else {"encoding": "utf-8"}
        with open_func(args.out, mode, **kwargs) as fh:
            if is_pickle:
                pickle.dump(out, fh)
            else:
                json.dump(out, fh, ensure_ascii=False)

        print(f"Feature classifier baseline written: {args.out} ({len(out)} queries)")


if __name__ == "__main__":
    main()
