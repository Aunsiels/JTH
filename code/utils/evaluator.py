#!/usr/bin/env python3
"""
evaluate.py – Open-set evaluator
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os.path
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import pandas as pd
import numpy as np

Key = Tuple[Any, str]  # (id, timestamp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str(x):
    """Robustly convert *anything* to a plain str (NaNs → '')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)

def _parse_id(x):
    """Robustly parse ID to int if possible, else str (NaNs → '')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return int(x)
    except (ValueError, TypeError):
        return str(x)


def _parse_key(key: str) -> Key:
    """Parse a JSON key and force both elements to str."""
    try:
        val = json.loads(key)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            date_str = _to_str(val[1])
            if "T" in date_str:
                date_str = date_str.split("T")[0]
            return _parse_id(val[0]), date_str
    except Exception:
        pass
    parts = key.strip("()[]{} ").split(",")
    if len(parts) == 2:
        return _parse_id(parts[0].strip("'\" ")), _to_str(parts[1].strip("'\" "))
    raise ValueError(f"Cannot parse JSON key: {key}")


def load_gold(path: Path, mode: str) -> Dict[Key, Any]:
    df = pd.read_csv(path, dtype=str)
    exp = ["candidate_id", "job_id", "timestamp"]
    if df.columns.tolist()[:3] != exp:
        raise ValueError(f"Gold CSV must have columns: {', '.join(exp)}")

    gold: Dict[Key, Any] = {}
    for cand, job, ts in df.itertuples(index=False):
        cand = _parse_id(cand)
        job = _parse_id(job)
        ts = _to_str(ts)
        if mode == "c2j":
            gold[(cand, ts)] = job
        else:
            gold[(job, ts)] = cand
    return gold


def load_preds(path: Path) -> Dict[Key, np.ndarray]:
    is_pickle = path.name.endswith(".pkl") or path.name.endswith(".pkl.gz")
    open_func = gzip.open if path.suffix == ".gz" else open
    mode = "rb" if is_pickle else "rt"
    kwargs = {} if is_pickle else {"encoding": "utf-8"}
    
    int_cache = {}
    def cached_int(x_str):
        val = int(x_str)
        return int_cache.setdefault(val, val)

    with open_func(path, mode, **kwargs) as fh:
        if is_pickle:
            try:
                data = pickle.load(fh)
            except Exception as e:
                print("Error decoding pickle", path)
                raise e
        else:
            try:
                data = json.load(fh, parse_int=cached_int)
            except json.decoder.JSONDecodeError as e:
                print("Error decoding", path)
                raise e

    items_ = {}
    for k in list(data.keys()):
        v = data.pop(k)
        try:
            arr = np.array(v, dtype=np.int64)
            if arr.size > 0 and arr.max() <= 2147483647 and arr.min() >= -2147483648:
                arr = arr.astype(np.int32)
        except (ValueError, TypeError, OverflowError):
            arr = np.array([_parse_id(i) for i in v], dtype=object)
        items_[_parse_key(k)] = arr
    return items_

def load_dates(path: Path, id_col: str) -> Dict[Any, str]:
    df = pd.read_csv(path, dtype=str, usecols=[id_col, "create_date"])
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dates = {}
    for i, d in zip(df[id_col], df["create_date"].fillna("")):
        dates[_parse_id(i)] = d
    return dates

def load_initial_known_queries(path: Path, mode: str) -> set[str]:
    """Load initial set of known query IDs from the training data."""
    query_col = "candidate_id" if mode == "c2j" else "job_id"
    try:
        df = pd.read_csv(path, dtype=str, usecols=[query_col])
    except ValueError:
        raise ValueError(f"Train CSV {path} is missing column: {query_col}")
    return set(_parse_id(x) for x in df[query_col].dropna().unique())

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(gold: Dict[Key, Any], preds: Dict[Key, np.ndarray], ks: Sequence[int], target_dates: Dict[Any, str], initial_known_queries: set):
    ks = sorted(set(ks))
    groups = ["all", "warm", "cold"]
    sums = {g: {m: {k: 0.0 for k in ks} for m in ("prec", "rec", "f1", "ndcg", "map", "hit")} for g in groups}
    mrr_total = {g: 0.0 for g in groups}
    n_queries = {g: 0 for g in groups}
    neg_better = {g: 0 for g in groups}

    known_queries = set(initial_known_queries)
    sorted_keys = sorted(gold.keys(), key=lambda k: k[1])
    # print(sorted(preds.keys()))

    for key in sorted_keys:
        true_item = gold[key]
        query_id, ts = key
        
        raw_pred_arr = preds.get(key)
        if raw_pred_arr is None:
            raw_pred_list = []
        elif isinstance(raw_pred_arr, np.ndarray):
            raw_pred_list = raw_pred_arr.tolist()
        else:
            raw_pred_list = raw_pred_arr

        pred_len = 0
        rank = None
        for tgt in raw_pred_list:
            t_date = target_dates.get(tgt, "")
            if not t_date or t_date <= ts:
                pred_len += 1
                if tgt == true_item and rank is None:
                    rank = pred_len
        # pred_len == 0 and query_id > 10000:
        #   print("Pred len", pred_len, "for query", query_id, "at time", ts, key in preds, type(key[0]))
        #   exit(1)


        is_warm = query_id in known_queries
        group_list = ["all", "warm" if is_warm else "cold"]
        known_queries.add(query_id)

        for g in group_list:
            n_queries[g] += 1

        rand_rank = random.randrange(pred_len) + 1 if pred_len else None

        for g in group_list:
            if rank:
                mrr_total[g] += 1 / rank
            if rand_rank is not None:
                if rank is None or rand_rank < rank:
                    neg_better[g] += 1
            for k in ks:
                if rank and rank <= k:
                    p = 1 / k
                    r = 1.0
                    f1 = 2 * p * r / (p + r)
                    ndcg = 1 / math.log2(rank + 1)
                    ap = 1 / rank
                    hit = 1.0
                else:
                    p = r = f1 = ndcg = ap = hit = 0.0
                sums[g]["prec"][k] += p
                sums[g]["rec"][k] += r
                sums[g]["f1"][k] += f1
                sums[g]["ndcg"][k] += ndcg
                sums[g]["map"][k] += ap
                sums[g]["hit"][k] += hit

    metric_dfs = {}
    neg_stats_all = {}
    
    for g in groups:
        if n_queries[g] > 0:
            metric_dfs[g] = pd.DataFrame({
                "K": ks,
                "Precision@K": [sums[g]['prec'][k] / n_queries[g] for k in ks],
                "Recall@K": [sums[g]['rec'][k] / n_queries[g] for k in ks],
                "F1-score@K": [sums[g]['f1'][k] / n_queries[g] for k in ks],
                "NDCG@K": [sums[g]['ndcg'][k] / n_queries[g] for k in ks],
                "MAP@K": [sums[g]['map'][k] / n_queries[g] for k in ks],
                "HitRate@K": [sums[g]['hit'][k] / n_queries[g] for k in ks],
            })
            neg_pr = neg_rc = neg_f1 = (n_queries[g] - neg_better[g]) / n_queries[g]
            neg_stats_all[g] = pd.Series({
                "Negative sample Precision": neg_pr,
                "Negative sample Recall": neg_rc,
                "Negative sample F1": neg_f1,
            })
        else:
            metric_dfs[g] = None
            neg_stats_all[g] = None

    mrrs = {g: (mrr_total[g] / n_queries[g] if n_queries[g] > 0 else 0.0) for g in groups}
    return metric_dfs, neg_stats_all, mrrs, n_queries


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Evaluate recommendations against a gold standard (string‑safe IDs)")
    ap.add_argument("gold", type=Path)
    ap.add_argument("preds", type=Path)
    ap.add_argument("--train", type=Path, required=True, help="Training data to identify initial warm entities.")
    ap.add_argument("--candidates", type=Path, required=True, help="Candidates CSV to get create_date")
    ap.add_argument("--jobs", type=Path, required=True, help="Jobs CSV to get create_date")
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    already_processed = set()
    if args.out:
        if os.path.isfile(args.out):
            df = pd.read_csv(args.out)
        else:
            df = pd.DataFrame()
        if "file" in df.columns:
            for f in df["file"]:
                already_processed.add(str(f))
    if str(args.preds) in already_processed:
        print(args.preds, "was already processed.")
        exit(0)

    random.seed(args.seed)

    # For filtering predictions based on target create_date
    cand_dates = load_dates(args.candidates, "candidate_id")
    job_dates = load_dates(args.jobs, "job_id")
    target_dates = job_dates if args.mode == "c2j" else cand_dates

    initial_known_queries = load_initial_known_queries(args.train, args.mode)

    gold = load_gold(args.gold, args.mode)
    preds = load_preds(args.preds)

    metric_dfs, neg_stats_all, mrrs, n_queries = compute_metrics(gold, preds, args.k, target_dates, initial_known_queries)

    print("\n=== Global metric ===")
    for g in ["all", "warm", "cold"]:
        print(f"MRR ({g}):", f"{mrrs[g]:.6f} (N={n_queries[g]})")

    for g in ["all", "warm", "cold"]:
        if metric_dfs[g] is not None:
            print(f"\n=== Ranking metrics (@K) - {g.upper()} ===")
            print(metric_dfs[g].to_string(index=False, float_format="{:.6f}".format))

    for g in ["all", "warm", "cold"]:
        if neg_stats_all[g] is not None:
            print(f"\n=== Negative-sample comparison - {g.upper()} ===")
            for k, v in neg_stats_all[g].items():
                print(f"{k}: {v:.6f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if os.path.isfile(args.out):
            df = pd.read_csv(args.out)
        else:
            df = pd.DataFrame()
        new_row = {"file": args.preds,
                   "mode": args.mode,
                   "date": datetime.fromtimestamp(os.path.getmtime(args.preds)).strftime("%Y-%m-%d:%H:%M:%S"),
                   "mrr": mrrs["all"]}

        for g in ["warm", "cold"]:
            new_row[f"mrr-{g}"] = mrrs[g]

        for g in ["all", "warm", "cold"]:
            if neg_stats_all[g] is not None:
                for k_stat, v in neg_stats_all[g].items():
                    key_name = k_stat if g == "all" else f"{k_stat}-{g}"
                    new_row[key_name] = v

        for g in ["all", "warm", "cold"]:
            if metric_dfs[g] is not None:
                for _, row in metric_dfs[g].iterrows():
                    k = row["K"]
                    for column in metric_dfs[g].columns:
                        if column == "K":
                            continue
                        col_name = column.replace("K", str(int(k)))
                        if g != "all":
                            col_name += f"-{g}"
                        new_row[col_name] = row[column]

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(args.out, index=False)
        print("Report written to", args.out)


if __name__ == "__main__":
    main()
