#!/usr/bin/env python3
"""
evaluate_recsys.py – Open-set evaluator (string‑safe)
====================================================
*Ensures **all IDs are treated as strings**, preventing silent type mismatches
when numeric IDs are parsed as integers.*

Changes (2025‑05‑07)
--------------------
* Added `_to_str()` utility and enforced `str()` conversion in every place IDs
  or timestamps enter keys/values (`load_gold`, `_parse_key`, and metric loop).
* Logic and output format unchanged (global MRR + @K metrics).
"""
from __future__ import annotations

import argparse
import json
import math
import os.path
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

Key = Tuple[str, str]  # (id, timestamp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str(x):
    """Robustly convert *anything* to a plain str (NaNs → '')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def _parse_key(key: str) -> Key:
    """Parse a JSON key and force both elements to str."""
    try:
        val = json.loads(key)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            date_str = _to_str(val[1])
            if "T" in date_str:
                date_str = date_str.split("T")[0]
            return _to_str(val[0]), date_str
    except Exception:
        pass
    parts = key.strip("()[]{} ").split(",")
    if len(parts) == 2:
        return _to_str(parts[0]), _to_str(parts[1])
    raise ValueError(f"Cannot parse JSON key: {key}")


def load_gold(path: Path, mode: str) -> Dict[Key, str]:
    df = pd.read_csv(path, dtype=str)
    exp = ["candidate_id", "job_id", "timestamp"]
    if df.columns.tolist()[:3] != exp:
        raise ValueError(f"Gold CSV must have columns: {', '.join(exp)}")

    gold: Dict[Key, str] = {}
    for cand, job, ts in df.itertuples(index=False):
        cand, job, ts = map(_to_str, (cand, job, ts))
        if mode == "c2j":
            gold[(cand, ts)] = job
        else:
            gold[(job, ts)] = cand
    return gold


def load_preds(path: Path) -> Dict[Key, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items_ = {_parse_key(k): [_to_str(i) for i in v] for k, v in data.items()}
    return items_


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(gold: Dict[Key, str], preds: Dict[Key, Sequence[str]], ks: Sequence[int]):
    ks = sorted(set(ks))
    sums = {m: {k: 0.0 for k in ks} for m in ("prec", "rec", "f1", "ndcg", "map", "hit")}
    mrr_total = 0.0
    n_queries = 0
    neg_better = 0

    for key, true_item in gold.items():
        true_item = _to_str(true_item)
        pred_list = list(map(_to_str, preds.get(key, [])))
        n_queries += 1
        rank = pred_list.index(true_item) + 1 if true_item in pred_list else None
        if rank:
            mrr_total += 1 / rank
        if pred_list:
            rand_rank = random.randrange(len(pred_list)) + 1
            if rank is None or rand_rank < rank:
                neg_better += 1
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
            sums["prec"][k] += p
            sums["rec"][k] += r
            sums["f1"][k] += f1
            sums["ndcg"][k] += ndcg
            sums["map"][k] += ap
            sums["hit"][k] += hit

    metric_df = pd.DataFrame({
        "K": ks,
        "Precision@K": [sums['prec'][k] / n_queries for k in ks],
        "Recall@K": [sums['rec'][k] / n_queries for k in ks],
        "F1-score@K": [sums['f1'][k] / n_queries for k in ks],
        "NDCG@K": [sums['ndcg'][k] / n_queries for k in ks],
        "MAP@K": [sums['map'][k] / n_queries for k in ks],
        "HitRate@K": [sums['hit'][k] / n_queries for k in ks],
    })

    neg_pr = neg_rc = neg_f1 = (n_queries - neg_better) / n_queries
    neg_stats = pd.Series({
        "Negative sample Precision": neg_pr,
        "Negative sample Recall": neg_rc,
        "Negative sample F1": neg_f1,
    })

    return metric_df, neg_stats, mrr_total / n_queries


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Evaluate recommendations against a gold standard (string‑safe IDs)")
    ap.add_argument("gold", type=Path)
    ap.add_argument("preds", type=Path)
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

    gold = load_gold(args.gold, args.mode)
    preds = load_preds(args.preds)

    metric_df, neg_stats, mrr = compute_metrics(gold, preds, args.k)

    print("\n=== Global metric ===\nMRR:", f"{mrr:.6f}")
    print("\n=== Ranking metrics (@K) ===")
    print(metric_df.to_string(index=False, float_format="{:.6f}".format))
    print("\n=== Negative-sample comparison ===")
    for k, v in neg_stats.items():
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
                   "mrr": mrr}
        for k, v in neg_stats.items():
            new_row[k] = v
        for _, row in metric_df.iterrows():
            k = row["K"]
            for column in metric_df.columns:
                if column == "K":
                    continue
                column_change = column.replace("K", str(int(k)))
                new_row[column_change] = row[column]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(args.out, index=False)
        print("Report written to", args.out)


if __name__ == "__main__":
    main()
