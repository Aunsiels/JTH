#!/usr/bin/env python3
"""
collab_mf_temporal.py
=====================
Incremental *matrix‑factorization* collaborative‑filtering baseline that walks
through the test set in timestamp order, updating after each batch.

Highlights
~~~~~~~~~~
* **Online learning:** after predicting for every test row at a timestamp, the
  newly revealed interactions are fed to the model with a few SGD steps.
* **Implicit feedback:** every interaction has target value 1. No negatives
  are needed—SGD on observed positives with L2 regularisation.
* **Latent factors stored in Python dicts** so new IDs can be added on the fly.
* **Dot‑product scoring** for ranking.

Usage
-----
```bash
python collab_mf_temporal.py TRAIN.csv TEST.csv OUT.json \
       --mode c2j \
       --dim 32 --init_lr 0.05 --reg 0.01 --epochs 5 --seed 42
```

Parameters
~~~~~~~~~~
--dim        Latent dimension (default 32)
--init_lr    Learning‑rate for initial training (default 0.05)
--reg        L2 regularisation coefficient (default 0.01)
--epochs     Number of passes over interactions:  • for initial fit and  • for
             each incremental batch (default 5)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Matrix‑factorisation model (implicit SGD)
# ---------------------------------------------------------------------------
class MFImplicit:
    def __init__(self, dim: int, lr: float, reg: float, rng: random.Random):
        self.K = dim
        self.lr0 = lr
        self.reg = reg
        self.rng = rng
        self.u_vecs: Dict[str, np.ndarray] = {}
        self.i_vecs: Dict[str, np.ndarray] = {}

    def _init_vec(self) -> np.ndarray:
        return np.array([self.rng.gauss(0, 0.1) for _ in range(self.K)])

    def _u(self, uid: str) -> np.ndarray:
        if uid not in self.u_vecs:
            self.u_vecs[uid] = self._init_vec()
        return self.u_vecs[uid]

    def _i(self, iid: str) -> np.ndarray:
        if iid not in self.i_vecs:
            self.i_vecs[iid] = self._init_vec()
        return self.i_vecs[iid]

    def sgd_epoch(self, interactions: List[Tuple[str, str]], lr: float):
        rng = self.rng
        rng.shuffle(interactions)
        for u_id, i_id in interactions:
            u = self._u(u_id)
            v = self._i(i_id)
            pred = np.dot(u, v)
            err = 1.0 - pred  # implicit: r_ui = 1
            # cache old u for simultaneous update
            u_old = u.copy()
            u += lr * (err * v - self.reg * u)
            v += lr * (err * u_old - self.reg * v)

    def fit(self, interactions: List[Tuple[str, str]], epochs: int):
        for e in range(epochs):
            self.sgd_epoch(interactions, self.lr0)

    def score_items_for_user(self, uid: str, items: List[str]) -> List[float]:
        u = self._u(uid)
        scores = []
        for iid in items:
            v = self._i(iid)
            scores.append(float(np.dot(u, v)))
        return scores

    def score_users_for_item(self, iid: str, users: List[str]) -> List[float]:
        v = self._i(iid)
        scores = []
        for uid in users:
            u = self._u(uid)
            scores.append(float(np.dot(u, v)))
        return scores

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="Temporal MF collaborative filtering baseline.")
    ap.add_argument("train", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=["c2j", "j2c"], default="c2j")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--init_lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5,
                    help="Epochs for initial training and each incremental batch")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)

# ---------------------------------------------------------------------------
# Main online loop
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # --- Load datasets ---
    train_df = pd.read_csv(args.train, dtype=str)
    test_df = pd.read_csv(args.test, dtype=str)
    if "timestamp" not in train_df.columns and "job_offer_opening_date" in train_df.columns:
        train_df = train_df.rename(columns={"job_offer_opening_date": "timestamp"})
    if "timestamp" not in test_df.columns and "job_offer_opening_date" in test_df.columns:
        test_df = test_df.rename(columns={"job_offer_opening_date": "timestamp"})

    # Convert timestamps
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

    # Prepare interaction tuples
    train_inter = list(zip(train_df["candidate_id"].astype(str), train_df["job_id"].astype(str)))

    # --- Model ---
    mf = MFImplicit(args.dim, args.init_lr, args.reg, rng)
    mf.fit(train_inter, args.epochs)

    # Universe of ids (to assure ranking over all items/users)
    all_cands = set(train_df["candidate_id"].astype(str))
    all_jobs = set(train_df["job_id"].astype(str))

    # Online evaluation
    output: Dict[str, List[str]] = {}

    for ts, group in test_df.sort_values("timestamp").groupby("timestamp"):
        cand_list = group["candidate_id"].astype(str).tolist()
        job_list = group["job_id"].astype(str).tolist()

        # Ensure new IDs get vectors now to be included in ranking sets
        all_cands.update(cand_list)
        all_jobs.update(job_list)

        if args.mode == "c2j":
            all_items_sorted = sorted(all_jobs)  # deterministic tie‑break
            for c_id in cand_list:
                scores = mf.score_items_for_user(c_id, all_items_sorted)
                ranking = [iid for _, iid in sorted(zip(scores, all_items_sorted), key=lambda t: -t[0])]
                output[json.dumps([c_id, ts.strftime("%Y-%m-%d")])] = ranking
        else:
            all_users_sorted = sorted(all_cands)
            for j_id in job_list:
                scores = mf.score_users_for_item(j_id, all_users_sorted)
                ranking = [uid for _, uid in sorted(zip(scores, all_users_sorted), key=lambda t: -t[0])]
                output[json.dumps([j_id, ts.strftime("%Y-%m-%d")])] = ranking

        # --- Incremental update with this batch ---
        batch_inter = list(zip(cand_list, job_list))
        mf.fit(batch_inter, args.epochs)  # a few epochs suffices

    # Save predictions
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Temporal MF baseline written: {args.out}  (queries: {len(output)})")


if __name__ == "__main__":
    main()
