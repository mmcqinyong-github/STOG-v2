"""E11: Cross-Domain Generalization — Leave-One-Domain-Out.

Uses 5 EPF markets + 3 long-term datasets as 8 domains.
Tests: (a) probe→rank generalization across domains, (b) LODO router performance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, EXPERT_REGISTRY
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")

# Domains
EPF_MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
LT_DATASETS = [
    ("ETTh1", "./dataset/Long-term multivariate dataset/ETT-small/ETTh1.csv", "hourly"),
    ("Weather", "./dataset/Long-term multivariate dataset/weather/weather.csv", "10min"),
    ("Exchange", "./dataset/Long-term multivariate dataset/exchange_rate/exchange_rate.csv", "daily"),
]
SEEDS = [2021, 42, 3407]
EXPERT_IDS = list(EXPERT_REGISTRY.keys())
TRAIN_CFG = {"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4}


def load_lt_dataset(name, path, freq):
    """Load long-term dataset and create EPFDataModule-compatible windows."""
    df = pd.read_csv(path)
    # Drop date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    # Use last column as target, rest as features
    data = df.values.astype(np.float32)
    # Handle NaN
    data = np.nan_to_num(data, nan=0.0)
    return data


def make_lt_windows(data, lookback=168, horizon=24, seed=2021):
    """Create train/val/test windows from long-term data."""
    n = len(data)
    # Chronological split: 70/10/20
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    def make_windows_from_split(split_data):
        X, y = [], []
        for i in range(len(split_data) - lookback - horizon + 1):
            X.append(split_data[i:i + lookback].flatten())
            y.append(split_data[i + lookback:i + lookback + horizon, -1])  # target = last column
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    train_X, train_y = make_windows_from_split(train_data)
    val_X, val_y = make_windows_from_split(val_data)
    test_X, test_y = make_windows_from_split(test_data)

    # z-score normalize using train stats
    mean = train_X.mean(dim=0)
    std = train_X.std(dim=0) + 1e-8
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    test_X = (test_X - mean) / std

    return {
        "train": train_X, "train_tgt": train_y,
        "val": val_X, "val_tgt": val_y,
        "test": test_X, "test_tgt": test_y,
    }


def train_expert_safe(eid, d_in, dm, seed):
    try:
        expert = get_expert(eid, d_in, hidden=256, drop=0.1)
        trainer = UnifiedTrainer(TRAIN_CFG)
        res = trainer.train_expert(expert, dm)
        return {
            "expert_id": eid, "seed": seed,
            "val_mse": res["val_mse"], "test_mse": res["test_mse"],
            "test_mae": res.get("test_mae", 0.0),
        }
    except Exception as ex:
        return {
            "expert_id": eid, "seed": seed,
            "val_mse": 9999.0, "test_mse": 9999.0, "test_mae": 9999.0,
            "error": str(ex),
        }


def run_e11():
    results = []

    # ===== Part 1: EPF markets (reuse E6 baseline results) =====
    print("Loading EPF baseline results from E6...")
    e6_df = pd.read_csv("./results/e6_epf_main.csv")
    for _, row in e6_df.iterrows():
        if row.get('test_mse', 9999) < 9000:
            results.append({
                "domain": row["market"],
                "domain_type": "EPF",
                "expert_id": row["expert_id"],
                "seed": int(row["seed"]),
                "test_mse": row["test_mse"],
                "test_mae": row.get("test_mae", 0.0),
            })
    print(f"  Loaded {len(results)} EPF results")

    # ===== Part 2: Long-term datasets (train fresh) =====
    for ds_name, ds_path, freq in LT_DATASETS:
        if not os.path.exists(ds_path):
            print(f"  SKIP {ds_name}: file not found at {ds_path}")
            continue
        print(f"\nDomain: {ds_name} ({freq})")
        data = load_lt_dataset(ds_name, ds_path, freq)
        if len(data) < 1000:
            print(f"  SKIP {ds_name}: too short ({len(data)} rows)")
            continue

        for seed in SEEDS:
            print(f"  Seed={seed}")
            set_seed(seed)
            windows = make_lt_windows(data, lookback=168, horizon=24, seed=seed)
            d_in = windows["train"].shape[1]

            # Create a minimal datamodule-like object
            class LTDM:
                pass
            dm = LTDM()
            dm.windows = windows

            for eid in EXPERT_IDS:
                res = train_expert_safe(eid, d_in, dm, seed)
                res["domain"] = ds_name
                res["domain_type"] = "LongTerm"
                results.append(res)
                print(f"    {eid}: test_mse={res['test_mse']:.2f}")

            # Checkpoint
            pd.DataFrame(results).to_csv("./results/e11_crossdomain.csv", index=False)

    # ===== Part 3: Cross-domain analysis =====
    df = pd.DataFrame(results)
    df.to_csv("./results/e11_crossdomain.csv", index=False)
    print(f"\nE11 total results: {len(df)} rows")

    # Rank per domain-seed
    df['rank'] = df.groupby(['domain', 'seed'])['test_mse'].rank(method='min')

    # LODO: for each domain, predict rank from other domains
    domains = df['domain'].unique()
    lodo_results = []
    for held_out in domains:
        train_domains = [d for d in domains if d != held_out]
        train_ranks = df[df['domain'].isin(train_domains)].groupby('expert_id')['rank'].mean()
        for seed in SEEDS:
            held = df[(df['domain'] == held_out) & (df['seed'] == seed)]
            if len(held) == 0:
                continue
            true_ranks = held.set_index('expert_id')['rank']
            common = train_ranks.index.intersection(true_ranks.index)
            if len(common) < 3:
                continue
            rho, pval = stats.spearmanr(train_ranks[common], true_ranks[common])
            lodo_results.append({
                "held_out": held_out,
                "seed": seed,
                "spearman_rho": rho,
                "p_value": pval,
                "n_experts": len(common),
            })

    lodo_df = pd.DataFrame(lodo_results)
    lodo_df.to_csv("./results/e11_lodo_spearman.csv", index=False)

    print("\n=== E11 LODO Spearman Summary ===")
    print(lodo_df.groupby('held_out')[['spearman_rho', 'p_value']].mean())
    print(f"\nOverall mean rho: {lodo_df['spearman_rho'].mean():.3f}")
    print(f"Overall mean p: {lodo_df['p_value'].mean():.3f}")

    # Cross-domain rank correlation matrix
    print("\n=== Cross-Domain Rank Correlation Matrix ===")
    domain_means = df.groupby(['domain', 'expert_id'])['test_mse'].mean().unstack()
    corr_mat = domain_means.T.corr(method='spearman')
    print(corr_mat.round(3))
    corr_mat.to_csv("./results/e11_domain_rank_correlation.csv")

    return df, lodo_df


if __name__ == "__main__":
    run_e11()
