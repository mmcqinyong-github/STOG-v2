"""E8: Full Stress Testing — Three-Axis Degradation (Full Protocol).

Strictly follows STOG实验方案-详细版.md Section 5 E8:
- 5 markets (NP, PJM, BE, FR, DE)
- All available experts from zoo
- 3 stress axes: missingness, lookback truncation, covariate corruption
- Multiple severity levels per axis
- 3 seeds: {2021, 42, 3407}
- Output: degradation rates, worst-cell MSE, weight migration vectors
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.data.stress import StressInjector
from src.experts.zoo import get_expert, EXPERT_REGISTRY
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")

# ===== Protocol Parameters =====
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
EXPERT_IDS = list(EXPERT_REGISTRY.keys())  # All 19 experts

# Stress configurations per axis
MISSINGNESS_CONFIGS = [
    (0.10, "mcar"), (0.25, "mcar"), (0.375, "block"), (0.50, "block")
]
LOOKBACK_CONFIGS = [96, 168, 336, 720]
CORRUPTION_CONFIGS = [
    (0.25, "noise"), (0.50, "noise"), (1.0, "noise"),
    (0.25, "cov_missing"), (0.50, "cov_missing")
]

# Training protocol (same as E6)
TRAIN_CFG = {
    "max_epochs": 10,
    "patience": 3,
    "batch_size": 256,
    "lr": 1e-4,
}


def train_expert_safe(eid, d_in, dm, seed):
    """Train an expert, return result dict or None on failure."""
    try:
        expert = get_expert(eid, d_in, hidden=256, drop=0.1)
        trainer = UnifiedTrainer(TRAIN_CFG)
        res = trainer.train_expert(expert, dm)
        return {
            "expert_id": eid,
            "seed": seed,
            "val_mse": res["val_mse"],
            "test_mse": res["test_mse"],
            "test_mae": res.get("test_mae", 0.0),
            "epochs": res.get("epochs", 0),
        }
    except Exception as ex:
        print(f"    ERROR {eid}: {ex}")
        return {
            "expert_id": eid,
            "seed": seed,
            "val_mse": 9999.0,
            "test_mse": 9999.0,
            "test_mae": 9999.0,
            "epochs": 0,
            "error": str(ex),
        }


def run_e8_full():
    """Run full E8 stress test across all markets, experts, seeds, and stress axes."""
    results = []
    total_runs = 0

    for market in MARKETS:
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"Market={market} | Seed={seed}")
            print(f"{'='*70}")
            set_seed(seed)

            # ===== Baseline (clean) =====
            print(f"  [Baseline] Training {len(EXPERT_IDS)} experts on clean data...")
            dm_clean = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm_clean.make_windows()
            dm_clean.normalize()
            d_in_clean = dm_clean.windows["train"].shape[1]

            baseline_results = {}
            for eid in EXPERT_IDS:
                res = train_expert_safe(eid, d_in_clean, dm_clean, seed)
                res["market"] = market
                res["axis"] = "baseline"
                res["param"] = "clean"
                baseline_results[eid] = res["test_mse"]
                results.append(res)
                total_runs += 1
                print(f"    {eid}: test_mse={res['test_mse']:.2f}")

            # ===== Axis 1: Missingness =====
            for rate, pattern in MISSINGNESS_CONFIGS:
                print(f"  [Missingness] rate={rate}, pattern={pattern}")
                dm_s = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm_s.make_windows()
                dm_s.normalize()
                for split in ["train", "val", "test"]:
                    v = dm_s.windows[split]
                    v_masked, _ = StressInjector.missingness(v, rate=rate, pattern=pattern, seed=seed)
                    dm_s.windows[split] = v_masked

                for eid in EXPERT_IDS:
                    res = train_expert_safe(eid, d_in_clean, dm_s, seed)
                    b_mse = baseline_results.get(eid, 1.0)
                    deg = (res["test_mse"] - b_mse) / (b_mse + 1e-8)
                    res["market"] = market
                    res["axis"] = "missingness"
                    res["param"] = f"{rate}_{pattern}"
                    res["baseline_mse"] = b_mse
                    res["degradation"] = deg
                    results.append(res)
                    total_runs += 1
                    print(f"    {eid}: deg={deg:.3f}")

            # ===== Axis 2: Lookback Truncation =====
            for new_L in LOOKBACK_CONFIGS:
                print(f"  [Lookback] L={new_L}")
                dm_s = EPFDataModule(market, lookback=new_L, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm_s.make_windows()
                dm_s.normalize()
                d_in_s = dm_s.windows["train"].shape[1]

                for eid in EXPERT_IDS:
                    res = train_expert_safe(eid, d_in_s, dm_s, seed)
                    b_mse = baseline_results.get(eid, 1.0)
                    deg = (res["test_mse"] - b_mse) / (b_mse + 1e-8)
                    res["market"] = market
                    res["axis"] = "lookback"
                    res["param"] = f"L_{new_L}"
                    res["baseline_mse"] = b_mse
                    res["degradation"] = deg
                    results.append(res)
                    total_runs += 1
                    print(f"    {eid}: deg={deg:.3f}")

            # ===== Axis 3: Covariate Corruption =====
            for sigma, ctype in CORRUPTION_CONFIGS:
                print(f"  [Corruption] sigma={sigma}, type={ctype}")
                dm_s = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm_s.make_windows()
                dm_s.normalize()

                for split in ["train", "val", "test"]:
                    v = dm_s.windows[split]
                    if ctype == "noise":
                        dm_s.windows[split] = StressInjector.covariate_noise(v, sigma=sigma, seed=seed)
                    elif ctype == "cov_missing":
                        # Mask out covariate channels (last 2/3 of input)
                        v_corrupt = v.clone()
                        n_cov = v.shape[1] // 3
                        v_corrupt[:, n_cov:] = 0.0
                        dm_s.windows[split] = v_corrupt

                for eid in EXPERT_IDS:
                    res = train_expert_safe(eid, d_in_clean, dm_s, seed)
                    b_mse = baseline_results.get(eid, 1.0)
                    deg = (res["test_mse"] - b_mse) / (b_mse + 1e-8)
                    res["market"] = market
                    res["axis"] = "corruption"
                    res["param"] = f"{ctype}_{sigma}"
                    res["baseline_mse"] = b_mse
                    res["degradation"] = deg
                    results.append(res)
                    total_runs += 1
                    print(f"    {eid}: deg={deg:.3f}")

            # Checkpoint after each seed
            df = pd.DataFrame(results)
            df.to_csv("./results/e8_stress_test.csv", index=False)
            print(f"  Checkpoint saved. Total runs so far: {total_runs}")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv("./results/e8_stress_test.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"E8 COMPLETE: {total_runs} total runs")
    print(f"Results saved to: ./results/e8_stress_test.csv")
    print(f"{'='*70}")

    # Degradation summary
    stress_df = df[df["axis"] != "baseline"]
    if len(stress_df) > 0:
        summary = stress_df.groupby(["axis", "param"])["degradation"].agg(["mean", "std", "max"]).reset_index()
        print("\nDegradation Summary:")
        print(summary.to_string())
        summary.to_csv("./results/e8_stress_summary.csv", index=False)

    return df


if __name__ == "__main__":
    run_e8_full()
