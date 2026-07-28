"""E8: Lightweight stress testing — three-axis degradation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.data.stress import StressInjector
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")


def run_e8_light(market="NP", experts=["M47", "M63", "M03"], seed=2021):
    """Lightweight stress test."""
    print("=" * 60)
    print(f"Running E8 (light): Stress Testing on {market}, seed={seed}")
    print("=" * 60)

    set_seed(seed)
    dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
    dm.make_windows()
    dm.normalize()
    d_in = dm.windows["train"].shape[1]

    results = []
    baseline_mse = {}

    # Baseline
    print("Running baselines...")
    for eid in experts:
        try:
            expert = get_expert(eid, d_in, hidden=256, drop=0.1)
            trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2, "batch_size": 256, "lr": 1e-4})
            res = trainer.train_expert(expert, dm)
            baseline_mse[eid] = res["test_mse"]
            print(f"  {eid} baseline MSE: {res['test_mse']:.3f}")
        except Exception as ex:
            baseline_mse[eid] = 999.0
            print(f"  {eid} baseline ERROR: {ex}")

    def run_stress(axis_name, param_name, dm_stress, d_in_stress):
        for eid in experts:
            try:
                expert = get_expert(eid, d_in_stress, hidden=256, drop=0.1)
                trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2, "batch_size": 256, "lr": 1e-4})
                res = trainer.train_expert(expert, dm_stress)
                deg = (res["test_mse"] - baseline_mse[eid]) / (baseline_mse[eid] + 1e-8)
                results.append({
                    "axis": axis_name, "param": param_name, "expert_id": eid, "seed": seed,
                    "baseline_mse": baseline_mse[eid], "stress_mse": res["test_mse"], "degradation": deg
                })
                print(f"  {eid} {axis_name}/{param_name}: MSE={res['test_mse']:.3f}, deg={deg:.3f}")
            except Exception as ex:
                results.append({
                    "axis": axis_name, "param": param_name, "expert_id": eid, "seed": seed,
                    "baseline_mse": baseline_mse[eid], "stress_mse": 999.0, "degradation": 0.0
                })
                print(f"  {eid} {axis_name}/{param_name} ERROR: {ex}")

    # Axis 1: Missingness
    print("\nAxis 1: Missingness...")
    for rate, pattern in [(0.1, "mcar"), (0.25, "block")]:
        dm_s = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
        dm_s.make_windows()
        dm_s.normalize()
        for split in ["train", "val", "test"]:
            v = dm_s.windows[split]
            v_masked, _ = StressInjector.missingness(v, rate=rate, pattern=pattern, seed=seed)
            dm_s.windows[split] = v_masked
        run_stress("missingness", f"{rate}_{pattern}", dm_s, d_in)

    # Axis 2: Noise
    print("\nAxis 2: Covariate Noise...")
    for sigma in [0.1, 0.3]:
        dm_s = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
        dm_s.make_windows()
        dm_s.normalize()
        for split in ["train", "val", "test"]:
            v = dm_s.windows[split]
            dm_s.windows[split] = StressInjector.covariate_noise(v, sigma=sigma, seed=seed)
        run_stress("noise", f"sigma_{sigma}", dm_s, d_in)

    # Axis 3: Truncation
    print("\nAxis 3: Lookback Truncation...")
    for new_L in [96, 48]:
        dm_s = EPFDataModule(market, lookback=new_L, horizon=24, seed=seed, data_dir="./dataset/epf")
        dm_s.make_windows()
        dm_s.normalize()
        d_in_t = dm_s.windows["train"].shape[1]
        run_stress("truncate", f"L_{new_L}", dm_s, d_in_t)

    df = pd.DataFrame(results)
    df.to_csv("./results/e8_stress_test.csv", index=False)
    print("\nE8 Complete. Average degradation by axis:")
    summary = df.groupby(["axis", "param"])["degradation"].mean().reset_index()
    print(summary)
    return df


if __name__ == "__main__":
    run_e8_light()
