"""Lightweight E7: Long-term forecasting on representative datasets."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.common import set_seed, ensure_dir
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer


def load_longterm(name, lookback=336, horizon=96):
    """Load a long-term dataset and create windows."""
    base = "./dataset/Long-term multivariate dataset"
    if name.startswith("ETT"):
        path = f"{base}/ETT-small/{name}.csv"
    elif name == "weather":
        path = f"{base}/weather/weather.csv"
    elif name == "traffic":
        path = f"{base}/traffic/traffic.csv"
    elif name == "exchange":
        path = f"{base}/exchange_rate/exchange_rate.csv"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = pd.read_csv(path)
    # Drop date column if present
    cols = [c for c in df.columns if c.lower() not in ["date", "ot"]]
    if "OT" in df.columns:
        cols = [c for c in df.columns if c != "OT"]
    # Use all numeric columns
    data = df[cols].select_dtypes(include=[np.number]).values.astype(np.float32)
    if data.shape[1] == 0:
        data = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    # Flatten multivariate: each window is (lookback * n_vars,)
    n_vars = data.shape[1]
    T = len(data)
    windows_inp, windows_tgt = [], []
    for i in range(T - lookback - horizon + 1):
        windows_inp.append(data[i:i+lookback].flatten())
        # Target: predict all variables' next horizon steps
        windows_tgt.append(data[i+lookback:i+lookback+horizon].flatten())

    inp = torch.tensor(np.stack(windows_inp), dtype=torch.float32)
    tgt = torch.tensor(np.stack(windows_tgt), dtype=torch.float32)

    n = len(windows_inp)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)

    return {
        "train": inp[:n_train], "train_tgt": tgt[:n_train],
        "val": inp[n_train:n_train+n_val], "val_tgt": tgt[n_train:n_train+n_val],
        "test": inp[n_train+n_val:], "test_tgt": tgt[n_train+n_val:],
        "n_vars": n_vars,
    }


def run_e7_light(datasets=["ETTh1", "weather", "traffic", "exchange"],
                 horizons=[96],
                 expert_ids=None,
                 seeds=[2021]):
    """Lightweight long-term experiment."""
    print("=" * 60)
    print("Running E7: Long-term Forecasting (lightweight)")
    print("=" * 60)

    if expert_ids is None:
        expert_ids = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
                      "M55", "M63", "M89", "M117", "M220", "M233"]

    ensure_dir("./results")
    results = []

    for ds_name in datasets:
        for H in horizons:
            print(f"\nDataset={ds_name}, Horizon={H}")
            try:
                windows = load_longterm(ds_name, lookback=336, horizon=H)
            except Exception as ex:
                print(f"  Failed to load {ds_name}: {ex}")
                continue

            d_in = windows["train"].shape[1]
            class MockDM:
                def __init__(self, d):
                    self.windows = d
            dm = MockDM(windows)

            for seed in seeds:
                for eid in tqdm(expert_ids, desc=f"{ds_name}-H{H}-seed{seed}"):
                    set_seed(seed)
                    try:
                        expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                        trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2, "batch_size": 256, "lr": 1e-4})
                        res = trainer.train_expert(expert, dm)
                        results.append({
                            "dataset": ds_name, "horizon": H, "expert_id": eid, "seed": seed,
                            "val_mse": res["val_mse"], "test_mse": res["test_mse"], "test_mae": res["test_mae"],
                            "n_vars": windows["n_vars"], "epochs": res["epochs"], "time_sec": res["time_sec"]
                        })
                    except Exception as ex:
                        print(f"  Error {eid}: {ex}")
                        results.append({
                            "dataset": ds_name, "horizon": H, "expert_id": eid, "seed": seed,
                            "val_mse": 99999.0, "test_mse": 99999.0, "test_mae": 99999.0,
                            "n_vars": windows["n_vars"], "epochs": 0, "time_sec": 0
                        })

                # Save incrementally
                pd.DataFrame(results).to_csv("./results/e7_longterm_light.csv", index=False)

    df = pd.DataFrame(results)
    if not df.empty:
        summary = df.groupby(["dataset", "expert_id"])["test_mse"].mean().reset_index()
        pivot = summary.pivot(index="expert_id", columns="dataset", values="test_mse")
        if not pivot.empty:
            pivot["avg"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("avg")
            print("\nE7 Complete. Top experts by avg test MSE:")
            print(pivot.head(10))
    return df


if __name__ == "__main__":
    run_e7_light()
