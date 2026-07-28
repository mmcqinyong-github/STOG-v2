"""Resume-capable runner for E4 and E6."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import spearmanr, linregress
from tqdm import tqdm

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")
ensure_dir("./logs")


def run_e4_fixed(delta_values=[0.1, 0.3, 0.6, 0.9], seeds=[2021, 42, 3407]):
    """E4: Validate Theorem 4 - gate benefit vs (1-delta)."""
    print("=" * 60)
    print("Running E4: Regime Overlap (fixed gate)")
    print("=" * 60)

    results = []
    for delta in delta_values:
        for seed in seeds:
            cfg = SynthConfig(T=5000, V=8, H=24, alpha=1.0, delta=delta, seed=seed)
            gen = SpatioTemporalFieldGenerator(cfg)
            data = gen.generate()

            eids = ["M52", "M233"]
            class MockDM:
                def __init__(self, d):
                    self.windows = d
            dm = MockDM({
                "train": data["train_inp"], "train_tgt": data["train_tgt"],
                "val": data["val_inp"], "val_tgt": data["val_tgt"],
                "test": data["test_inp"], "test_tgt": data["test_tgt"],
            })

            preds_val = []
            preds_test = []
            for eid in eids:
                set_seed(seed)
                expert = get_expert(eid, data["train_inp"].shape[1], hidden=128)
                trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4})
                try:
                    res = trainer.train_expert(expert, dm)
                    expert.eval()
                    dev = next(expert.parameters()).device
                    with torch.no_grad():
                        pv = expert(data["val_inp"].to(dev))
                        pt = expert(data["test_inp"].to(dev))
                        if pv.dim() == 1:
                            pv = pv.unsqueeze(-1)
                        if pt.dim() == 1:
                            pt = pt.unsqueeze(-1)
                    preds_val.append(pv)
                    preds_test.append(pt)
                except Exception as ex:
                    print(f"  Error training {eid}: {ex}")
                    preds_val.append(torch.zeros_like(data["val_tgt"]))
                    preds_test.append(torch.zeros_like(data["test_tgt"]))
                set_seed(seed)
                expert = get_expert(eid, data["train_inp"].shape[1], hidden=128)
                trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4})
                try:
                    res = trainer.train_expert(expert, dm)
                    expert.eval()
                    with torch.no_grad():
                        pv = expert(data["val_inp"])
                        pt = expert(data["test_inp"])
                        if pv.dim() == 1:
                            pv = pv.unsqueeze(-1)
                        if pt.dim() == 1:
                            pt = pt.unsqueeze(-1)
                    preds_val.append(pv)
                    preds_test.append(pt)
                except Exception:
                    preds_val.append(torch.zeros_like(data["val_tgt"]))
                    preds_test.append(torch.zeros_like(data["test_tgt"]))

            if len(preds_test) >= 2:
                # Ensure all on CPU for ensemble math
                preds_val = [p.cpu() for p in preds_val]
                preds_test = [p.cpu() for p in preds_test]
                val_tgt_cpu = data["val_tgt"].cpu()
                test_tgt_cpu = data["test_tgt"].cpu()

                # Static equal-weight ensemble
                static_pred = (preds_test[0] + preds_test[1]) / 2
                mse_static = ((static_pred - test_tgt_cpu) ** 2).mean().item()

                # Learned convex combination gate on validation data
                best_mse = float('inf')
                best_lam = 0.5
                for lam in np.linspace(0, 1, 21):
                    gate_pred_val = lam * preds_val[0] + (1 - lam) * preds_val[1]
                    mse_val = ((gate_pred_val - val_tgt_cpu) ** 2).mean().item()
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_lam = lam

                gate_pred_test = best_lam * preds_test[0] + (1 - best_lam) * preds_test[1]
                mse_gate = ((gate_pred_test - test_tgt_cpu) ** 2).mean().item()
                # Static equal-weight ensemble
                static_pred = (preds_test[0] + preds_test[1]) / 2
                mse_static = ((static_pred - data["test_tgt"]) ** 2).mean().item()

                # Learned convex combination gate on validation data
                # Grid search for optimal lambda in [0,1]
                best_mse = float('inf')
                best_lam = 0.5
                for lam in np.linspace(0, 1, 21):
                    gate_pred_val = lam * preds_val[0] + (1 - lam) * preds_val[1]
                    mse_val = ((gate_pred_val - data["val_tgt"]) ** 2).mean().item()
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_lam = lam

                gate_pred_test = best_lam * preds_test[0] + (1 - best_lam) * preds_test[1]
                mse_gate = ((gate_pred_test - data["test_tgt"]) ** 2).mean().item()
            else:
                mse_static = 999.0
                mse_gate = 999.0

            benefit = mse_static - mse_gate
            results.append({
                "delta": delta, "seed": seed,
                "mse_static": mse_static, "mse_gate": mse_gate, "benefit": benefit,
                "one_minus_delta": 1 - delta, "best_lambda": best_lam
            })

    df = pd.DataFrame(results)
    df.to_csv("./results/e4_regime_overlap.csv", index=False)
    slope, intercept, r_value, p_value, std_err = linregress(df["one_minus_delta"], df["benefit"])
    print(f"\nE4 Complete. R^2(benefit ~ 1-delta): {r_value**2:.4f}, slope={slope:.4f}, p={p_value:.4g}")
    return df


def run_e6_chunk(markets=["NP"], expert_ids=None, seeds=[2021, 42, 3407], resume=True):
    """Run E6 for a subset of markets with resume support."""
    print("=" * 60)
    print(f"Running E6 chunk: markets={markets}")
    print("=" * 60)

    if expert_ids is None:
        expert_ids = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
                      "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]

    out_path = "./results/e6_epf_main.csv"
    done_set = set()
    if resume and os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            done_set.add((row["market"], row["expert_id"], row["seed"]))
        results = existing.to_dict('records')
        print(f"Resuming: {len(done_set)} runs already completed.")
    else:
        results = []

    for market in markets:
        for seed in seeds:
            print(f"\nMarket={market}, Seed={seed}")
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm.make_windows()
            dm.normalize()
            d_in = dm.windows["train"].shape[1]
            for eid in tqdm(expert_ids, desc=f"{market}-{seed}"):
                key = (market, eid, seed)
                if key in done_set:
                    continue
                try:
                    expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                    trainer = UnifiedTrainer({"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4})
                    res = trainer.train_expert(expert, dm)
                    results.append({
                        "market": market, "expert_id": eid, "seed": seed,
                        "val_mse": res["val_mse"], "test_mse": res["test_mse"], "test_mae": res["test_mae"],
                        "epochs": res["epochs"], "time_sec": res["time_sec"]
                    })
                except Exception as ex:
                    print(f"  Error with {eid}: {ex}")
                    results.append({
                        "market": market, "expert_id": eid, "seed": seed,
                        "val_mse": 999.0, "test_mse": 999.0, "test_mae": 999.0,
                        "epochs": 0, "time_sec": 0
                    })
                # Save after every expert to minimize data loss on timeout
                pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)
    summary = df.groupby(["market", "expert_id"])["test_mse"].mean().reset_index()
    pivot = summary.pivot(index="expert_id", columns="market", values="test_mse")
    if not pivot.empty:
        pivot["avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg")
        print("\nE6 Complete. Top experts by avg test MSE:")
        print(pivot.head(10))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["e4", "e6"], required=True)
    parser.add_argument("--markets", default="NP,PJM,BE,FR,DE", help="Comma-separated markets for E6")
    args = parser.parse_args()

    if args.exp == "e4":
        run_e4_fixed()
    elif args.exp == "e6":
        markets = args.markets.split(",")
        run_e6_chunk(markets=markets)
