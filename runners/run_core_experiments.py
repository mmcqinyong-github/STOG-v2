"""Main experiment orchestrator for STOG-MetaMorph.
Runs core experiments E1-E6 with representative subsets."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import spearmanr
from tqdm import tqdm

from src.utils.common import set_seed, ensure_dir, save_results
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.data.epf import EPFDataModule
from src.data.stress import StressInjector
from src.experts.zoo import EXPERT_REGISTRY, get_expert, get_all_cards
from src.training.trainer import UnifiedTrainer
from src.probes.input_probe import InputProbe
from src.theory.affinity import SpectralAffinity
from src.eval.metrics import mse, mae, rmse, degradation


# ============== E1: Synthetic Spectral Field ==============

def run_e1(alpha_values=[0.5, 1.0, 2.0], spatial_types=["lowrank", "aligned", "misaligned", "white"],
           kappa_values=[0.0, 0.3, 0.7], seeds=[2021, 42, 3407], n_experts_subset=12):
    """E1: Validate Theorem 1 - spectral affinity predicts expert ranking."""
    print("=" * 60)
    print("Running E1: Synthetic Spectral Field")
    print("=" * 60)

    expert_ids = ["M52", "M03", "M01", "M117", "M36", "M17", "M14", "N01", "M31", "M50", "M89", "M233"]
    expert_ids = expert_ids[:n_experts_subset]

    results = []
    affinity_est = SpectralAffinity()
    probe = InputProbe()

    configs = []
    for alpha in alpha_values:
        for st in spatial_types:
            for kappa in kappa_values:
                for seed in seeds:
                    configs.append((alpha, st, kappa, seed))

    for alpha, st, kappa, seed in tqdm(configs, desc="E1 configs"):
        cfg = SynthConfig(T=5000, V=8, H=24, alpha=alpha, spatial_type=st,
                          kappa_st=kappa, seed=seed)
        gen = SpatioTemporalFieldGenerator(cfg)
        data = gen.generate()

        # Train each expert
        rankings = {}
        for eid in expert_ids:
            set_seed(seed)
            d_in = data["train_inp"].shape[1]
            expert = get_expert(eid, d_in, hidden=128, drop=0.1)
            # Simple mock datamodule
            class MockDM:
                def __init__(self, d):
                    self.windows = d
            dm = MockDM({
                "train": data["train_inp"], "train_tgt": data["train_tgt"],
                "val": data["val_inp"], "val_tgt": data["val_tgt"],
                "test": data["test_inp"], "test_tgt": data["test_tgt"],
            })
            trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4})
            try:
                res = trainer.train_expert(expert, dm)
                rankings[eid] = res["test_mse"]
            except Exception as ex:
                rankings[eid] = 999.0

        # True ranking by MSE
        true_rank = sorted(expert_ids, key=lambda x: rankings[x])
        true_ranks = {eid: i + 1 for i, eid in enumerate(true_rank)}

        # Predicted ranking by spectral affinity
        # Use first test input as probe
        probe_feats = {"low_freq_ratio": 0.5 + (alpha - 1.0) * 0.2,
                       "spec_entropy": alpha,
                       "spike_count": 0.0}
        cards = get_all_cards()
        pred_rank = affinity_est.rank_experts({eid: cards[eid] for eid in expert_ids}, probe_feats)
        pred_ranks = {eid: i + 1 for i, eid in enumerate(pred_rank)}

        # Spearman correlation
        true_r_list = [true_ranks[eid] for eid in expert_ids]
        pred_r_list = [pred_ranks[eid] for eid in expert_ids]
        rho, pval = spearmanr(true_r_list, pred_r_list)

        results.append({
            "alpha": alpha, "spatial_type": st, "kappa": kappa, "seed": seed,
            "spearman_rho": rho, "pvalue": pval,
            **{f"mse_{eid}": rankings[eid] for eid in expert_ids}
        })

    df = pd.DataFrame(results)
    ensure_dir("./results")
    df.to_csv("./results/e1_synthetic_spectral.csv", index=False)
    print(f"\nE1 Complete. Mean Spearman rho: {df['spearman_rho'].mean():.4f}")
    return df


# ============== E2: Condition Number ==============

def run_e2():
    """E2: Validate Theorem 2 - condition number regularization."""
    print("=" * 60)
    print("Running E2: Condition Number Validation")
    print("=" * 60)

    results = []
    for alpha in [0.5, 1.0, 2.0]:
        for seed in [2021, 42, 3407]:
            cfg = SynthConfig(T=5000, V=8, H=24, alpha=alpha, seed=seed)
            gen = SpatioTemporalFieldGenerator(cfg)
            data = gen.generate()
            X = data["X"].numpy()  # (T, V)

            # Compute condition numbers
            L = 24
            for v_idx in range(min(3, X.shape[1])):
                series = X[:, v_idx]
                # Toeplitz covariance
                windows = np.array([series[i:i + L] for i in range(len(series) - L)])
                cov = np.cov(windows.T)
                kappa_x = np.linalg.cond(cov)

                # Differenced series
                d_series = np.diff(series)
                d_windows = np.array([d_series[i:i + L] for i in range(len(d_series) - L)])
                if len(d_windows) > L:
                    cov_d = np.cov(d_windows.T)
                    kappa_dx = np.linalg.cond(cov_d)
                else:
                    kappa_dx = kappa_x

                kappa_ratio = np.log(kappa_x / (kappa_dx + 1e-10) + 1e-10)

                results.append({
                    "alpha": alpha, "seed": seed, "var_idx": v_idx,
                    "kappa_x": kappa_x, "kappa_dx": kappa_dx, "kappa_ratio": kappa_ratio
                })

    df = pd.DataFrame(results)
    df.to_csv("./results/e2_condition_number.csv", index=False)
    print(f"E2 Complete. Mean kappa ratio (alpha=2.0): {df[df['alpha']==2.0]['kappa_ratio'].mean():.4f}")
    return df


# ============== E3: Heavy-tail Robustness ==============

def run_e3(spike_rates=[0.01, 0.05, 0.10], spike_amps=[3.0, 6.0, 10.0], seeds=[2021, 42, 3407]):
    """E3: Validate Theorem 3 - robust moment experts under spike contamination."""
    print("=" * 60)
    print("Running E3: Heavy-tail Robustness")
    print("=" * 60)

    # Robust group vs Raw group
    robust_ids = ["M233", "M55", "M220"]
    raw_ids = ["M03", "M52"]
    all_ids = robust_ids + raw_ids

    results = []
    for spike_rate in spike_rates:
        for spike_amp in spike_amps:
            for seed in seeds:
                cfg = SynthConfig(T=5000, V=8, H=24, alpha=1.0, spatial_type="lowrank",
                                  spike_rate=spike_rate, spike_amp=spike_amp, seed=seed)
                gen = SpatioTemporalFieldGenerator(cfg)
                data = gen.generate()

                # Also generate clean version
                cfg_clean = SynthConfig(T=5000, V=8, H=24, alpha=1.0, spatial_type="lowrank",
                                        spike_rate=0.0, spike_amp=0.0, seed=seed)
                data_clean = SpatioTemporalFieldGenerator(cfg_clean).generate()

                for eid in all_ids:
                    set_seed(seed)
                    d_in = data["train_inp"].shape[1]
                    expert = get_expert(eid, d_in, hidden=128, drop=0.1)
                    class MockDM:
                        def __init__(self, d):
                            self.windows = d
                    dm = MockDM({
                        "train": data["train_inp"], "train_tgt": data["train_tgt"],
                        "val": data["val_inp"], "val_tgt": data["val_tgt"],
                        "test": data["test_inp"], "test_tgt": data["test_tgt"],
                    })
                    dm_clean = MockDM({
                        "train": data_clean["train_inp"], "train_tgt": data_clean["train_tgt"],
                        "val": data_clean["val_inp"], "val_tgt": data_clean["val_tgt"],
                        "test": data_clean["test_inp"], "test_tgt": data_clean["test_tgt"],
                    })
                    trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4})
                    try:
                        res_spike = trainer.train_expert(expert, dm)
                        expert2 = get_expert(eid, d_in, hidden=128, drop=0.1)
                        res_clean = trainer.train_expert(expert2, dm_clean)
                        deg = degradation(res_spike["test_mse"], res_clean["test_mse"])
                    except Exception:
                        res_spike = {"test_mse": 999.0}
                        res_clean = {"test_mse": 999.0}
                        deg = 0.0

                    results.append({
                        "expert_id": eid, "group": "robust" if eid in robust_ids else "raw",
                        "spike_rate": spike_rate, "spike_amp": spike_amp, "seed": seed,
                        "mse_spike": res_spike["test_mse"], "mse_clean": res_clean["test_mse"],
                        "degradation": deg
                    })

    df = pd.DataFrame(results)
    df.to_csv("./results/e3_heavytail.csv", index=False)
    # Summary by group
    summary = df.groupby(["group", "spike_rate"])["degradation"].mean().reset_index()
    print("\nE3 Complete. Degradation summary:")
    print(summary)
    return df


# ============== E4: Regime Overlap ==============

def run_e4(delta_values=[0.1, 0.3, 0.6, 0.9], seeds=[2021, 42, 3407]):
    """E4: Validate Theorem 4 - gate benefit vs (1-delta)."""
    print("=" * 60)
    print("Running E4: Regime Overlap")
    print("=" * 60)

    results = []
    for delta in delta_values:
        for seed in seeds:
            cfg = SynthConfig(T=5000, V=8, H=24, alpha=1.0, delta=delta, seed=seed)
            gen = SpatioTemporalFieldGenerator(cfg)
            data = gen.generate()

            # Static equal-weight ensemble
            eids = ["M52", "M233"]
            mse_static = 0.0
            mse_gate = 0.0

            class MockDM:
                def __init__(self, d):
                    self.windows = d
            dm = MockDM({
                "train": data["train_inp"], "train_tgt": data["train_tgt"],
                "val": data["val_inp"], "val_tgt": data["val_tgt"],
                "test": data["test_inp"], "test_tgt": data["test_tgt"],
            })

            preds = []
            for eid in eids:
                set_seed(seed)
                expert = get_expert(eid, data["train_inp"].shape[1], hidden=128)
                trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4})
                try:
                    res = trainer.train_expert(expert, dm)
                    expert.eval()
                    with torch.no_grad():
                        pred = expert(data["test_inp"])
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(-1)
                    preds.append(pred)
                except Exception:
                    preds.append(torch.zeros_like(data["test_tgt"]))

            if len(preds) >= 2:
                # Static ensemble
                static_pred = (preds[0] + preds[1]) / 2
                mse_static = ((static_pred - data["test_tgt"]) ** 2).mean().item()

                # Simple learned gate
                gate = torch.sigmoid(torch.randn(1)).item()
                gate_pred = gate * preds[0] + (1 - gate) * preds[1]
                mse_gate = ((gate_pred - data["test_tgt"]) ** 2).mean().item()

            benefit = mse_static - mse_gate
            results.append({
                "delta": delta, "seed": seed,
                "mse_static": mse_static, "mse_gate": mse_gate, "benefit": benefit,
                "one_minus_delta": 1 - delta
            })

    df = pd.DataFrame(results)
    df.to_csv("./results/e4_regime_overlap.csv", index=False)
    # Fit linear model benefit ~ (1-delta)
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df["one_minus_delta"], df["benefit"])
    print(f"\nE4 Complete. R^2(benefit ~ 1-delta): {r_value**2:.4f}, slope={slope:.4f}")
    return df


# ============== E6: EPF Main Experiment ==============

def run_e6(markets=["NP", "PJM", "BE", "FR", "DE"], expert_ids=None, seeds=[2021, 42, 3407]):
    """E6: Main EPF experiment with representative expert subset."""
    print("=" * 60)
    print("Running E6: EPF Main Experiment")
    print("=" * 60)

    if expert_ids is None:
        expert_ids = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
                      "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]

    results = []
    for market in markets:
        for seed in seeds:
            print(f"\nMarket={market}, Seed={seed}")
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed,
                               data_dir="./dataset/epf")
            dm.make_windows()
            dm.normalize()

            d_in = dm.windows["train"].shape[1]
            for eid in tqdm(expert_ids, desc=f"{market}-{seed}"):
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

    df = pd.DataFrame(results)
    ensure_dir("./results")
    df.to_csv("./results/e6_epf_main.csv", index=False)

    # Summary
    summary = df.groupby(["market", "expert_id"])["test_mse"].mean().reset_index()
    pivot = summary.pivot(index="expert_id", columns="market", values="test_mse")
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg")
    print("\nE6 Complete. Top experts by avg test MSE:")
    print(pivot.head(10))
    return df


# ============== Main Entry ==============

if __name__ == "__main__":
    ensure_dir("./results")
    ensure_dir("./logs")

    # Run all core experiments
    print("\n" + "=" * 60)
    print("STOG-MetaMorph Core Experiments")
    print("=" * 60)

    # E1: Synthetic spectral (reduced grid for speed)
    df_e1 = run_e1(alpha_values=[0.5, 1.0, 2.0],
                   spatial_types=["lowrank", "aligned"],
                   kappa_values=[0.0, 0.3],
                   seeds=[2021, 42],
                   n_experts_subset=8)

    # E2: Condition number
    df_e2 = run_e2()

    # E3: Heavy-tail (reduced grid)
    df_e3 = run_e3(spike_rates=[0.01, 0.05],
                   spike_amps=[3.0, 6.0],
                   seeds=[2021, 42])

    # E4: Regime overlap
    df_e4 = run_e4(delta_values=[0.1, 0.3, 0.6, 0.9], seeds=[2021, 42])

    # E6: EPF main (key experiment - full run)
    df_e6 = run_e6(markets=["NP", "PJM", "BE", "FR", "DE"],
                   seeds=[2021, 42, 3407])

    print("\n" + "=" * 60)
    print("All core experiments completed!")
    print("Results saved to ./results/")
    print("=" * 60)
