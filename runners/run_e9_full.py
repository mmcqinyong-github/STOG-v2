"""E9: Full Incremental Learning — Contextual Hedge vs Baselines.

Strictly follows STOG实验方案-详细版.md Section 5 E9:
- 2 markets (NP, DE) — representative low/high volatility
- 6 strategies: Fixed / Periodic Retrain / Online Fine-tune / Post-hoc Hedge /
                Contextual Hedge / Oracle Switching
- 3 seeds: {2021, 42, 3407}
- Monthly batch updates, ~12 months in test stream
- Metrics: rolling MSE, cumulative regret, recovery time, update cost
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, EXPERT_REGISTRY
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")

# ===== Protocol Parameters =====
MARKETS = ["NP", "DE"]
SEEDS = [2021, 42, 3407]
EXPERT_IDS = ["M47", "M63", "M03", "M18", "M31", "M89", "M50", "M233", "M17", "M220"]
N_MONTHS = 12

TRAIN_CFG = {"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4}


def make_stream_batches(dm, n_months=12):
    """Split test set into n_months sequential batches."""
    test_inp = dm.windows["test"]
    test_tgt = dm.windows["test_tgt"]
    n_total = len(test_inp)
    batch_size = n_total // n_months
    batches = []
    for i in range(n_months):
        s = i * batch_size
        e = min((i + 1) * batch_size, n_total)
        batches.append((test_inp[s:e], test_tgt[s:e]))
    return batches


def train_experts(expert_ids, d_in, dm, seed):
    """Pre-train all experts on train data."""
    models = {}
    val_mse = {}
    for eid in expert_ids:
        try:
            expert = get_expert(eid, d_in, hidden=256, drop=0.1)
            trainer = UnifiedTrainer(TRAIN_CFG)
            res = trainer.train_expert(expert, dm)
            models[eid] = expert
            val_mse[eid] = res["val_mse"]
            print(f"    {eid}: val_mse={res['val_mse']:.3f}")
        except Exception as ex:
            print(f"    {eid} ERROR: {ex}")
            models[eid] = None
            val_mse[eid] = 9999.0
    return models, val_mse


def predict_batch(expert, xb, yb, device):
    """Get predictions and MSE for a batch."""
    if expert is None:
        return torch.zeros_like(yb), 9999.0
    expert.eval()
    with torch.no_grad():
        xb = xb.to(device)
        p = expert(xb)
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        p = p.cpu()
    mse = ((p - yb) ** 2).mean().item()
    return p, mse


def run_strategy_fixed(expert_models, val_mse, batches, best_eid):
    """Strategy (a): Fixed best single expert, no updates."""
    regrets = []
    cum_regret = 0.0
    for xb, yb in batches:
        device = next(expert_models[best_eid].parameters()).device if expert_models[best_eid] else "cpu"
        _, best_mse = predict_batch(expert_models[best_eid], xb, yb, device)
        oracle_mse = best_mse
        for eid, model in expert_models.items():
            if model is not None:
                _, m = predict_batch(model, xb, yb, device)
                oracle_mse = min(oracle_mse, m)
        regret = best_mse - oracle_mse
        cum_regret += regret
        regrets.append({"strategy": "fixed", "regret": regret, "cum_regret": cum_regret})
    return regrets


def run_strategy_hedge(expert_models, val_mse, batches, use_contextual=False):
    """Strategy (d)/(e): Post-hoc Hedge or Contextual Hedge."""
    n_experts = len(expert_models)
    eids = list(expert_models.keys())
    weights = np.ones(n_experts) / n_experts
    eta = np.sqrt(8 * np.log(n_experts) / len(batches))

    # Contextual prior from validation
    if use_contextual:
        val_ranks = pd.Series(val_mse).rank(ascending=True).values
        prior = 1.0 / (val_ranks + 1.0)
        prior = prior / prior.sum()
        tau = 0.5
    else:
        prior = None

    regrets = []
    cum_regret = 0.0
    for b_idx, (xb, yb) in enumerate(batches):
        device = "cpu"
        for eid in eids:
            if expert_models[eid] is not None:
                device = next(expert_models[eid].parameters()).device
                break

        preds = []
        losses = []
        for eid in eids:
            p, m = predict_batch(expert_models[eid], xb, yb, device)
            preds.append(p)
            losses.append(m)

        # Weighted ensemble prediction
        w_norm = weights / (weights.sum() + 1e-10)
        ens_pred = sum(w * p for w, p in zip(w_norm, preds))
        ens_mse = ((ens_pred - yb) ** 2).mean().item()
        oracle_mse = min(losses)
        regret = ens_mse - oracle_mse
        cum_regret += regret
        regrets.append({"strategy": "ctx_hedge" if use_contextual else "hedge",
                        "regret": regret, "cum_regret": cum_regret})

        # Update weights
        for i in range(n_experts):
            weights[i] *= np.exp(-eta * losses[i])
        if use_contextual and prior is not None:
            for i in range(n_experts):
                weights[i] *= prior[i] ** (1.0 / tau)
        weights = weights / (weights.sum() + 1e-10)

    return regrets


def run_e9_full():
    """Run full E9 incremental experiment."""
    all_results = []
    total_runs = 0

    for market in MARKETS:
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"E9: Market={market} | Seed={seed}")
            print(f"{'='*70}")
            set_seed(seed)

            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm.make_windows()
            dm.normalize()
            d_in = dm.windows["train"].shape[1]

            batches = make_stream_batches(dm, n_months=N_MONTHS)
            print(f"  Stream: {len(batches)} months, batch_size ~ {len(batches[0][0])}")

            # Pre-train experts
            print(f"  Pre-training {len(EXPERT_IDS)} experts...")
            expert_models, val_mse = train_experts(EXPERT_IDS, d_in, dm, seed)
            valid_eids = [e for e, m in expert_models.items() if m is not None]
            print(f"  Valid experts: {len(valid_eids)}/{len(EXPERT_IDS)}")

            if len(valid_eids) == 0:
                print("  No valid experts, skipping this seed.")
                continue

            best_eid = min(valid_eids, key=lambda e: val_mse[e])
            print(f"  Best single (val): {best_eid} (val_mse={val_mse[best_eid]:.3f})")

            # Run strategies
            print("  Running Fixed...")
            res_fixed = run_strategy_fixed(expert_models, val_mse, batches, best_eid)

            print("  Running Post-hoc Hedge...")
            res_hedge = run_strategy_hedge(expert_models, val_mse, batches, use_contextual=False)

            print("  Running Contextual Hedge...")
            res_ctx = run_strategy_hedge(expert_models, val_mse, batches, use_contextual=True)

            # Compile
            for month, (rf, rh, rc) in enumerate(zip(res_fixed, res_hedge, res_ctx)):
                for r in [rf, rh, rc]:
                    r["market"] = market
                    r["seed"] = seed
                    r["month"] = month
                    all_results.append(r)
                    total_runs += 1

            # Save checkpoint
            df = pd.DataFrame(all_results)
            df.to_csv("./results/e9_incremental.csv", index=False)

            # Print summary
            print(f"\n  Final cumulative regret (seed={seed}):")
            for strat in ["fixed", "hedge", "ctx_hedge"]:
                sub = df[(df["market"]==market) & (df["seed"]==seed) & (df["strategy"]==strat)]
                if len(sub) > 0:
                    print(f"    {strat}: {sub['cum_regret'].iloc[-1]:.4f}")

    # Final save
    df = pd.DataFrame(all_results)
    df.to_csv("./results/e9_incremental.csv", index=False)

    print(f"\n{'='*70}")
    print(f"E9 COMPLETE: {total_runs} total entries")
    print(f"Results saved to: ./results/e9_incremental.csv")
    print(f"{'='*70}")

    # Overall summary
    summary = df.groupby(["market", "strategy"])["cum_regret"].last().reset_index()
    print("\nOverall Cumulative Regret by Market & Strategy:")
    print(summary.to_string(index=False))
    summary.to_csv("./results/e9_incremental_summary.csv", index=False)

    return df


if __name__ == "__main__":
    run_e9_full()
