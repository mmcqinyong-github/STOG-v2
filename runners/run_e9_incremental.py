"""E9: Incremental learning with contextual Hedge."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")


def run_e9_incremental(seed=2021, n_batches=20, experts=["M47", "M63", "M03", "M18", "M31"]):
    """E9: Incremental regret experiment."""
    print("=" * 60)
    print("Running E9: Incremental Learning (Contextual Hedge)")
    print("=" * 60)

    set_seed(seed)
    # Generate a longer field and split into sequential batches
    cfg = SynthConfig(T=20000, V=8, H=24, alpha=1.0, delta=0.3, seed=seed)
    gen = SpatioTemporalFieldGenerator(cfg)
    data = gen.generate()

    # Split test data into sequential windows (simulating stream)
    test_inp = data["test_inp"]
    test_tgt = data["test_tgt"]
    n_total = len(test_inp)
    batch_size = n_total // n_batches

    # Pre-train all experts on train data
    class MockDM:
        def __init__(self, d):
            self.windows = d
    dm_train = MockDM({
        "train": data["train_inp"], "train_tgt": data["train_tgt"],
        "val": data["val_inp"], "val_tgt": data["val_tgt"],
        "test": data["test_inp"], "test_tgt": data["test_tgt"],
    })
    d_in = data["train_inp"].shape[1]

    print("Pre-training experts...")
    expert_models = {}
    expert_val_mse = {}
    for eid in experts:
        try:
            expert = get_expert(eid, d_in, hidden=128, drop=0.1)
            trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2, "batch_size": 256, "lr": 1e-4})
            res = trainer.train_expert(expert, dm_train)
            expert_models[eid] = expert
            expert_val_mse[eid] = res["val_mse"]
            print(f"  {eid}: val_mse={res['val_mse']:.3f}")
        except Exception as ex:
            print(f"  {eid} ERROR: {ex}")
            expert_models[eid] = None
            expert_val_mse[eid] = 999.0

    # --- Strategy 1: Post-hoc Hedge (no prior) ---
    print("\nStrategy 1: Post-hoc Hedge...")
    n_experts = len(experts)
    weights_hedge = np.ones(n_experts) / n_experts
    eta = np.sqrt(8 * np.log(n_experts) / n_batches)
    cum_regret_hedge = []
    oracle_loss_hedge = []

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_total)
        xb = test_inp[start:end]
        yb = test_tgt[start:end]
        if yb.dim() == 1:
            yb = yb.unsqueeze(-1)

        # Predict with current weights
        preds = []
        losses = []
        for idx, eid in enumerate(experts):
            if expert_models[eid] is None:
                preds.append(torch.zeros_like(yb))
                losses.append(999.0)
                continue
            expert_models[eid].eval()
            with torch.no_grad():
                p = expert_models[eid](xb.to(next(expert_models[eid].parameters()).device))
                if p.dim() == 1:
                    p = p.unsqueeze(-1)
                p = p.cpu()
            preds.append(p)
            mse = ((p - yb) ** 2).mean().item()
            losses.append(mse)

        # Weighted prediction
        ensemble_pred = sum(w * p for w, p in zip(weights_hedge, preds))
        ens_mse = ((ensemble_pred - yb) ** 2).mean().item()
        oracle_mse = min(losses)
        cum_regret_hedge.append(ens_mse - oracle_mse)
        oracle_loss_hedge.append(oracle_mse)

        # Hedge update
        for idx in range(n_experts):
            weights_hedge[idx] *= np.exp(-eta * losses[idx])
        weights_hedge = weights_hedge / weights_hedge.sum()

    # --- Strategy 2: Contextual Hedge (probe prior + regret) ---
    print("Strategy 2: Contextual Hedge...")
    weights_ctx = np.ones(n_experts) / n_experts
    cum_regret_ctx = []
    # Prior from validation performance (inverse ranking)
    val_ranks = pd.Series(expert_val_mse).rank(ascending=True).values
    prior = 1.0 / (val_ranks + 1.0)
    prior = prior / prior.sum()
    tau = 0.5  # temperature for prior

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_total)
        xb = test_inp[start:end]
        yb = test_tgt[start:end]
        if yb.dim() == 1:
            yb = yb.unsqueeze(-1)

        preds = []
        losses = []
        for idx, eid in enumerate(experts):
            if expert_models[eid] is None:
                preds.append(torch.zeros_like(yb))
                losses.append(999.0)
                continue
            expert_models[eid].eval()
            with torch.no_grad():
                p = expert_models[eid](xb.to(next(expert_models[eid].parameters()).device))
                if p.dim() == 1:
                    p = p.unsqueeze(-1)
                p = p.cpu()
            preds.append(p)
            mse = ((p - yb) ** 2).mean().item()
            losses.append(mse)

        # Contextual weight: prior / tau - eta * cumulative_loss
        ensemble_pred = sum(w * p for w, p in zip(weights_ctx, preds))
        ens_mse = ((ensemble_pred - yb) ** 2).mean().item()
        oracle_mse = min(losses)
        cum_regret_ctx.append(ens_mse - oracle_mse)

        # Update with prior-informed Hedge
        for idx in range(n_experts):
            weights_ctx[idx] = prior[idx] ** (1.0/tau) * np.exp(-eta * losses[idx] * (b+1))
        weights_ctx = weights_ctx / weights_ctx.sum()

    # --- Strategy 3: Static Best-Single (baseline) ---
    print("Strategy 3: Static Best-Single...")
    best_eid = min(expert_val_mse, key=expert_val_mse.get)
    cum_regret_static = []
    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_total)
        xb = test_inp[start:end]
        yb = test_tgt[start:end]
        if yb.dim() == 1:
            yb = yb.unsqueeze(-1)

        losses = []
        for eid in experts:
            if expert_models[eid] is None:
                losses.append(999.0)
                continue
            expert_models[eid].eval()
            with torch.no_grad():
                p = expert_models[eid](xb.to(next(expert_models[eid].parameters()).device))
                if p.dim() == 1:
                    p = p.unsqueeze(-1)
                p = p.cpu()
            mse = ((p - yb) ** 2).mean().item()
            losses.append(mse)

        oracle_mse = min(losses)
        best_mse = losses[experts.index(best_eid)] if best_eid in experts else 999.0
        cum_regret_static.append(best_mse - oracle_mse)

    # --- Compile results ---
    results = []
    for b in range(n_batches):
        results.append({
            "batch": b, "seed": seed,
            "hedge_regret": cum_regret_hedge[b],
            "ctx_regret": cum_regret_ctx[b],
            "static_regret": cum_regret_static[b],
            "hedge_cum": sum(cum_regret_hedge[:b+1]),
            "ctx_cum": sum(cum_regret_ctx[:b+1]),
            "static_cum": sum(cum_regret_static[:b+1]),
        })

    df = pd.DataFrame(results)
    df.to_csv("./results/e9_incremental.csv", index=False)

    print("\nE9 Complete. Final cumulative regret:")
    print(f"  Contextual Hedge: {df['ctx_cum'].iloc[-1]:.4f}")
    print(f"  Post-hoc Hedge:   {df['hedge_cum'].iloc[-1]:.4f}")
    print(f"  Static Best:      {df['static_cum'].iloc[-1]:.4f}")
    print(f"  Warmup advantage (Ctx vs Hedge first 5 batches): {df['ctx_cum'].iloc[4] - df['hedge_cum'].iloc[4]:.4f}")
    return df


if __name__ == "__main__":
    run_e9_incremental()
