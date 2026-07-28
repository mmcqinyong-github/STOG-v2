"""Baselines for routing comparison, computed from saved per-window predictions.

B1  best-single (val MSE argmin)
B3  equal-weight average ensemble (all experts)
B4  validation-weighted (inverse val MSE)
B5  static TopK (K=3 by val, equal weight)
B7  random TopK (K=3 random, 20 reps, negative control)
B20 Oracle (per-window actual best expert, upper bound)
B8  FFORMA-lite (features -> per-expert error, HistGradientBoosting, CPU)

Outputs: results/e6_routing/baselines.csv
         results/e6_routing/fforma_errhat_{market}_{seed}.npz (for Spearman vs router)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from src.experts.zoo import EXPERT_REGISTRY

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
# E6 canonical 19 experts (same as run_e4_e6.py; M36/M51 in registry were never
# part of E6 and are excluded here for consistency)
EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
              "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
E = len(EXPERT_IDS)
PRED_DIR = "./results/preds"
OUT_DIR = "./results/e6_routing"


def load_block(market, seed):
    meta = np.load(f"{PRED_DIR}/meta_{market}_{seed}.npz")
    val_true, test_true = meta["val_true"], meta["test_true"]
    nv, H = val_true.shape
    nt = test_true.shape[0]
    val_pred = np.empty((E, nv, H), np.float32)
    test_pred = np.empty((E, nt, H), np.float32)
    train_err = []
    for i, eid in enumerate(EXPERT_IDS):
        d = np.load(f"{PRED_DIR}/{market}_{eid}_{seed}.npz")
        val_pred[i] = d["val_pred"]
        test_pred[i] = d["test_pred"]
        train_err.append(d["train_err"])
    train_err = np.stack(train_err, axis=1)  # (n_train, E)
    return meta, val_pred, test_pred, val_true, test_true, train_err


def mse(pred, true):
    return float(((pred - true) ** 2).mean())


def fit_fforma(X, Y_log, rng):
    """Per-expert HistGBM on subsample. X:(n,F), Y_log:(n,E) -> list of models."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = X.shape[0]
    sub = rng.choice(n, size=min(n, 8000), replace=False)
    models = []
    for e in range(Y_log.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=80, max_depth=6,
                                          learning_rate=0.1, random_state=0)
        m.fit(X[sub], Y_log[sub, e])
        models.append(m)
    return models


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for market in MARKETS:
        for seed in SEEDS:
            t0 = time.time()
            meta, val_pred, test_pred, val_true, test_true, train_err = load_block(market, seed)
            val_err = ((val_pred - val_true[None]) ** 2).mean(axis=2)    # (E, nv)
            test_err = ((test_pred - test_true[None]) ** 2).mean(axis=2)  # (E, nt)
            val_mse = val_err.mean(axis=1)                                 # (E,)
            rec = {"market": market, "seed": seed}

            # B1 best-single
            e_star = int(np.argmin(val_mse))
            rec["B1_best_single"] = float(test_err[e_star].mean())
            rec["B1_expert"] = EXPERT_IDS[e_star]
            # B3 average ensemble
            rec["B3_avg_ensemble"] = mse(test_pred.mean(axis=0), test_true)
            # B4 validation-weighted (inverse val MSE)
            w = (1.0 / val_mse) / (1.0 / val_mse).sum()
            rec["B4_val_weighted"] = mse(np.tensordot(w, test_pred, axes=(0, 0)), test_true)
            # B5 static TopK (K=3, equal weight)
            top3 = np.argsort(val_mse)[:3]
            rec["B5_static_top3"] = mse(test_pred[top3].mean(axis=0), test_true)
            # B7 random TopK (20 reps)
            rng = np.random.RandomState(1234)
            rms = []
            for _ in range(20):
                r3 = rng.choice(E, size=3, replace=False)
                rms.append(mse(test_pred[r3].mean(axis=0), test_true))
            rec["B7_random_top3"] = float(np.mean(rms))
            rec["B7_random_top3_std"] = float(np.std(rms))
            # B20 Oracle
            rec["B20_oracle"] = float(test_err.min(axis=0).mean())
            # B8 FFORMA-lite
            X_tr = np.concatenate([meta["feat_train"], meta["feat_val"]], axis=0)
            Y_tr = np.log(np.concatenate([train_err, val_err.T], axis=0) + 1e-8)
            models = fit_fforma(X_tr, Y_tr, np.random.RandomState(0))
            err_hat = np.stack([m.predict(meta["feat_test"]) for m in models], axis=1)  # (nt, E)
            w8 = np.exp(-err_hat - (-err_hat).max(axis=1, keepdims=True))  # softmax(-err_hat)
            w8 /= w8.sum(axis=1, keepdims=True)
            pred8 = np.einsum("ne,enh->nh", w8, test_pred)
            rec["B8_fforma_lite"] = mse(pred8, test_true)
            np.savez(f"{OUT_DIR}/fforma_errhat_{market}_{seed}.npz", err_hat=err_hat.astype(np.float32))

            for k, v in rec.items():
                if k.startswith("B") and not k.endswith("_std") and k != "B1_expert":
                    rows.append({"method": k, "market": market, "seed": seed, "test_mse": v})
            print(f"[{market}/{seed}] B1={rec['B1_best_single']:.3f}({rec['B1_expert']}) "
                  f"B3={rec['B3_avg_ensemble']:.3f} B4={rec['B4_val_weighted']:.3f} "
                  f"B5={rec['B5_static_top3']:.3f} B7={rec['B7_random_top3']:.3f} "
                  f"B8={rec['B8_fforma_lite']:.3f} B20={rec['B20_oracle']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/baselines.csv", index=False)
    summ = df.groupby("method")["test_mse"].mean().sort_values()
    print("\nBaselines 5-market mean test MSE:")
    print(summ)


if __name__ == "__main__":
    main()
