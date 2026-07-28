"""V3 statistical baselines under chronological split: B15 (MSTL-lite), B16 (LEAR),
plus naive-24h seasonal-naive (MASE denominator).

Split is seed-independent: one npz per market in results/preds_v3/:
  {market}_MSTL.npz    val_pred, test_pred [n,24], train_err [n_train]
  {market}_LEAR.npz    val_pred, test_pred, train_err
  {market}_naive24.npz val_pred, test_pred   (forecast = same hour previous day)

Reuses the implementations from run_baselines_stats.py (v2), only the split
changes to chronological (first 70% train / next 10% val / last 20% test).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np

from run_baselines_stats import load_price_series, b15_forecast_fast, b16_lear, build_features

EPF_DIR = "./dataset/epf"
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
L, H = 168, 24
OUT_DIR = "./results/preds_v3"
os.makedirs(OUT_DIR, exist_ok=True)


def chronological_indices(n):
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    args = ap.parse_args()
    t_start = time.time()
    for market in args.markets.split(","):
        # resume: skip markets whose npz all exist
        if all(os.path.exists(f"{OUT_DIR}/{market}_{k}.npz")
               for k in ["MSTL", "LEAR", "naive24"]):
            print(f"[{market}] already done, skip", flush=True)
            continue
        t0 = time.time()
        x = load_price_series(market)
        n = len(x) - L - H + 1
        train_idx, val_idx, test_idx = chronological_indices(n)

        # ---- B15 MSTL-lite: train/val/test window forecasts (rolling origin) ----
        p15_tr = b15_forecast_fast(x, train_idx + L)
        p15_va = b15_forecast_fast(x, val_idx + L)
        p15_te = b15_forecast_fast(x, test_idx + L)
        tgt_tr = np.stack([x[i + L:i + L + H] for i in train_idx])
        train_err15 = ((p15_tr - tgt_tr) ** 2).mean(axis=1).astype(np.float32)
        np.savez(f"{OUT_DIR}/{market}_MSTL.npz",
                 val_pred=p15_va.astype(np.float32),
                 test_pred=p15_te.astype(np.float32),
                 train_err=train_err15)

        # ---- B16 LEAR-lite (ridge; lambda on val) ----
        p16_te, lam = b16_lear(x, train_idx, val_idx, test_idx)
        # rebuild fitted model to also emit val/train predictions
        Xtr = build_features(x, train_idx)
        Ytr = np.stack([x[i + L:i + L + H] for i in train_idx])
        Xva = build_features(x, val_idx)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        ymu = Ytr.mean(axis=0)
        A = ((Xtr - mu) / sd).T @ ((Xtr - mu) / sd) + lam * np.eye(Xtr.shape[1])
        B = ((Xtr - mu) / sd).T @ (Ytr - ymu)
        W = np.linalg.solve(A, B)
        p16_tr = ((Xtr - mu) / sd) @ W + ymu
        p16_va = ((Xva - mu) / sd) @ W + ymu
        train_err16 = ((p16_tr - Ytr) ** 2).mean(axis=1).astype(np.float32)
        np.savez(f"{OUT_DIR}/{market}_LEAR.npz",
                 val_pred=p16_va.astype(np.float32),
                 test_pred=p16_te.astype(np.float32),
                 train_err=train_err16)

        # ---- naive-24h seasonal naive (MASE denominator) ----
        pnv = np.stack([x[i + L - 24:i + L] for i in val_idx])
        pnt = np.stack([x[i + L - 24:i + L] for i in test_idx])
        np.savez(f"{OUT_DIR}/{market}_naive24.npz",
                 val_pred=pnv.astype(np.float32),
                 test_pred=pnt.astype(np.float32))

        tgt_va = np.stack([x[i + L:i + L + H] for i in val_idx])
        tgt_te = np.stack([x[i + L:i + L + H] for i in test_idx])
        print(f"[{market}] MSTL test MSE={((p15_te-tgt_te)**2).mean():.3f} | "
              f"LEAR test MSE={((p16_te-tgt_te)**2).mean():.3f} (lam={lam}) | "
              f"naive24 test MAE={np.abs(pnt-tgt_te).mean():.3f} | "
              f"{time.time()-t0:.1f}s", flush=True)

    print(f"total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
