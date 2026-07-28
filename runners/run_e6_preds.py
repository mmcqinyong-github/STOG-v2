"""E6 retrain with per-window prediction saving (for routing experiments).

Resume-capable: skips (market, expert, seed) whose npz already exists.
Saves per (market, expert, seed): results/preds/{market}_{eid}_{seed}.npz
    - val_pred  [n_val, 24] float32
    - test_pred [n_test, 24] float32
    - train_err [n_train] float32  (per-window MSE, mean over horizon)
Saves per (market, seed): results/preds/meta_{market}_{seed}.npz
    - val_true, test_true [n, 24]
    - feat_train, feat_val, feat_test [n, 12]  (probe + FFORMA features)
Also appends run metrics to results/e6_routing/e6_preds_runs.csv (new file;
does NOT touch historical results/e6_epf_main.csv).

Usage: python run_e6_preds.py --markets NP --seeds 2021,42,3407
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew as _skew, kurtosis as _kurt

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, EXPERT_REGISTRY
from src.training.trainer import UnifiedTrainer

ensure_dir("./results/preds")
ensure_dir("./results/e6_routing")

EXPERT_IDS = list(EXPERT_REGISTRY.keys())  # 19 experts (N10 fixed)
FEAT_NAMES = ["mean", "std", "skew", "kurt", "acf1", "acf24", "spec_centroid",
              "spec_decay", "season_strength", "trend_slope", "cond_number",
              "regime_overlap"]


def price_channel(inp: torch.Tensor) -> np.ndarray:
    """inp: (n, 3*L) flattened interleaved [x,z1,z2] -> (n, L) price series."""
    n, D = inp.shape
    L = D // 3
    return inp.detach().cpu().numpy().reshape(n, L, 3)[:, :, 0].astype(np.float64)


def compute_features(X: np.ndarray) -> np.ndarray:
    """Vectorized per-window features. X: (n, L). Returns (n, 12)."""
    n, L = X.shape
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    Xc = X - mean[:, None]
    sk = _skew(X, axis=1)
    ku = _kurt(X, axis=1)  # excess kurtosis
    # ACF (biased, vectorized)
    var = (Xc ** 2).mean(axis=1) + 1e-12
    def acf(lag):
        if lag == 0:
            return np.ones(n)
        return (Xc[:, :-lag] * Xc[:, lag:]).mean(axis=1) / var
    acf1 = acf(1)
    acf24 = acf(24) if L > 24 else np.zeros(n)
    # Spectrum
    P = np.abs(np.fft.rfft(Xc, axis=1)) ** 2  # (n, L//2+1)
    freqs = np.fft.rfftfreq(L)
    Psum = P[:, 1:].sum(axis=1) + 1e-12  # exclude DC
    spec_centroid = (P[:, 1:] * freqs[1:]).sum(axis=1) / Psum
    # spectral decay: slope of log-periodogram vs log-freq regression
    lf = np.log(freqs[1:] + 1e-12)
    lP = np.log(P[:, 1:] + 1e-12)
    lf_c = lf - lf.mean()
    spec_decay = (lP * lf_c).sum(axis=1) / ((lf_c ** 2).sum() + 1e-12)
    # seasonal strength: energy at 24h period bin (k = L/24) +-1 and 1st harmonic
    k24 = int(round(L / 24.0))
    idx = [k for k in [k24 - 1, k24, k24 + 1, 2 * k24] if 0 < k < P.shape[1]]
    season_strength = P[:, idx].sum(axis=1) / Psum
    # trend slope (per unit time)
    t = np.arange(L, dtype=np.float64)
    t_c = t - t.mean()
    trend_slope = (Xc * t_c).sum(axis=1) / ((t_c ** 2).sum() + 1e-12)
    # condition number of lagged (Toeplitz ACF, p=12) covariance
    p = 12
    r = np.stack([acf(l) for l in range(p)], axis=1)  # (n, p)
    r[:, 0] = 1.0
    i, j = np.indices((p, p))
    Tm = r[:, np.abs(i - j)]  # (n, p, p)
    ev = np.linalg.eigvalsh(Tm)
    cond_number = ev[:, -1] / np.clip(ev[:, 0], 1e-8, None)
    # regime overlap proxy: 1 - clipped bimodality coefficient (BC > 5/9 ~ bimodal)
    bc = (sk ** 2 + 1.0) / np.clip(ku + 3.0, 1e-3, None)
    regime_overlap = 1.0 - np.clip(bc / (5.0 / 9.0), 0.0, 1.0)
    feats = np.stack([mean, std, sk, ku, acf1, acf24, spec_centroid, spec_decay,
                      season_strength, trend_slope, cond_number, regime_overlap], axis=1)
    return feats.astype(np.float32)


@torch.no_grad()
def predict_batched(expert, inp: torch.Tensor, device, bs=8192) -> np.ndarray:
    expert.eval()
    out = []
    for i in range(0, inp.shape[0], bs):
        xb = inp[i:i + bs].to(device)
        p = expert(xb)
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NP,PJM,BE,FR,DE")
    ap.add_argument("--seeds", default="2021,42,3407")
    ap.add_argument("--experts", default=",".join(EXPERT_IDS))
    args = ap.parse_args()
    markets = args.markets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    expert_ids = args.experts.split(",")

    runs_csv = "./results/e6_routing/e6_preds_runs.csv"
    for market in markets:
        for seed in seeds:
            t0 = time.time()
            meta_path = f"./results/preds/meta_{market}_{seed}.npz"
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm.make_windows()
            dm.normalize()
            d_in = dm.windows["train"].shape[1]

            if not os.path.exists(meta_path):
                feats = {s: compute_features(price_channel(dm.windows[s]))
                         for s in ["train", "val", "test"]}
                np.savez(meta_path,
                         val_true=dm.windows["val_tgt"].cpu().numpy().astype(np.float32),
                         test_true=dm.windows["test_tgt"].cpu().numpy().astype(np.float32),
                         feat_train=feats["train"], feat_val=feats["val"],
                         feat_test=feats["test"],
                         feat_names=np.array(FEAT_NAMES))
                print(f"[{market}/{seed}] meta saved "
                      f"(n_train={feats['train'].shape[0]}, n_val={feats['val'].shape[0]}, "
                      f"n_test={feats['test'].shape[0]})", flush=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for eid in expert_ids:
                out_path = f"./results/preds/{market}_{eid}_{seed}.npz"
                if os.path.exists(out_path):
                    continue
                te = time.time()
                try:
                    set_seed(seed)
                    expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                    trainer = UnifiedTrainer({"max_epochs": 10, "patience": 3,
                                              "batch_size": 256, "lr": 1e-4})
                    res = trainer.train_expert(expert, dm)
                    val_pred = predict_batched(expert, dm.windows["val"], device)
                    test_pred = predict_batched(expert, dm.windows["test"], device)
                    train_pred = predict_batched(expert, dm.windows["train"], device)
                    train_true = dm.windows["train_tgt"].cpu().numpy().astype(np.float32)
                    train_err = ((train_pred - train_true) ** 2).mean(axis=1).astype(np.float32)
                    np.savez(out_path, val_pred=val_pred, test_pred=test_pred,
                             train_err=train_err)
                    row = {"market": market, "expert_id": eid, "seed": seed,
                           "val_mse": res["val_mse"], "test_mse": res["test_mse"],
                           "test_mae": res["test_mae"], "epochs": res["epochs"],
                           "time_sec": res["time_sec"]}
                except Exception as ex:
                    print(f"  ERROR {market}/{eid}/{seed}: {ex}", flush=True)
                    row = {"market": market, "expert_id": eid, "seed": seed,
                           "val_mse": 999.0, "test_mse": 999.0, "test_mae": 999.0,
                           "epochs": 0, "time_sec": 0.0}
                pd.DataFrame([row]).to_csv(
                    runs_csv, mode="a", index=False,
                    header=not os.path.exists(runs_csv) or os.path.getsize(runs_csv) == 0)
                print(f"  [{market}/{seed}] {eid} done in {time.time()-te:.1f}s "
                      f"val={row['val_mse']:.3f} test={row['test_mse']:.3f}", flush=True)
            print(f"[{market}/{seed}] block finished in {time.time()-t0:.1f}s", flush=True)
            del dm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
