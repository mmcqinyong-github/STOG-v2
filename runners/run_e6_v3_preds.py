"""E6 v3 (reviewer-driven rerun): chronological split, per-window prediction saving.

Resume-capable: skips (market, expert, seed) whose npz already exists.
Saves per (market, expert, seed): results/preds_v3/{market}_{eid}_{seed}.npz
    - val_pred  [n_val, 24] float32
    - test_pred [n_test, 24] float32
    - train_err [n_train] float32  (per-window MSE, mean over horizon)
Saves per market (split is seed-independent under chronological mode):
    results/preds_v3/meta_{market}.npz
    - val_true, test_true [n, 24]
    - feat_train, feat_val, feat_test [n, 12]
Appends run metrics to results/e6_v3/e6v3_runs.csv (does NOT touch v2 files).

Priority order: seeds outer (2021 -> 42 -> 3407 -> 7 -> 12345), markets inner,
so whole (market, seed) blocks complete early for downstream router runs.

Usage: python run_e6_v3_preds.py --time-budget 280
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer
from run_e6_preds import compute_features, price_channel, FEAT_NAMES

ensure_dir("./results/preds_v3")
ensure_dir("./results/e6_v3")

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407, 7, 12345]
EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
              "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]


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
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--experts", default=",".join(EXPERT_IDS))
    ap.add_argument("--time-budget", type=float, default=1e9,
                    help="seconds; stop before starting a new block when exceeded")
    args = ap.parse_args()
    markets = args.markets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    expert_ids = args.experts.split(",")

    t_start = time.time()
    runs_csv = "./results/e6_v3/e6v3_runs.csv"
    done, total = 0, len(markets) * len(seeds) * len(expert_ids)

    for seed in seeds:                      # seeds OUTER (priority order)
        for market in markets:
            if time.time() - t_start > args.time_budget:
                print(f"[budget] stopping; rerun same command to resume", flush=True)
                return
            t0 = time.time()
            meta_path = f"./results/preds_v3/meta_{market}.npz"
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed,
                               data_dir="./dataset/epf",
                               split_mode="chronological")
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
                print(f"[{market}] meta saved "
                      f"(n_train={feats['train'].shape[0]}, n_val={feats['val'].shape[0]}, "
                      f"n_test={feats['test'].shape[0]})", flush=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for eid in expert_ids:
                out_path = f"./results/preds_v3/{market}_{eid}_{seed}.npz"
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
                done += 1
                print(f"  [{seed}/{market}] {eid} done in {time.time()-te:.1f}s "
                      f"val={row['val_mse']:.3f} test={row['test_mse']:.3f} "
                      f"({done}/{total} this session)", flush=True)
            print(f"[{seed}/{market}] block finished in {time.time()-t0:.1f}s", flush=True)
            del dm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
