"""E7-v2: long-horizon cross-task generalization (trimmed version).

Blocked by dataset x horizon; resume-capable at run granularity: every
(dataset, horizon, expert, seed) already present in e7v2_runs.csv is skipped,
each finished run is appended immediately, and --budget (seconds) makes the
script exit cleanly before the 300s Bash limit.

Protocol: chronological 0.7/0.1/0.2 split on raw rows (windows built inside
each split, no leakage), z-score with TRAIN-split statistics for both inputs
and the target channel (standard long-term protocol; target = last column),
L=336, H in {96, 720}, 10 epochs, patience=3, batch=256, lr=1e-4, hidden=256.

Per run saves per-window val/test errors to results/e7_v2/errs/*.npz (for
probe transfer analysis). For seed 2021 additionally saves raw test
predictions (for TopK routing fusion) and a per-block meta npz with
val/test targets + 12-dim probe features (same definitions as run_e6_preds).

Usage:
  python run_e7_v2.py --dataset ETTh1 --horizon 96 --budget 270
  python run_e7_v2.py --dataset ETTh1 --horizon 96 --seeds 2021 --experts M03,M52
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer
from run_e6_preds import compute_features, predict_batched


class BatchedEvalTrainer(UnifiedTrainer):
    """Protocol-identical to UnifiedTrainer; only val/test EVALUATION is
    computed in chunks (same MSE, avoids the one-shot giant attention alloc
    that stalls PatchTST-M50 on Weather-H720). Training loop untouched."""

    @staticmethod
    def _eval_mse(expert, inp, tgt, bs=4096):
        import torch.nn as nn
        crit = nn.MSELoss(reduction="sum")
        tot, n = 0.0, inp.shape[0]
        with torch.no_grad():
            for i in range(0, n, bs):
                p = expert(inp[i:i + bs])
                if p.dim() == 1:
                    p = p.unsqueeze(-1)
                tot += crit(p, tgt[i:i + bs]).item()
        return tot / (n * tgt.shape[1])

    def train_expert(self, expert, datamodule, verbose=False):
        import time as _t
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        expert = expert.to(self.device)
        train_inp = datamodule.windows["train"].to(self.device)
        train_tgt = datamodule.windows["train_tgt"].to(self.device)
        val_inp = datamodule.windows["val"].to(self.device)
        val_tgt = datamodule.windows["val_tgt"].to(self.device)
        test_inp = datamodule.windows["test"].to(self.device)
        test_tgt = datamodule.windows["test_tgt"].to(self.device)
        if train_tgt.dim() == 1:
            train_tgt = train_tgt.unsqueeze(-1)
            val_tgt = val_tgt.unsqueeze(-1)
            test_tgt = test_tgt.unsqueeze(-1)
        loader = DataLoader(TensorDataset(train_inp, train_tgt),
                            batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(expert.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        crit = nn.MSELoss()
        best_val, pat = float("inf"), 0
        t0 = _t.time()
        for epoch in range(self.max_epochs):
            expert.train()
            for xb, yb in loader:
                opt.zero_grad()
                pred = expert(xb)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
            expert.eval()
            val_mse = self._eval_mse(expert, val_inp, val_tgt)
            if val_mse < best_val:
                best_val, pat = val_mse, 0
            else:
                pat += 1
                if pat >= self.patience:
                    break
        expert.eval()
        test_pred = predict_batched(expert, datamodule.windows["test"],
                                    self.device)
        tt = test_tgt.cpu().numpy()
        test_mse = float(((test_pred - tt) ** 2).mean())
        test_mae = float(np.abs(test_pred - tt).mean())
        return {"val_mse": best_val, "test_mse": test_mse, "test_mae": test_mae,
                "epochs": epoch + 1, "time_sec": _t.time() - t0}

# E6 canonical 19 experts (same list as run_router.py; M36/M51 excluded)
EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
              "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
DATA_DIR = "./dataset/Long-term multivariate dataset"
DATASETS = {
    "ETTh1": f"{DATA_DIR}/ETT-small/ETTh1.csv",
    "ETTm1": f"{DATA_DIR}/ETT-small/ETTm1.csv",
    "Weather": f"{DATA_DIR}/weather/weather.csv",
}
SEEDS = [2021, 42, 3407]
TRAIN_CFG = {"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4}
OUT_DIR = "./results/e7_v2"
RUNS_CSV = f"{OUT_DIR}/e7v2_runs.csv"


def make_windows_chrono(data: np.ndarray, L: int, H: int):
    """Chronological 0.7/0.1/0.2 row split; windows within each split.
    Inputs z-scored per-feature with train stats; target (last column)
    z-scored with train target stats. Returns dict of tensors."""
    T = len(data)
    n_train = int(T * 0.7)
    n_val = int(T * 0.1)
    parts = {"train": data[:n_train],
             "val": data[n_train:n_train + n_val],
             "test": data[n_train + n_val:]}
    mu = parts["train"].mean(axis=0, keepdims=True)
    sd = parts["train"].std(axis=0, keepdims=True) + 1e-8
    out = {}
    for split, arr in parts.items():
        arrn = (arr - mu) / sd
        n_win = len(arrn) - L - H + 1
        if n_win <= 0:
            raise ValueError(f"split {split} too short: {len(arrn)} rows")
        sw = np.lib.stride_tricks.sliding_window_view(arrn, (L, arrn.shape[1]))[:, 0]
        X = sw[:n_win].reshape(n_win, -1).astype(np.float32)          # (n, L*V)
        tseries = arrn[:, -1]                                          # target col
        tsw = np.lib.stride_tricks.sliding_window_view(tseries, L + H)[:n_win]
        y = tsw[:, L:].astype(np.float32)                              # (n, H)
        xin = tsw[:, :L].astype(np.float64)                            # (n, L) target hist
        out[split] = torch.from_numpy(X)
        out[f"{split}_tgt"] = torch.from_numpy(y)
        out[f"{split}_feat"] = compute_features(xin)
    return out


def done_keys():
    if not os.path.exists(RUNS_CSV):
        return set()
    df = pd.read_csv(RUNS_CSV)
    return set(zip(df.dataset, df.horizon, df.expert_id, df.seed))


def append_row(row: dict):
    hdr = not os.path.exists(RUNS_CSV)
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=hdr, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--horizon", type=int, required=True, choices=[24, 96, 720])
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--experts", default=",".join(EXPERT_IDS))
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--budget", type=float, default=270.0)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="override protocol batch size (documented deviation)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated expert ids to skip this invocation")
    args = ap.parse_args()

    t_start = time.time()
    ds, H, L = args.dataset, args.horizon, args.lookback
    seeds = [int(s) for s in args.seeds.split(",")]
    eids = [e for e in args.experts.split(",") if e not in args.exclude.split(",")]
    ensure_dir(OUT_DIR)
    ensure_dir(f"{OUT_DIR}/errs")
    ensure_dir(f"{OUT_DIR}/preds")

    df = pd.read_csv(DATASETS[ds])
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    data = np.nan_to_num(df.values.astype(np.float32), nan=0.0)
    print(f"[E7v2] {ds} H={H} L={L} rows={len(data)} vars={data.shape[1]}", flush=True)

    W = make_windows_chrono(data, L, H)
    d_in = W["train"].shape[1]
    print(f"[E7v2] d_in={d_in} n_train={len(W['train'])} n_val={len(W['val'])} "
          f"n_test={len(W['test'])}", flush=True)

    # per-block meta (targets + probe features) saved once
    meta_path = f"{OUT_DIR}/preds/meta_{ds}_H{H}.npz"
    if not os.path.exists(meta_path):
        np.savez_compressed(
            meta_path,
            val_true=W["val_tgt"].numpy(), test_true=W["test_tgt"].numpy(),
            feat_val=W["val_feat"], feat_test=W["test_feat"])

    class DM:
        pass
    dm = DM()
    dm.windows = {k: v for k, v in W.items() if not k.endswith("_feat")}

    done = done_keys()
    cfg = dict(TRAIN_CFG)
    if args.batch_size != 256:
        cfg["batch_size"] = args.batch_size
        print(f"[E7v2] DEVIATION: batch_size={args.batch_size}", flush=True)
    trainer = BatchedEvalTrainer(cfg)
    device = trainer.device

    for seed in seeds:
        for eid in eids:
            if (ds, H, eid, seed) in done:
                continue
            if time.time() - t_start > args.budget:
                print(f"[E7v2] budget reached, exiting cleanly", flush=True)
                return
            set_seed(seed)
            t0 = time.time()
            err_path = f"{OUT_DIR}/errs/{ds}_H{H}_{seed}_{eid}.npz"
            try:
                expert = get_expert(eid, d_in, hidden=256, drop=0.1, horizon=H)
                res = trainer.train_expert(expert, dm)
                # per-window errors for probe-transfer analysis
                val_pred = predict_batched(expert, W["val"], device)
                test_pred = predict_batched(expert, W["test"], device)
                val_err = ((val_pred - W["val_tgt"].numpy()) ** 2).mean(axis=1)
                test_err = ((test_pred - W["test_tgt"].numpy()) ** 2).mean(axis=1)
                np.savez_compressed(err_path, val_err=val_err.astype(np.float32),
                                    test_err=test_err.astype(np.float32))
                if seed == 2021:
                    np.savez_compressed(
                        f"{OUT_DIR}/preds/{ds}_H{H}_{eid}_2021.npz",
                        test_pred=test_pred.astype(np.float32))
                row = {"dataset": ds, "horizon": H, "expert_id": eid, "seed": seed,
                       "val_mse": res["val_mse"], "test_mse": res["test_mse"],
                       "test_mae": res["test_mae"], "epochs": res["epochs"],
                       "time_sec": res["time_sec"]}
                status = f"val={res['val_mse']:.4f} test={res['test_mse']:.4f}"
            except Exception as ex:
                row = {"dataset": ds, "horizon": H, "expert_id": eid, "seed": seed,
                       "val_mse": 9999.0, "test_mse": 9999.0, "test_mae": 9999.0,
                       "epochs": 0, "time_sec": time.time() - t0,
                       "error": str(ex)[:200]}
                status = f"ERROR {str(ex)[:80]}"
            append_row(row)
            done.add((ds, H, eid, seed))
            torch.cuda.empty_cache()
            print(f"[E7v2] {ds}/H{H}/{eid}/s{seed}: {status} "
                  f"({time.time()-t0:.1f}s, total {time.time()-t_start:.0f}s)",
                  flush=True)

    print(f"[E7v2] block {ds}/H{H} complete in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
