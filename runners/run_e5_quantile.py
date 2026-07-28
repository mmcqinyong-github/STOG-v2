"""E5 quantile-head finetuning for 12 representative experts.

Protocol (mirrors run_e6_preds.py point-prediction phase):
  1) set_seed(seed); rebuild expert; retrain encoder+point head with the SAME
     MSE protocol (10 epochs, patience 3, lr 1e-4) -> reproduces the point-pred
     phase weights (same seed/data order; GPU nondeterminism aside).
  2) Freeze expert; attach a monotone quantile head on encode() features and
     train it with pinball loss (<=5 epochs, patience 2, lr 1e-3). Encoder is
     kept frozen so quantile training is a pure "head finetune" (zero change to
     the point-prediction representation).
  3) Save per-window quantile preds and penultimate (encode) features.

Resume-capable: skips (market, expert, seed) whose npz already exists.

Output: results/preds_quantile/{market}_{eid}_{seed}.npz
    - val_quant, test_quant [n, 24, 5] float32  (monotone, taus=.1/.25/.5/.75/.9)
    - val_encode, test_encode [n, 256] float16  (penultimate features, for F5)
    - taus [5]

Usage: python run_e5_quantile.py --markets NP,PJM,DE --seeds 2021,42,3407
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ensure_dir("./results/preds_quantile")

EXPERTS_12 = ["M03", "M52", "M47", "M63", "M17", "M14",
              "M50", "M18", "M31", "M55", "M233", "M89"]
TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]


class MonotoneQuantileHead(nn.Module):
    """Linear head on encode features with monotone quantile parametrization:
    q_1 = raw_1; q_k = q_{k-1} + softplus(raw_k)."""

    def __init__(self, hdim: int, horizon: int, n_taus: int):
        super().__init__()
        self.proj = nn.Linear(hdim, horizon * n_taus)
        self.horizon = horizon
        self.n_taus = n_taus

    def forward(self, h):
        B = h.shape[0]
        raw = self.proj(h).view(B, self.horizon, self.n_taus)
        base = raw[..., :1]
        inc = torch.nn.functional.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(inc, dim=-1)], dim=-1)


def pinball_loss(q, y, taus):
    """q: (B,H,Q), y: (B,H)."""
    err = y.unsqueeze(-1) - q  # (B,H,Q)
    t = torch.tensor(taus, device=q.device).view(1, 1, -1)
    return torch.maximum(t * err, (t - 1.0) * err).mean()


def train_quantile_head(expert, qhead, dm, device, taus,
                        max_epochs=5, patience=2, lr=1e-3, bs=256):
    for p in expert.parameters():
        p.requires_grad_(False)
    expert.eval()
    tr_in = dm.windows["train"].to(device)
    tr_tg = dm.windows["train_tgt"].to(device)
    va_in = dm.windows["val"].to(device)
    va_tg = dm.windows["val_tgt"].to(device)

    loader = DataLoader(TensorDataset(tr_in, tr_tg), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(qhead.parameters(), lr=lr)
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        qhead.train()
        for xb, yb in loader:
            opt.zero_grad()
            with torch.no_grad():
                h = expert.encode(xb)
            loss = pinball_loss(qhead(h), yb, taus)
            loss.backward()
            opt.step()
        qhead.eval()
        with torch.no_grad():
            v = pinball_loss(qhead(expert.encode(va_in)), va_tg, taus).item()
        if v < best_val - 1e-5:
            best_val, best_state, wait = v, {k: t.clone() for k, t in qhead.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        qhead.load_state_dict(best_state)
    qhead.eval()
    return best_val, epoch + 1


@torch.no_grad()
def infer_batched(expert, qhead, inp, device, bs=8192):
    qs, hs = [], []
    for i in range(0, inp.shape[0], bs):
        xb = inp[i:i + bs].to(device)
        h = expert.encode(xb)
        qs.append(qhead(h).cpu().numpy())
        hs.append(h.cpu().numpy())
    return np.concatenate(qs, 0).astype(np.float32), np.concatenate(hs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NP,PJM,DE")
    ap.add_argument("--seeds", default="2021,42,3407")
    ap.add_argument("--experts", default=",".join(EXPERTS_12))
    args = ap.parse_args()
    markets = args.markets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    expert_ids = args.experts.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for market in markets:
        for seed in seeds:
            t0 = time.time()
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed,
                               data_dir="./dataset/epf")
            dm.make_windows()
            dm.normalize()
            d_in = dm.windows["train"].shape[1]
            for eid in expert_ids:
                out_path = f"./results/preds_quantile/{market}_{eid}_{seed}.npz"
                if os.path.exists(out_path):
                    continue
                te = time.time()
                try:
                    set_seed(seed)
                    expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                    trainer = UnifiedTrainer({"max_epochs": 10, "patience": 3,
                                              "batch_size": 256, "lr": 1e-4})
                    res = trainer.train_expert(expert, dm)  # reproduce point phase
                    expert.to(device)
                    # quantile head on encode features
                    with torch.no_grad():
                        hdim = expert.encode(dm.windows["train"][:4].to(device)).shape[-1]
                    qhead = MonotoneQuantileHead(hdim, 24, len(TAUS)).to(device)
                    val_pin, q_epochs = train_quantile_head(expert, qhead, dm, device, TAUS)
                    va_q, va_h = infer_batched(expert, qhead, dm.windows["val"], device)
                    te_q, te_h = infer_batched(expert, qhead, dm.windows["test"], device)
                    np.savez(out_path,
                             val_quant=va_q, test_quant=te_q,
                             val_encode=va_h.astype(np.float16),
                             test_encode=te_h.astype(np.float16),
                             taus=np.array(TAUS, dtype=np.float32),
                             val_pinball=np.float32(val_pin),
                             point_val_mse=np.float32(res["val_mse"]),
                             point_test_mse=np.float32(res["test_mse"]))
                    print(f"  [{market}/{seed}] {eid} {time.time()-te:.1f}s "
                          f"pinball={val_pin:.4f} (mse={res['test_mse']:.2f})", flush=True)
                except Exception as ex:
                    print(f"  ERROR {market}/{eid}/{seed}: {ex}", flush=True)
            print(f"[{market}/{seed}] block {time.time()-t0:.1f}s", flush=True)
            del dm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
