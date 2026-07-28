"""E5 fusion-geometry comparison (Theorem 5b/5c) -- zero extra heavy training.

Fusions (test split; E=12 quantile experts):
  F1 Vincentization : W2 barycenter approx = per-quantile average over experts
  F2 LinearPool     : mixture CDF (probability averaging), exact for
                      piecewise-linear CDFs, re-inverted at the 5 taus
  F3 MedianPool     : per-quantile median over experts
  F4 OutputWeighted : simplex weights minimizing VAL pinball, applied to test
  F5 HiddenFusion   : ridge regression on concatenated penultimate (encode)
                      features (val-fit) -> point forecast. LIGHTWEIGHT
                      APPROXIMATION of representation-level fusion (no shared
                      head retraining); reported as such.

Metrics on test: CRPS (trapezoidal pinball integral approx), pinball (mean),
WQL (sum pinball / sum |y|), MSE of the median quantile (point).

Outputs:
  results/e5/e5_fusion_comparison.csv  (fusion x market x seed x metric)
  results/e5/e5_fusion_summary.csv     (mean/std across market-seeds)

Resume-capable: skips fusion/market/seed rows already in the comparison csv.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import ensure_dir

ensure_dir("./results/e5")

EXPERTS_12 = ["M03", "M52", "M47", "M63", "M17", "M14",
              "M50", "M18", "M31", "M55", "M233", "M89"]
TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
# trapezoidal weights for CRPS integral over [0,1]
_B = np.concatenate([[0.0], (TAUS[:-1] + TAUS[1:]) / 2, [1.0]])
CRPS_W = np.diff(_B)  # sums to 1


# ---------------- metrics ----------------

def pinball_per_tau(q, y):
    """q: (...,H,Q), y: (...,H) -> (Q,) mean pinball per tau."""
    err = y[..., None] - q
    rho = np.maximum(TAUS * err, (TAUS - 1.0) * err)
    return rho.reshape(-1, len(TAUS)).mean(axis=0)


def metric_row(q, y):
    """q: (n,H,Q), y: (n,H)."""
    rho = pinball_per_tau(q, y)
    pin = float(rho.mean())
    crps = float(2.0 * (CRPS_W * rho).sum())
    absy = float(np.abs(y).sum()) + 1e-12
    wql = float((rho * y.size).sum() / absy)  # mean_tau( sum rho / sum|y| )
    med = q[..., int(np.argmin(np.abs(TAUS - 0.5)))]
    mse = float(((med - y) ** 2).mean())
    return {"crps": crps, "pinball": pin, "wql": wql, "mse": mse}


# ---------------- fusions ----------------

def f1_vincent(Q):
    return Q.mean(axis=0)


def f3_median(Q):
    return np.median(Q, axis=0)


def _cdf_at(Qe, ys):
    """Evaluate piecewise-linear CDF of each expert at union points ys.
    Qe: (E,n,H,Qt) quantile values; ys: (n,H,G) sorted union grid.
    Returns F: (E,n,H,G). Linear extrapolation outside [q_lo, q_hi], clamped."""
    E, n, H, Qt = Qe.shape
    G = ys.shape[-1]
    F = np.empty((E, n, H, G), dtype=np.float32)
    for e in range(E):
        q = Qe[e]  # (n,H,Qt)
        # index of interval for each grid point
        idx = np.sum(ys[..., None] >= q[..., None, :], axis=-1)  # (n,H,G) in 0..Qt
        idx = np.clip(idx, 1, Qt - 1)
        q0 = np.take_along_axis(q, idx - 1, axis=-1)
        q1 = np.take_along_axis(q, idx, axis=-1)
        t0 = TAUS[idx - 1]
        t1 = TAUS[idx]
        slope = (t1 - t0) / np.maximum(q1 - q0, 1e-8)
        f = t0 + (ys - q0) * slope
        F[e] = np.clip(f, 0.0, 1.0)
    return F


def f2_linearpool(Q, chunk=1024):
    """Exact mixture-CDF pool for piecewise-linear expert CDFs.
    Q: (E,n,H,Qt). Returns quantiles at TAUS of the averaged CDF."""
    E, n, H, Qt = Q.shape
    out = np.empty((n, H, Qt), dtype=np.float32)
    for s in range(0, n, chunk):
        Qc = Q[:, s:s + chunk]  # (E,c,H,Qt)
        c = Qc.shape[1]
        # union grid: all expert quantile values, sorted
        U = np.sort(Qc.transpose(1, 2, 0, 3).reshape(c, H, E * Qt), axis=-1)
        F = _cdf_at(Qc, U).mean(axis=0)  # (c,H,G)
        # invert at TAUS: for each tau find interval in F
        idx = np.sum(F[..., None] <= TAUS.reshape(1, 1, 1, Qt), axis=-2)  # (c,H,Qt)
        G = U.shape[-1]
        idx = np.clip(idx, 1, G - 1)
        f0 = np.take_along_axis(F, idx - 1, axis=-1)  # wait: F is (c,H,G), idx is (c,H,Qt)
        # need gather along G axis: use np.take_along_axis with idx arrays
        f0 = np.take_along_axis(F, idx - 1, axis=-1)
        f1 = np.take_along_axis(F, idx, axis=-1)
        u0 = np.take_along_axis(U, idx - 1, axis=-1)
        u1 = np.take_along_axis(U, idx, axis=-1)
        w = (TAUS.reshape(1, 1, Qt) - f0) / np.maximum(f1 - f0, 1e-8)
        out[s:s + chunk] = (u0 + np.clip(w, 0, 1) * (u1 - u0)).astype(np.float32)
    return out


def f4_output_weighted(Qv, yv, Qt, device, steps=300, lr=0.05):
    """Simplex weights minimizing val pinball; apply to test quantiles."""
    Qv_t = torch.tensor(Qv, device=device)  # (E,nv,H,Qt)
    yv_t = torch.tensor(yv, device=device)
    taus_t = torch.tensor(TAUS, device=device)
    logits = torch.zeros(Qv.shape[0], device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(logits, dim=0)
        q = (w.view(-1, 1, 1, 1) * Qv_t).sum(0)
        err = yv_t.unsqueeze(-1) - q
        loss = torch.maximum(taus_t * err, (taus_t - 1) * err).mean()
        loss.backward()
        opt.step()
    w = torch.softmax(logits, dim=0).detach().cpu().numpy()
    return (w.reshape(-1, 1, 1, 1) * Qt).sum(0), w


def f5_hidden_ridge(Hv, yv, Ht, alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)):
    """Ridge from concatenated standardized penultimate features -> 24h point.
    y is centered by the val mean (intercept). Alpha picked by 80/20 val split."""
    mu, sd = Hv.mean(0), Hv.std(0) + 1e-6
    Xv = (Hv - mu) / sd
    Xt = (Ht - mu) / sd
    ym = yv.mean(axis=0)
    yc = yv - ym
    n = Xv.shape[0]
    ntr = int(n * 0.8)
    Xtr, Xho = Xv[:ntr], Xv[ntr:]
    ytr, yho = yc[:ntr], yc[ntr:]
    G = Xtr.T @ Xtr
    best_a, best_mse = alphas[0], np.inf
    for a in alphas:
        W = np.linalg.solve(G + a * ntr * np.eye(G.shape[0]), Xtr.T @ ytr)
        mse = ((Xho @ W - yho) ** 2).mean()
        if mse < best_mse:
            best_mse, best_a = mse, a
    G = Xv.T @ Xv
    W = np.linalg.solve(G + best_a * n * np.eye(G.shape[0]), Xv.T @ yc)
    return Xt @ W + ym, best_a


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NP,PJM,DE")
    ap.add_argument("--seeds", default="2021,42,3407")
    args = ap.parse_args()
    markets, seeds = args.markets.split(","), [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cmp_path = "./results/e5/e5_fusion_comparison.csv"
    done = set()
    if os.path.exists(cmp_path):
        d0 = pd.read_csv(cmp_path)
        done = set(zip(d0.fusion, d0.market, d0.seed))
    rows, w_rows = [], []

    for market in markets:
        for seed in seeds:
            meta = np.load(f"./results/preds/meta_{market}_{seed}.npz")
            yv = meta["val_true"].astype(np.float64)
            yt = meta["test_true"].astype(np.float64)
            Qv, Qt, Hv, Ht = [], [], [], []
            for eid in EXPERTS_12:
                d = np.load(f"./results/preds_quantile/{market}_{eid}_{seed}.npz")
                Qv.append(d["val_quant"]); Qt.append(d["test_quant"])
                Hv.append(d["val_encode"].astype(np.float32))
                Ht.append(d["test_encode"].astype(np.float32))
            Qv = np.stack(Qv).astype(np.float64)  # (E,nv,24,5)
            Qt_ = np.stack(Qt).astype(np.float64)
            Hv = np.concatenate(Hv, axis=1).astype(np.float64)
            Ht = np.concatenate(Ht, axis=1).astype(np.float64)

            t0 = time.time()
            # oracle single expert (test-selected; upper reference, labeled oracle)
            per_exp = [metric_row(Qt_[e], yt)["crps"] for e in range(len(EXPERTS_12))]
            cand = {}
            cand["F1_Vincentization"] = f1_vincent(Qt_)
            cand["F2_LinearPool"] = f2_linearpool(Qt_)
            cand["F3_MedianPool"] = f3_median(Qt_)
            q4, w4 = f4_output_weighted(Qv, yv, Qt_, device)
            cand["F4_OutputWeighted"] = q4
            w_rows.append({"market": market, "seed": seed,
                           **{e: round(float(x), 4) for e, x in zip(EXPERTS_12, w4)}})
            e_best = int(np.argmin(per_exp))
            cand["OracleBestSingle"] = Qt_[e_best]

            for name, q in cand.items():
                if (name, market, seed) in done:
                    continue
                m = metric_row(q.astype(np.float64), yt)
                rows.append({"fusion": name, "market": market, "seed": seed, **m})

            if ("F5_HiddenRidge", market, seed) not in done:
                p5, a5 = f5_hidden_ridge(Hv, yv, Ht)
                mse5 = float(((p5 - yt) ** 2).mean())
                rows.append({"fusion": "F5_HiddenRidge", "market": market, "seed": seed,
                             "crps": np.nan, "pinball": np.nan, "wql": np.nan,
                             "mse": mse5})
                print(f"  [{market}/{seed}] F5 ridge alpha={a5} mse={mse5:.3f}", flush=True)
            print(f"[{market}/{seed}] fusions in {time.time()-t0:.1f}s "
                  f"(oracle={EXPERTS_12[e_best]})", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(cmp_path, mode="a", index=False,
                                  header=not os.path.exists(cmp_path))
    if w_rows:
        wpath = "./results/e5/e5_f4_weights.csv"
        pd.DataFrame(w_rows).to_csv(wpath, mode="a", index=False,
                                    header=not os.path.exists(wpath))

    df = pd.read_csv(cmp_path)
    summ = df.groupby("fusion").agg(
        crps_mean=("crps", "mean"), crps_std=("crps", "std"),
        pinball_mean=("pinball", "mean"), pinball_std=("pinball", "std"),
        wql_mean=("wql", "mean"), wql_std=("wql", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        n=("mse", "count")).reset_index()
    summ.to_csv("./results/e5/e5_fusion_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
