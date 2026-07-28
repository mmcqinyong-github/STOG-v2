"""P3 Task 1: Statistical baselines B15 (biseasonal MSTL-lite) and B16 (LEAR-ridge) on EPF.

Protocol-matched to E6:
- same windows, same shuffle (RandomState(2021)), same test split (20%)
- test MSE/MAE computed on RAW price scale (targets are unnormalized in E6 trainer)
- B15: biseasonal MSTL decomposition (weekly 168 + daily 24) + drift extrapolation,
  fit per test window on the past 672h of history (rolling-origin, no future leakage)
- B16: LEAR-lite = multi-output ridge regression on [168 price lags, dow one-hot, hod one-hot],
  lambda selected on val windows
Also verifies alignment by reproducing E6 expert MSE from stored npz preds.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd

EPF_DIR = "./dataset/epf"
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
L, H = 168, 24
SPLIT_SEED = 2021
OUT_DIR = "./results/p3"
FIG_DIR = "./results/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_price_series(market):
    df = pd.read_csv(os.path.join(EPF_DIR, f"{market}.csv"))
    df.columns = [c.strip() for c in df.columns]
    price_col = "OT" if "OT" in df.columns else df.columns[-1]
    return df[price_col].values.astype(np.float64)


def make_split_indices(n):
    idx = np.arange(n)
    np.random.RandomState(SPLIT_SEED).shuffle(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def verify_alignment(market, test_idx, x):
    """Reproduce one E6 expert's test MSE from stored npz to confirm protocol alignment."""
    path = f"./results/preds/{market}_M01_2021.npz"
    if not os.path.exists(path):
        return None
    pred = np.load(path)["test_pred"].astype(np.float64)
    tgt = np.stack([x[i + L:i + L + H] for i in test_idx])
    if pred.shape != tgt.shape:
        return {"shape_mismatch": (pred.shape, tgt.shape)}
    return float(((pred - tgt) ** 2).mean())


# ---------------- B15: biseasonal MSTL-lite ----------------
def b15_forecast(x, origins, hist_len=672):
    """Vectorized biseasonal decomposition forecast.

    For each origin t0 (exclusive), use history x[t0-hist_len:t0].
    weekly profile (period 168) + daily residual profile (period 24) + drift.
    Returns (n_windows, 24) forecasts.
    """
    n = len(origins)
    out = np.empty((n, H), dtype=np.float64)
    for j, t0 in enumerate(origins):
        t0 = int(t0)
        h0 = max(0, t0 - hist_len)
        hist = x[h0:t0]
        if len(hist) < 168:  # pad at series start by repeating edge
            hist = np.concatenate([np.full(168 - len(hist), hist[0]), hist])
        # weekly profile: align by absolute timestamp mod 168
        week_phase = np.arange(h0, t0) % 168
        wp = np.zeros(168)
        for k in range(168):
            vals = hist[week_phase == k]
            wp[k] = vals.mean() if len(vals) else hist.mean()
        resid = hist - wp[week_phase]
        # daily profile on residual
        day_phase = np.arange(h0, t0) % 24
        dd = np.zeros(24)
        for m in range(24):
            vals = resid[day_phase == m]
            dd[m] = vals.mean() if len(vals) else 0.0
        # drift: slope from last two days, damped
        if len(hist) >= 48:
            slope = (hist[-24:].mean() - hist[-48:-24].mean()) / 24.0
        else:
            slope = 0.0
        slope = np.clip(slope, -np.std(hist) / 4, np.std(hist) / 4)
        h_idx = np.arange(H)
        ts = t0 + h_idx
        out[j] = wp[ts % 168] + dd[ts % 24] + slope * (h_idx + 1) * 0.5
    return out


def b15_forecast_fast(x, origins, hist_len=672):
    """Faster binned version of b15_forecast using np.bincount."""
    n = len(origins)
    out = np.empty((n, H), dtype=np.float64)
    h_idx = np.arange(H)
    for j, t0 in enumerate(origins):
        t0 = int(t0)
        h0 = max(0, t0 - hist_len)
        hist = x[h0:t0]
        if len(hist) < 168:
            hist = np.concatenate([np.full(168 - len(hist), hist[0]), hist])
            h0 = t0 - len(hist)
        ts_hist = np.arange(h0, t0)
        wk = ts_hist % 168
        cnt_w = np.bincount(wk, minlength=168)
        sum_w = np.bincount(wk, weights=hist, minlength=168)
        wp = sum_w / np.maximum(cnt_w, 1)
        wp[cnt_w == 0] = hist.mean()
        resid = hist - wp[wk]
        dy = ts_hist % 24
        cnt_d = np.bincount(dy, minlength=24)
        sum_d = np.bincount(dy, weights=resid, minlength=24)
        dd = sum_d / np.maximum(cnt_d, 1)
        if len(hist) >= 48:
            slope = (hist[-24:].mean() - hist[-48:-24].mean()) / 24.0
        else:
            slope = 0.0
        slope = np.clip(slope, -np.std(hist) / 4, np.std(hist) / 4)
        ts = t0 + h_idx
        out[j] = wp[ts % 168] + dd[ts % 24] + slope * (h_idx + 1) * 0.5
    return out


# ---------------- B16: LEAR-lite (multi-output ridge) ----------------
def build_features(x, idxs):
    """Features: 168 raw lags + dow one-hot(7) of forecast day + hod one-hot(24) of origin."""
    n = len(idxs)
    Xl = np.stack([x[i:i + L] for i in idxs])
    dow = ((idxs + L) // 24) % 7
    hod = (idxs + L) % 24
    Xd = np.zeros((n, 7)); Xd[np.arange(n), dow] = 1.0
    Xh = np.zeros((n, 24)); Xh[np.arange(n), hod] = 1.0
    return np.concatenate([Xl, Xd, Xh], axis=1)


def b16_lear(x, train_idx, val_idx, test_idx):
    Xtr = build_features(x, train_idx)
    Ytr = np.stack([x[i + L:i + L + H] for i in train_idx])
    Xva = build_features(x, val_idx)
    Yva = np.stack([x[i + L:i + L + H] for i in val_idx])
    Xte = build_features(x, test_idx)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    Xte_n = (Xte - mu) / sd
    # center targets (intercept) — features are already standardized
    ymu = Ytr.mean(axis=0)
    best_lam, best_vmse, best_W = None, np.inf, None
    for lam in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        A = Xtr_n.T @ Xtr_n + lam * np.eye(Xtr_n.shape[1])
        B = Xtr_n.T @ (Ytr - ymu)
        W = np.linalg.solve(A, B)
        vmse = ((Xva_n @ W + ymu - Yva) ** 2).mean()
        if vmse < best_vmse:
            best_vmse, best_lam, best_W = vmse, lam, W
    pred = Xte_n @ best_W + ymu
    return pred, best_lam


def main():
    t_start = time.time()
    e6 = pd.read_csv("./results/e6_epf_main.csv")
    e6_mean = e6.groupby(["market", "expert_id"])["test_mse"].mean().reset_index()

    rows = []
    for market in MARKETS:
        t0 = time.time()
        x = load_price_series(market)
        n = len(x) - L - H + 1
        all_idx = np.arange(n)
        train_idx, val_idx, test_idx = make_split_indices(n)

        # protocol verification
        v = verify_alignment(market, test_idx, x)
        print(f"[{market}] n={n} test={len(test_idx)} | M01 repro MSE={v}")

        tgt_test = np.stack([x[i + L:i + L + H] for i in test_idx])

        # B15
        p15 = b15_forecast_fast(x, test_idx + L)  # origins = i + L (exclusive)
        mse15 = float(((p15 - tgt_test) ** 2).mean())
        mae15 = float(np.abs(p15 - tgt_test).mean())
        # B16
        p16, lam = b16_lear(x, train_idx, val_idx, test_idx)
        mse16 = float(((p16 - tgt_test) ** 2).mean())
        mae16 = float(np.abs(p16 - tgt_test).mean())
        print(f"[{market}] B15 MSE={mse15:.3f} MAE={mae15:.3f} | B16 MSE={mse16:.3f} MAE={mae16:.3f} (lam={lam}) | {time.time()-t0:.1f}s")

        # ranking among 19 experts + B15 + B16
        em = e6_mean[e6_mean["market"] == market].copy()
        methods = list(zip(em["expert_id"], em["test_mse"])) + [("B15_MSTL", mse15), ("B16_LEAR", mse16)]
        methods.sort(key=lambda t: t[1])
        for rank, (name, mse) in enumerate(methods, start=1):
            if name in ("B15_MSTL", "B16_LEAR"):
                rows.append({"market": market, "method": name, "test_mse": mse,
                             "test_mae": mae15 if name == "B15_MSTL" else mae16,
                             "rank_among_21": rank, "n_methods": len(methods),
                             "best_expert": methods[0][0], "best_expert_mse": methods[0][1]})
        # store full ranking table for figure
        for rank, (name, mse) in enumerate(methods, start=1):
            rows.append({"market": market, "method": f"ALL::{name}", "test_mse": mse,
                         "test_mae": np.nan, "rank_among_21": rank, "n_methods": len(methods),
                         "best_expert": methods[0][0], "best_expert_mse": methods[0][1]})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "b15_b16_baselines.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} | total {time.time()-t_start:.1f}s")
    print(df[df["method"].isin(["B15_MSTL", "B16_LEAR"])].to_string(index=False))

    # ---- figure: ranking bar chart ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharex=False)
    for ax, market in zip(axes, MARKETS):
        sub = df[(df["market"] == market) & df["method"].str.startswith("ALL::")].copy()
        sub["name"] = sub["method"].str.replace("ALL::", "", regex=False)
        sub = sub.sort_values("rank_among_21")
        colors = ["#d62728" if n == "B15_MSTL" else "#ff7f0e" if n == "B16_LEAR" else "#9ecae1"
                  for n in sub["name"]]
        ax.barh(sub["name"], sub["test_mse"], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"{market} (B15 rank {int(sub[sub['name']=='B15_MSTL']['rank_among_21'].iloc[0])}/21)")
        ax.set_xlabel("test MSE (raw scale)")
        ax.tick_params(axis='y', labelsize=7)
    fig.suptitle("P3: B15 biseasonal-MSTL / B16 LEAR vs 19 deep experts — per-market ranking")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "p3_mstl_baseline_ranking.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
