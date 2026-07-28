#!/usr/bin/env python3
"""Analysis v3 / Task 4b: (1) validate the v2-hedge replication against the
saved run_e9_v2 rows files; (2) compute the NORMALIZED-scale cumulative regret
of the v2 hedge (the space where losses live in [0,1] and the O(sqrt(T ln N))
bound applies) and test its sqrt(T) growth — empirical support for the SI
sentence "clip 后有界损失归约保持 O(√(T ln N))".
"""
import numpy as np
import pandas as pd

EXPERT_IDS = ["M47", "M63", "M03", "M18", "M31", "M89", "M50", "M233",
              "M17", "M220"]
N_MONTHS = 12
WEIGHT_FLOOR = 1e-4
N = len(EXPERT_IDS)
ETA = np.sqrt(8.0 * np.log(N) / N_MONTHS)

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)) + 1e-300)

def norm_weights(logw, floor=WEIGHT_FLOOR):
    w = np.exp(logw - logsumexp(logw))
    w = np.maximum(w, floor)
    return w / w.sum()

def norm_regret_form(losses):
    lo = losses.min()
    hi = np.quantile(losses, 0.95)
    rng = max(hi - lo, 1e-8)
    return np.clip((losses - lo) / rng, 0.0, 1.0)

def winsorize(hist):
    all_l = np.concatenate(hist)
    cap = max(np.quantile(all_l, 0.95), 1e-8)
    return np.minimum(hist[-1], cap), cap

# ---------- (1) validation against saved rows files ----------
val_rows = []
for market in ["NP", "PJM", "BE", "FR", "DE"]:
    for seed in [2021, 42, 3407]:
        import os
        rows_path = f"results/e9_v2/rows_{market}_{seed}.csv"
        if not os.path.exists(rows_path):
            continue                      # e9_v2 finished NP/DE blocks only
        saved = pd.read_csv(rows_path)
        saved_h = saved[saved.strategy == "hedge"].sort_values("month")
        meta = np.load(f"results/preds/meta_{market}_{seed}.npz")
        test_true = meta["test_true"].astype(np.float64)
        n_test = test_true.shape[0]
        bs_ = n_test // N_MONTHS
        slices = [slice(i * bs_, min((i + 1) * bs_, n_test)) for i in range(N_MONTHS)]
        P = np.zeros((N, n_test, 24))
        for i, eid in enumerate(EXPERT_IDS):
            P[i] = np.load(f"results/preds/{market}_{eid}_{seed}.npz")["test_pred"].astype(np.float64)
        E10 = np.zeros((N, N_MONTHS))
        for m, sl in enumerate(slices):
            E10[:, m] = ((P[:, sl, :] - test_true[sl][None]) ** 2).mean(axis=(1, 2))
        logw = np.log(np.ones(N) / N)
        hist = []
        rep_raw, rep_norm_loss = [], []
        for m in range(N_MONTHS):
            raw = E10[:, m].copy()
            hist.append(raw.copy())
            win, cap = winsorize(hist)
            r = norm_regret_form(win)
            w = norm_weights(logw)
            ens = np.tensordot(w, P[:, slices[m], :], axes=1)
            rep_raw.append(float(((ens - test_true[slices[m]]) ** 2).mean()))
            rep_norm_loss.append(float(np.dot(w, r)))
            logw += -ETA * r
            logw -= logsumexp(logw)
        d_raw = np.abs(np.array(rep_raw) - saved_h.loss_raw.to_numpy()).max()
        d_norm = np.abs(np.array(rep_norm_loss) - saved_h.loss_norm.to_numpy()).max()
        val_rows.append({"market": market, "seed": seed,
                         "max_abs_diff_loss_raw": d_raw,
                         "max_abs_diff_loss_norm": d_norm})
vt = pd.DataFrame(val_rows)
print("replication validation (max abs diff per block):")
print(vt.to_string(index=False))
print("overall max:", vt.max_abs_diff_loss_raw.max(), vt.max_abs_diff_loss_norm.max())

# ---------- (2) normalized-scale regret of v2 hedge vs sqrt(T ln N /2) ------
reg_rows = []
for market in ["NP", "PJM", "BE", "FR", "DE"]:
    for seed in [2021, 42, 3407]:
        meta = np.load(f"results/preds/meta_{market}_{seed}.npz")
        test_true = meta["test_true"].astype(np.float64)
        n_test = test_true.shape[0]
        bs_ = n_test // N_MONTHS
        slices = [slice(i * bs_, min((i + 1) * bs_, n_test)) for i in range(N_MONTHS)]
        P = np.zeros((N, n_test, 24))
        vm = np.zeros(N)
        val_true = meta["val_true"].astype(np.float64)
        for i, eid in enumerate(EXPERT_IDS):
            d = np.load(f"results/preds/{market}_{eid}_{seed}.npz")
            P[i] = d["test_pred"].astype(np.float64)
            vm[i] = float(((d["val_pred"].astype(np.float64) - val_true) ** 2).mean())
        E10 = np.zeros((N, N_MONTHS))
        for m, sl in enumerate(slices):
            E10[:, m] = ((P[:, sl, :] - test_true[sl][None]) ** 2).mean(axis=(1, 2))
        best_idx = int(np.argmin(vm))          # fixed arm = val-best expert
        logw = np.log(np.ones(N) / N)
        hist = []
        cum_reg = 0.0
        for m in range(N_MONTHS):
            raw = E10[:, m].copy()
            hist.append(raw.copy())
            win, cap = winsorize(hist)
            r = norm_regret_form(win)
            w = norm_weights(logw)
            cum_reg += float(np.dot(w, r)) - float(r[best_idx])
            reg_rows.append({"market": market, "seed": seed, "month": m,
                             "cum_regret_norm": cum_reg,
                             "bound": np.sqrt((m + 1) * np.log(N) / 2.0)})
            logw += -ETA * r
            logw -= logsumexp(logw)
rg = pd.DataFrame(reg_rows)
fin = rg[rg.month == N_MONTHS - 1]
print("\nnormalized-scale final cum regret vs bound sqrt(T ln N / 2)=3.717:")
print(fin.groupby("market").cum_regret_norm.agg(["mean", "max"]).round(3).to_string())
print(f"overall: mean {fin.cum_regret_norm.mean():.3f}, max {fin.cum_regret_norm.max():.3f}, "
      f"blocks within bound: {(fin.cum_regret_norm <= fin.bound).mean()*100:.0f}%")

# sqrt(T) growth test: fit mean cum_regret(T) = a * sqrt(T) + b, report R2
mean_curve = rg.groupby("month").cum_regret_norm.mean()
T = (mean_curve.index + 1).to_numpy(float)
A = np.vstack([np.sqrt(T), np.ones_like(T)]).T
coef, res, *_ = np.linalg.lstsq(A, mean_curve.to_numpy(), rcond=None)
pred = A @ coef
r2 = 1 - np.sum((mean_curve.to_numpy() - pred) ** 2) / np.sum(
    (mean_curve.to_numpy() - mean_curve.mean()) ** 2)
# compare with linear-in-T fit
A2 = np.vstack([T, np.ones_like(T)]).T
coef2, *_ = np.linalg.lstsq(A2, mean_curve.to_numpy(), rcond=None)
pred2 = A2 @ coef2
r2_lin = 1 - np.sum((mean_curve.to_numpy() - pred2) ** 2) / np.sum(
    (mean_curve.to_numpy() - mean_curve.mean()) ** 2)
print(f"\ngrowth fit: cum_regret ≈ {coef[0]:.3f}*sqrt(T) + {coef[1]:.3f} (R2={r2:.3f}) "
      f"vs linear-in-T R2={r2_lin:.3f}")
print("mean curve:", mean_curve.round(3).to_dict())

with open("results/analysis_v3/e9_hedge_validation_and_bound.csv", "w") as f:
    f.write("# === replication validation vs saved rows files ===\n")
    vt.to_csv(f, index=False)
    f.write("\n# === normalized-scale cumulative regret of v2 hedge ===\n")
    rg.to_csv(f, index=False)
    f.write(f"\n# sqrt(T) fit a={coef[0]:.4f} b={coef[1]:.4f} R2={r2:.4f}; "
            f"linear R2={r2_lin:.4f}\n")
