#!/usr/bin/env python3
"""Analysis v3 / Task 4: standard (unnormalized) Hedge control arm, zero-training
simulation on the saved NP/PJM/BE/FR/DE predictions used by run_e9_v2.py.

Two arms on the identical 12-month loss stream (10 experts):
  - v2 hedge    : exact replication of run_e9_v2.py (rolling-q95 winsorize ->
                  [0,1] regret-form -> log-space update with eta=sqrt(8 lnN/T)
                  -> normalize with 1e-4 weight floor).
  - std hedge   : textbook Hedge: logw += -eta * raw_loss, log-space
                  (logsumexp), NO winsorize, NO regret-form normalization,
                  NO weight floor. Same eta, same ensemble decision rule
                  (weighted-prediction MSE) so only the weight update differs.

Also recorded: per-month weight entropy / max weight (one-hot collapse
diagnostics), oracle & fixed references, and the bounded-loss regret bound
sqrt(T ln N / 2) for the SI statement.
"""
import os
import numpy as np
import pandas as pd

OUT = "results/analysis_v3"
os.makedirs(OUT, exist_ok=True)

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

rows = []
summ = []
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

        best_idx = int(np.argmin(vm))
        logw_v2 = np.log(np.ones(N) / N)
        logw_std = np.log(np.ones(N) / N)
        hist = []
        cum = {"v2": 0.0, "std": 0.0, "fixed": 0.0, "oracle": 0.0}
        cum_reg = {"v2": 0.0, "std": 0.0}   # vs fixed (raw loss scale)
        first_collapse = None
        for m in range(N_MONTHS):
            raw = E10[:, m].copy()
            hist.append(raw.copy())
            win, cap = winsorize(hist)
            r = norm_regret_form(win)
            w_v2 = norm_weights(logw_v2)
            w_std = np.exp(logw_std - logsumexp(logw_std))   # no floor
            if first_collapse is None and w_std.max() > 0.99:
                first_collapse = m
            ens_v2 = np.tensordot(w_v2, P[:, slices[m], :], axes=1)
            ens_std = np.tensordot(w_std, P[:, slices[m], :], axes=1)
            l_v2 = float(((ens_v2 - test_true[slices[m]]) ** 2).mean())
            l_std = float(((ens_std - test_true[slices[m]]) ** 2).mean())
            l_fix = float(raw[best_idx]); l_or = float(raw.min())
            cum["v2"] += l_v2; cum["std"] += l_std
            cum["fixed"] += l_fix; cum["oracle"] += l_or
            cum_reg["v2"] += l_v2 - l_fix; cum_reg["std"] += l_std - l_fix
            ent = lambda w: float(-(w * np.log(w + 1e-300)).sum() / np.log(N))
            rows.append({"market": market, "seed": seed, "month": m,
                         "loss_v2": l_v2, "loss_std": l_std,
                         "loss_fixed": l_fix, "loss_oracle": l_or,
                         "regret_v2_vs_fixed": l_v2 - l_fix,
                         "regret_std_vs_fixed": l_std - l_fix,
                         "entropy_v2": ent(w_v2), "entropy_std": ent(w_std),
                         "maxw_v2": float(w_v2.max()), "maxw_std": float(w_std.max()),
                         "cum_regret_v2": cum_reg["v2"],
                         "cum_regret_std": cum_reg["std"],
                         "bound_sqrt_TlnN2": np.sqrt((m + 1) * np.log(N) / 2)})
            logw_v2 += -ETA * r
            logw_v2 -= logsumexp(logw_v2)
            logw_std += -ETA * raw          # textbook: raw losses directly
            logw_std -= logsumexp(logw_std)
        order = sorted(["v2", "std", "fixed", "oracle"], key=lambda s: cum[s])
        summ.append({"market": market, "seed": seed,
                     "cum_v2": cum["v2"], "cum_std": cum["std"],
                     "cum_fixed": cum["fixed"], "cum_oracle": cum["oracle"],
                     "regret_v2_vs_fixed": cum_reg["v2"],
                     "regret_std_vs_fixed": cum_reg["std"],
                     "std_over_v2": cum["std"] / cum["v2"],
                     "first_month_maxw_std>0.99": first_collapse,
                     "ranking": ">".join(order)})

df = pd.DataFrame(rows)
sm = pd.DataFrame(summ)
with open(os.path.join(OUT, "e9_standard_hedge_comparison.csv"), "w") as f:
    f.write("# === per-month rows (std = textbook unnormalized Hedge) ===\n")
    df.to_csv(f, index=False)
    f.write("\n# === per-block summary ===\n")
    sm.to_csv(f, index=False)

# ---- aggregate verdict numbers ----
g = sm[["cum_v2", "cum_std", "cum_fixed", "cum_oracle"]].mean()
rankings = sm["ranking"].value_counts()
reg_ratio = (sm["regret_std_vs_fixed"] / sm["regret_v2_vs_fixed"].replace(0, np.nan))
print(sm.to_string(index=False))
print("\nmean cumulative loss over 15 blocks:")
print(g.round(3).to_string())
print("\nstrategy rankings (best>...>worst):")
print(rankings.to_string())
print(f"\nstd/v2 cumulative-loss ratio: mean {sm['std_over_v2'].mean():.4f} "
      f"min {sm['std_over_v2'].min():.4f} max {sm['std_over_v2'].max():.4f}")
print(f"std one-hot collapse month (first maxw>0.99): "
      f"{sorted(set(x for x in sm['first_month_maxw_std>0.99'] ))}")
print(f"regret(std)/regret(v2) vs fixed: median {reg_ratio.median():.3f}")
# bound comparison (bound applies only to normalized [0,1] losses)
print("\nfinal-month cum regret vs sqrt(T ln N /2) bound (raw scale, bound "
      "applies only to normalized [0,1] losses):")
fin = df[df.month == N_MONTHS - 1]
print(fin[["market", "seed", "cum_regret_v2", "cum_regret_std",
           "bound_sqrt_TlnN2"]].to_string(index=False))
