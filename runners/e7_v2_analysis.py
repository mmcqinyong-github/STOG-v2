"""E7-v2 analysis: summary, rank reversal, probe zero-transfer, routing reversal.

Inputs:
  results/e7_v2/e7v2_runs.csv          per-run metrics (513 runs incl. H=24 ref)
  results/e7_v2/errs/*.npz             per-window val/test errors
  results/e7_v2/preds/*.npz            seed-2021 test predictions + block meta
  results/e6_epf_main.csv              EPF per-run metrics (19 experts)
  results/preds/*.npz                  EPF per-window preds/errors/features

Outputs:
  results/e7_v2/e7v2_summary.csv
  results/e7_v2/e7v2_rank_reversal.csv
  results/e7_v2/e7v2_probe_transfer.csv
  results/figures/e7_v2_rank_reversal_heatmap.png
  results/figures/e7_v2_horizon_scaling_curves.png
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experts.zoo import get_all_cards

EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
              "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
E = len(EXPERT_IDS)
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
DATASETS = ["ETTh1", "ETTm1", "Weather"]
HORIZONS = [24, 96, 720]
CORE_H = [96, 720]
PROBE_IDX = [7, 10, 3, 11, 8]  # spec_decay, cond_number, kurt, regime_overlap, season_strength
OUT = "./results/e7_v2"
EPF_PRED = "./results/preds"

CARDS = get_all_cards()
FAMILY = {e: (CARDS[e].family if e in CARDS else "?") for e in EXPERT_IDS}

runs = pd.read_csv(f"{OUT}/e7v2_runs.csv")
runs = runs[runs.test_mse < 9000].copy()

# ============ 1. summary ============
summ = (runs.groupby(["dataset", "horizon", "expert_id"])
        .agg(val_mse=("val_mse", "mean"), test_mse=("test_mse", "mean"),
             test_mse_std=("test_mse", "std"), test_mae=("test_mae", "mean"),
             epochs=("epochs", "mean"), time_sec=("time_sec", "sum"))
        .reset_index())
summ["family"] = summ.expert_id.map(FAMILY)
summ["rank"] = summ.groupby(["dataset", "horizon"])["test_mse"].rank(method="min")
summ = summ.sort_values(["dataset", "horizon", "rank"])
summ.to_csv(f"{OUT}/e7v2_summary.csv", index=False)
print("=== top-3 per block ===")
for (ds, h), g in summ.groupby(["dataset", "horizon"]):
    t3 = g.nsmallest(3, "test_mse")
    print(f"{ds} H={h}: " + ", ".join(
        f"{r.expert_id}({r.test_mse:.4f})" for r in t3.itertuples()))

# ============ 2. rank reversal (EPF vs long-term) ============
epf = pd.read_csv("./results/e6_epf_main.csv")
epf = epf[epf.expert_id.isin(EXPERT_IDS) & (epf.test_mse < 9000)].copy()
epf["rank"] = epf.groupby(["market", "seed"])["test_mse"].rank(method="min")
epf_rank = epf.groupby("expert_id")["rank"].mean().rename("epf_mean_rank")

lt = runs[runs.horizon.isin(CORE_H)].copy()
lt["rank"] = lt.groupby(["dataset", "horizon", "seed"])["test_mse"].rank(method="min")
lt_rank = lt.groupby("expert_id")["rank"].mean().rename("lt_mean_rank")

rev = pd.DataFrame({"expert_id": EXPERT_IDS}).set_index("expert_id")
rev["family"] = pd.Series(FAMILY)
rev = rev.join(epf_rank).join(lt_rank)
# per-block mean ranks
for (ds, h), g in lt.groupby(["dataset", "horizon"]):
    rev[f"rank_{ds}_H{h}"] = g.groupby("expert_id")["rank"].mean()
rev["rank_diff"] = rev.lt_mean_rank - rev.epf_mean_rank  # >0: worse on LT
rev = rev.reset_index().sort_values("lt_mean_rank")
rev.to_csv(f"{OUT}/e7v2_rank_reversal.csv", index=False)

# KL divergence between softmax(-rank) distributions
def rank_dist(r):
    z = -r.values.astype(float)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
p_epf, p_lt = rank_dist(rev.set_index("expert_id").loc[EXPERT_IDS, "epf_mean_rank"]), \
              rank_dist(rev.set_index("expert_id").loc[EXPERT_IDS, "lt_mean_rank"])
kl = float((p_lt * np.log(p_lt / p_epf)).sum())
print(f"\nKL(p_LT || p_EPF) over softmax(-mean-rank): {kl:.4f}")
print("\n=== rank reversal table ===")
print(rev[["expert_id", "family", "epf_mean_rank", "lt_mean_rank", "rank_diff"]]
      .round(2).to_string(index=False))

# ============ 3. probe zero-transfer ============
print("\n=== training EPF probe scorer (pooled 5 markets x 3 seeds) ===")
t0 = time.time()
Xs, Ys = [], []
for m in MARKETS:
    for s in SEEDS:
        meta = np.load(f"{EPF_PRED}/meta_{m}_{s}.npz")
        val_true = meta["val_true"]
        feats = np.concatenate([meta["feat_train"], meta["feat_val"]], axis=0)
        terrs, verrs = [], []
        for eid in EXPERT_IDS:
            d = np.load(f"{EPF_PRED}/{m}_{eid}_{s}.npz")
            terrs.append(d["train_err"])
            verrs.append(((d["val_pred"] - val_true) ** 2).mean(axis=1))
        Y = np.log(np.concatenate([np.stack(terrs, 1), np.stack(verrs, 1)], 0) + 1e-8)
        Xs.append(feats[:, PROBE_IDX])
        Ys.append(Y)
X = np.concatenate(Xs); Y = np.concatenate(Ys)
Y = Y - Y.mean(axis=1, keepdims=True)
mu, sd = X.mean(0), X.std(0) + 1e-8
X = (X - mu) / sd
rng = np.random.RandomState(0)
sub = rng.choice(len(X), size=min(40000, len(X)), replace=False)
from sklearn.ensemble import HistGradientBoostingRegressor
sc_models = []
for e in range(E):
    mdl = HistGradientBoostingRegressor(max_iter=120, max_depth=6,
                                        learning_rate=0.08, random_state=0)
    mdl.fit(X[sub], Y[sub, e])
    sc_models.append(mdl)
print(f"scorer trained on {len(X)} EPF windows ({time.time()-t0:.0f}s)")

def lt_err_block(ds, h, seed, split):
    errs = []
    for eid in EXPERT_IDS:
        d = np.load(f"{OUT}/errs/{ds}_H{h}_{seed}_{eid}.npz")
        errs.append(d[f"{split}_err"])
    return np.stack(errs, 0)  # (E, n)

def predict_errhat(feats12):
    Xq = (feats12[:, PROBE_IDX] - mu) / sd
    return np.stack([m.predict(Xq) for m in sc_models], axis=1)  # (n, E)

spear_rows, route_rows = [], []
rng_s = np.random.RandomState(7)
for ds in DATASETS:
    for h in CORE_H:
        meta = np.load(f"{OUT}/preds/meta_{ds}_H{h}.npz")
        for seed in SEEDS:
            val_err = lt_err_block(ds, h, seed, "val")      # (E, n_val)
            err_hat_val = predict_errhat(meta["feat_val"])
            nv = val_err.shape[1]
            subw = rng_s.choice(nv, size=min(2000, nv), replace=False)
            rs = [spearmanr(err_hat_val[i], val_err[:, i]).statistic for i in subw]
            spear_rows.append({"dataset": ds, "horizon": h, "seed": seed,
                               "spearman_probe_zero_transfer": float(np.nanmean(rs)),
                               "frac_windows_positive": float(np.mean(np.array(rs) > 0)),
                               "n_windows": len(subw)})

        # --- routing reversal (seed 2021, test windows, TopK=3 fusion) ---
        meta_t = np.load(f"{OUT}/preds/meta_{ds}_H{h}.npz")
        test_true = meta_t["test_true"]
        test_pred = np.stack([
            np.load(f"{OUT}/preds/{ds}_H{h}_{eid}_2021.npz")["test_pred"]
            for eid in EXPERT_IDS], 0)                        # (E, n, H)
        err_hat_t = predict_errhat(meta_t["feat_test"])       # (n, E)
        nt = err_hat_t.shape[0]
        topk = np.argsort(err_hat_t, axis=1)[:, :3]
        s = -np.take_along_axis(err_hat_t, topk, 1)
        s = s - s.max(1, keepdims=True)
        w = np.exp(s); w /= w.sum(1, keepdims=True)
        preds_k = np.transpose(test_pred, (1, 0, 2))[np.arange(nt)[:, None], topk]
        fused = (preds_k * w[:, :, None]).sum(1)
        def mse_of(p): return float(((p - test_true) ** 2).mean())
        i47, i63 = EXPERT_IDS.index("M47"), EXPERT_IDS.index("M63")
        val_err21 = lt_err_block(ds, h, 2021, "val")
        best_val = int(np.argmin(val_err21.mean(1)))
        row = {"dataset": ds, "horizon": h,
               "router_top3_probe_zero_transfer": mse_of(fused),
               "epf_champion_M47": mse_of(test_pred[i47]),
               "epf_champion_M63": mse_of(test_pred[i63]),
               "epf_champ_mean_M47M63": mse_of(0.5 * (test_pred[i47] + test_pred[i63])),
               "best_single_by_val": mse_of(test_pred[best_val]),
               "best_single_id": EXPERT_IDS[best_val],
               "oracle_top3": float(np.sort(((test_pred - test_true[None]) ** 2)
                                            .mean(2), axis=0)[:3].mean())}
        route_rows.append(row)
        print(f"[route] {ds}/H{h}: probeTop3={row['router_top3_probe_zero_transfer']:.4f} "
              f"M47={row['epf_champion_M47']:.4f} M63={row['epf_champion_M63']:.4f} "
              f"bestVal({row['best_single_id']})={row['best_single_by_val']:.4f}")

spear_df = pd.DataFrame(spear_rows)
route_df = pd.DataFrame(route_rows)
probe_out = pd.concat([spear_df.assign(section="probe_spearman"),
                       route_df.assign(section="routing")], ignore_index=True)
probe_out.to_csv(f"{OUT}/e7v2_probe_transfer.csv", index=False)
print("\n=== probe zero-transfer Spearman (val windows) ===")
print(spear_df.groupby(["dataset", "horizon"])
      [["spearman_probe_zero_transfer", "frac_windows_positive"]].mean().round(3))

# ============ 4. figures ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 4a. rank reversal heatmap: experts x tasks
blocks = [("EPF", None, None)] + [(f"{ds}\nH{h}", ds, h) for ds in DATASETS for h in CORE_H]
order = rev.sort_values(["family", "lt_mean_rank"]).expert_id.tolist()
M = np.zeros((len(order), len(blocks)))
rev_i = rev.set_index("expert_id")
for j, (name, ds, h) in enumerate(blocks):
    col = "epf_mean_rank" if ds is None else f"rank_{ds}_H{h}"
    M[:, j] = rev_i.loc[order, col].values
fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=19)
ax.set_xticks(range(len(blocks)), [b[0] for b in blocks], fontsize=9)
ax.set_yticks(range(len(order)),
              [f"{e} ({FAMILY[e]})" for e in order], fontsize=9)
for i in range(len(order)):
    for j in range(len(blocks)):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
ax.set_title("E7-v2 rank reversal: expert mean rank by task "
             "(green=best; EPF = 5-market avg; LT = 3-seed avg)")
fig.colorbar(im, label="mean rank (1=best)")
fig.tight_layout()
fig.savefig("./results/figures/e7_v2_rank_reversal_heatmap.png", dpi=150)
plt.close(fig)

# 4b. horizon scaling curves: family-mean normalized test MSE vs H
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
fams = sorted(set(FAMILY.values()))
colors = plt.cm.tab20(np.linspace(0, 1, len(fams)))
for ax, ds in zip(axes, DATASETS):
    sub = summ[summ.dataset == ds]
    for f, c in zip(fams, colors):
        g = sub[sub.family == f].groupby("horizon")["test_mse"].mean()
        g = g.reindex(HORIZONS)
        ax.plot(HORIZONS, g.values, "-o", color=c, label=f, lw=1.5, ms=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(HORIZONS, ["24", "96", "720"])
    ax.set_xlabel("horizon"); ax.set_title(ds)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test MSE (z-scored target)")
axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.suptitle("E7-v2 horizon scaling: family-mean test MSE, H=24->96->720 (L=336)")
fig.tight_layout()
fig.savefig("./results/figures/e7_v2_horizon_scaling_curves.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("\nfigures saved")
