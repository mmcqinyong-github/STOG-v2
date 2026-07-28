"""E3-v2: heavy-tail influence-function analysis + EPF real-spike validation.

A) Densify spike amplitude grid: rate fixed 5%, amps {4.5, 8, 10} sigma
   (amps {3, 6} at rates {1%, 5%} already exist in results/e3_heavytail.csv),
   seeds {2021, 42}, 5 experts (robust: M233/M55/M220; raw: M03/M52),
   same protocol (T=5000, V=8, H=24, 5 epochs, hidden 128).
   Clean MSEs are reused from the old CSV (clean field does not depend on
   spike params). Checkpointed per (amp, seed, expert).

B) Influence function: per expert x seed, fit
       log(mse_spike / mse_clean) = a + beta_rate * log(rate) + beta_amp * log(amp)
   Theory: raw group beta_rate ~ 1.0, moment group beta_rate significantly < 1.

C) EPF real spikes (zero training): reuse results/preds predictions.
   Spike window = any point with |z| > 4 (z vs global test_true stats).
   Degradation ratio = MSE(spike windows) / MSE(clean windows) per expert per
   market x seed; robust vs raw group comparison + paired Wilcoxon over the
   15 market-seed units.

Outputs:
  results/e3_v2/e3v2_new_runs.csv            (checkpoint)
  results/e3_v2/e3v2_influence_function.csv  (beta exponents)
  results/e3_v2/e3v2_epf_spike.csv           (real-spike degradation)
  results/figures/e3_v2_influence_function_fit.png
  results/figures/e3_v2_epf_spike_degradation.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ROBUST = ["M233", "M55", "M220"]
RAW = ["M03", "M52"]
ALL = ROBUST + RAW
NEW_AMPS = [4.5, 8.0, 10.0]
RATE = 0.05
SEEDS = [2021, 42]
OUT_DIR = "./results/e3_v2"
CKPT = f"{OUT_DIR}/e3v2_new_runs.csv"

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
EPF_SEEDS = [2021, 42, 3407]
PRED_DIR = "./results/preds"


class MockDM:
    def __init__(self, d):
        self.windows = d


def group_of(eid):
    return "robust" if eid in ROBUST else "raw"


# ---------------- Phase A: densify spike grid ----------------

def phase_train():
    ensure_dir(OUT_DIR)
    done = pd.read_csv(CKPT) if os.path.exists(CKPT) else pd.DataFrame()
    done_keys = set()
    if len(done):
        done_keys = set(zip(done.spike_amp, done.seed, done.expert_id))
    for amp in NEW_AMPS:
        for seed in SEEDS:
            cfg = SynthConfig(T=5000, V=8, H=24, alpha=1.0, spatial_type="lowrank",
                              spike_rate=RATE, spike_amp=amp, seed=seed)
            data = SpatioTemporalFieldGenerator(cfg).generate()
            dm = MockDM({"train": data["train_inp"], "train_tgt": data["train_tgt"],
                         "val": data["val_inp"], "val_tgt": data["val_tgt"],
                         "test": data["test_inp"], "test_tgt": data["test_tgt"]})
            d_in = data["train_inp"].shape[1]
            for eid in ALL:
                if (amp, seed, eid) in done_keys:
                    continue
                set_seed(seed)
                expert = get_expert(eid, d_in, hidden=128, drop=0.1)
                tr = UnifiedTrainer({"max_epochs": 5, "patience": 2,
                                     "batch_size": 256, "lr": 1e-4})
                try:
                    mse_spike = tr.train_expert(expert, dm)["test_mse"]
                except Exception as ex:
                    print(f"  [warn] {eid} amp={amp} seed={seed}: {ex}")
                    mse_spike = np.nan
                rec = {"expert_id": eid, "group": group_of(eid),
                       "spike_rate": RATE, "spike_amp": amp, "seed": seed,
                       "mse_spike": mse_spike}
                done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
                done.to_csv(CKPT, index=False)
                print(f"[e3v2] amp={amp} seed={seed} {eid}: mse={mse_spike:.4f}", flush=True)
    return pd.read_csv(CKPT)


# ---------------- Phase B: influence function ----------------

def phase_influence():
    old = pd.read_csv("./results/e3_heavytail.csv")
    old = old[old["seed"].isin(SEEDS) & old["expert_id"].isin(ALL)]
    old = old[["expert_id", "group", "spike_rate", "spike_amp", "seed",
               "mse_spike", "mse_clean", "degradation"]]
    new = pd.read_csv(CKPT)
    # clean MSEs from old CSV (same seed/protocol, spike-independent)
    clean_map = (old.groupby(["expert_id", "seed"])["mse_clean"].first().to_dict())
    new["mse_clean"] = [clean_map[(e, s)] for e, s in zip(new.expert_id, new.seed)]
    new["degradation"] = (new.mse_spike - new.mse_clean) / (new.mse_clean + 1e-10)
    df = pd.concat([old, new[old.columns]], ignore_index=True)
    df["log_ratio"] = np.log(df.mse_spike / df.mse_clean)

    rows = []
    for (eid, seed), g in df.groupby(["expert_id", "seed"]):
        Xd = np.column_stack([np.ones(len(g)),
                              np.log(g.spike_rate.values),
                              np.log(g.spike_amp.values)])
        y = g.log_ratio.values
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        r2 = 1 - resid.var() / (y.var() + 1e-12)
        rows.append({"expert_id": eid, "group": group_of(eid), "seed": seed,
                     "beta_rate": beta[1], "beta_amp": beta[2], "fit_r2": r2,
                     "n_points": len(g)})
    betas = pd.DataFrame(rows)
    # fixed-amp two-point beta (theory's degradation-1 ~ rate^beta, i.e. the CSV
    # degradation column); only amps {3,6} have both rates. Negative
    # degradation (M220, near-zero regime) excluded from the log fit.
    rows2 = []
    for (eid, seed, amp), g in df[df.spike_amp.isin([3.0, 6.0])].groupby(
            ["expert_id", "seed", "spike_amp"]):
        g = g.sort_values("spike_rate")
        if len(g) != 2:
            continue
        d = g.degradation.values
        if (d <= 0).any():
            rows2.append({"expert_id": eid, "group": group_of(eid), "seed": seed,
                          "spike_amp": amp, "beta_rate_fixed_amp": np.nan,
                          "note": "negative degradation excluded"})
            continue
        rows2.append({"expert_id": eid, "group": group_of(eid), "seed": seed,
                      "spike_amp": amp,
                      "beta_rate_fixed_amp": float(np.log(d[1] / d[0]) / np.log(5)),
                      "note": ""})
    betas2 = pd.DataFrame(rows2)
    # group-level stats
    stats = []
    for grp in ["raw", "robust"]:
        b = betas[betas.group == grp]["beta_rate"].values
        b2 = betas2[betas2.group == grp]["beta_rate_fixed_amp"].dropna().values
        stats.append({"group": grp, "beta_rate_mean": b.mean(),
                      "beta_rate_std": b.std(), "n": len(b),
                      "beta_rate_fixed_amp_mean": b2.mean() if len(b2) else np.nan,
                      "beta_rate_fixed_amp_std": b2.std() if len(b2) else np.nan,
                      "n_fixed_amp": len(b2)})
        print(f"[e3v2] {grp}: beta_rate(joint) = {b.mean():.3f} +- {b.std():.3f} (n={len(b)}) | "
              f"beta_rate(fixed-amp) = {b2.mean() if len(b2) else np.nan:.3f} "
              f"+- {b2.std() if len(b2) else np.nan:.3f} (n={len(b2)})")
    try:
        w1 = wilcoxon(betas[betas.group == "raw"].beta_rate - 1.0).pvalue
    except Exception:
        w1 = np.nan
    try:
        w2 = wilcoxon(betas[betas.group == "robust"].beta_rate - 1.0,
                      alternative="less").pvalue
    except Exception:
        w2 = np.nan
    try:
        u, pu = mannwhitneyu(betas[betas.group == "raw"].beta_rate,
                             betas[betas.group == "robust"].beta_rate,
                             alternative="greater")
    except Exception:
        pu = np.nan
    # fixed-amp group tests
    b2_raw = betas2[betas2.group == "raw"].beta_rate_fixed_amp.dropna().values
    b2_rob = betas2[betas2.group == "robust"].beta_rate_fixed_amp.dropna().values
    try:
        w1b = wilcoxon(b2_raw - 1.0).pvalue
    except Exception:
        w1b = np.nan
    try:
        w2b = wilcoxon(b2_rob - 1.0, alternative="less").pvalue
    except Exception:
        w2b = np.nan
    try:
        _, pub = mannwhitneyu(b2_raw, b2_rob, alternative="greater")
    except Exception:
        pub = np.nan
    print(f"[e3v2] fixed-amp: raw beta==1? p={w1b:.4g} | robust beta<1? p={w2b:.4g} "
          f"| raw>robust? p={pub:.4g}")
    summ = pd.DataFrame(stats)
    summ["wilcoxon_raw_beta_eq_1_p"] = w1
    summ["wilcoxon_robust_beta_lt_1_p"] = w2
    summ["mannwhitney_raw_gt_robust_p"] = pu
    summ["fixedamp_wilcoxon_raw_beta_eq_1_p"] = w1b
    summ["fixedamp_wilcoxon_robust_beta_lt_1_p"] = w2b
    summ["fixedamp_mannwhitney_raw_gt_robust_p"] = pub
    out = pd.concat([betas.assign(row_type="per_expert_seed"),
                     betas2.assign(row_type="fixed_amp_two_point"),
                     summ.assign(row_type="group_summary")], ignore_index=True)
    out.to_csv(f"{OUT_DIR}/e3v2_influence_function.csv", index=False)
    print(f"[e3v2] raw beta==1? p={w1:.4g} | robust beta<1? p={w2:.4g} | raw>robust? p={pu:.4g}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, grp in zip(axes, ["raw", "robust"]):
        sub = df[df.group == grp]
        for eid in sorted(sub.expert_id.unique()):
            g = sub[sub.expert_id == eid]
            ax.scatter(g.spike_rate, np.exp(g.log_ratio), label=eid, s=40, alpha=0.8)
        b = betas[betas.group == grp].beta_rate.mean()
        xs = np.array([0.01, 0.05])
        # reference slope beta through geometric mean
        ym = np.exp(sub[sub.spike_rate == 0.01].log_ratio.mean())
        ax.plot(xs, ym * (xs / 0.01) ** b, "k--",
                label=f"fit slope beta={b:.2f}")
        ax.plot(xs, ym * (xs / 0.01) ** 1.0, "r:", label="beta=1 reference")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("spike rate"); ax.set_title(f"{grp} group")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MSE_spike / MSE_clean")
    fig.suptitle("E3-v2 influence function: degradation vs spike rate (log-log)")
    fig.tight_layout()
    ensure_dir("./results/figures")
    fig.savefig("./results/figures/e3_v2_influence_function_fit.png",
                dpi=150, bbox_inches="tight")
    print("Saved results/figures/e3_v2_influence_function_fit.png")
    return df


# ---------------- Phase C: EPF real spikes ----------------

def phase_epf():
    rows = []
    for market in MARKETS:
        for seed in EPF_SEEDS:
            meta = np.load(f"{PRED_DIR}/meta_{market}_{seed}.npz")
            test_true = meta["test_true"]  # (nt, H)
            mu, sd = test_true.mean(), test_true.std() + 1e-10
            z = np.abs((test_true - mu) / sd)
            spike_mask = (z > 4.0).any(axis=1)  # (nt,)
            n_spike = int(spike_mask.sum())
            if n_spike < 5 or spike_mask.sum() > len(spike_mask) - 5:
                print(f"[e3v2-epf] {market}/{seed}: only {n_spike} spike windows, skipped")
                continue
            for eid in ALL:
                d = np.load(f"{PRED_DIR}/{market}_{eid}_{seed}.npz")
                pred = d["test_pred"]
                err = ((pred - test_true) ** 2).mean(axis=1)  # (nt,)
                ratio = err[spike_mask].mean() / (err[~spike_mask].mean() + 1e-12)
                rows.append({"market": market, "seed": seed, "expert_id": eid,
                             "group": group_of(eid),
                             "n_spike_windows": n_spike, "n_clean_windows": int((~spike_mask).sum()),
                             "mse_spike": float(err[spike_mask].mean()),
                             "mse_clean": float(err[~spike_mask].mean()),
                             "degradation_ratio": float(ratio)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e3v2_epf_spike.csv", index=False)

    # group comparison: per market-seed unit, mean ratio per group
    units = df.groupby(["market", "seed", "group"])["degradation_ratio"].mean().unstack()
    print("\n=== E3-v2 EPF real-spike degradation ratio (spike/clean MSE) ===")
    print(units.round(3).to_string())
    try:
        w = wilcoxon(units["raw"] - units["robust"], alternative="greater")
        print(f"paired Wilcoxon raw > robust over {len(units)} units: "
              f"W={w.statistic:.1f}, p={w.pvalue:.4g}")
        wil_p = w.pvalue
    except Exception as ex:
        print(f"wilcoxon failed: {ex}")
        wil_p = np.nan
    print(f"group means: raw={units['raw'].mean():.3f}, robust={units['robust'].mean():.3f}")
    summ = pd.DataFrame([{"raw_mean": units["raw"].mean(),
                          "robust_mean": units["robust"].mean(),
                          "wilcoxon_raw_gt_robust_p": wil_p,
                          "n_units": len(units)}])
    summ.to_csv(f"{OUT_DIR}/e3v2_epf_spike_summary.csv", index=False)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    order = RAW + ROBUST
    data_plot = [df[df.expert_id == e]["degradation_ratio"].values for e in order]
    colors = ["#d62728"] * len(RAW) + ["#2ca02c"] * len(ROBUST)
    bp = ax.boxplot(data_plot, tick_labels=order, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_color(c)
    ax.set_ylabel("MSE(spike windows) / MSE(clean windows)")
    ax.set_title("EPF real spikes: per-expert degradation (15 market-seed units)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    x = np.arange(len(units))
    wdt = 0.35
    ax.bar(x - wdt / 2, units["raw"], wdt, label="raw (M03,M52)", color="#d62728", alpha=0.8)
    ax.bar(x + wdt / 2, units["robust"], wdt, label="robust (M233,M55,M220)", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}/{s}" for m, s in units.index], rotation=45, fontsize=8)
    ax.set_ylabel("group mean degradation ratio")
    ax.set_title(f"paired by market-seed (Wilcoxon p={wil_p:.3g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./results/figures/e3_v2_epf_spike_degradation.png",
                dpi=150, bbox_inches="tight")
    print("Saved results/figures/e3_v2_epf_spike_degradation.png")


def main():
    phase_train()
    phase_influence()
    phase_epf()


if __name__ == "__main__":
    main()
