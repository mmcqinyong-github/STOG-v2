"""E10-v2: Operator Transplant Causal ATE — REAL operator grafting (pre-registration C5).

Replaces the invalid hidden_dim-perturbation proxy of run_e10_full.py.
For each (base, operator) pair we train:
  TREAT = base + grafted operator branch (operator-specific information pathway)
  CTRL  = base + capacity-matched placebo branch (same module shape/params,
          operator pathway neutralized) -> controls the capacity confound.
ATE = MSE_TREAT - MSE_CTRL (negative = operator beneficial).

Design: 3 bases (M52 DLinear / M17 ModernTCN / M50 PatchTST) x 4 operators
(diff/moment/graph/gate) x 2 arms x 5 markets x 3 seeds = 360 runs.

Checkpoint/resume: every finished run is appended to results/e10_v2/e10v2_runs.csv;
re-running the script skips rows already present. A --time-budget lets the driver
exit cleanly before the Bash 300s cap.

Usage:
  python run_e10_v2.py                # run pending experiments (with time budget)
  python run_e10_v2.py --time-budget 280
  python run_e10_v2.py --analyze      # build CSV summaries + figures from runs CSV
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule, MARKET_META
from src.experts.zoo import get_expert, graft_operator, get_all_cards
from src.training.trainer import UnifiedTrainer

RESULTS_DIR = "./results/e10_v2"
RUNS_CSV = os.path.join(RESULTS_DIR, "e10v2_runs.csv")
FIG_DIR = "./results/figures"

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
BASE_MODELS = ["M52", "M17", "M50"]
OPERATORS = ["diff", "moment", "graph", "gate"]
ARMS = ["treat", "ctrl"]

TRAIN_CFG = {"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4}


# ----------------------------- Phase 1: runs -----------------------------

def pending_grid(done_keys):
    for market in MARKETS:
        for seed in SEEDS:
            for base in BASE_MODELS:
                for op in OPERATORS:
                    for arm in ARMS:
                        if (market, seed, base, op, arm) not in done_keys:
                            yield market, seed, base, op, arm


def load_done_keys():
    if not os.path.exists(RUNS_CSV):
        return set()
    df = pd.read_csv(RUNS_CSV)
    return set(zip(df.market, df.seed, df.base_model, df.operator, df.arm))


def run_phase(time_budget=280.0):
    ensure_dir(RESULTS_DIR)
    t_start = time.time()
    done = load_done_keys()
    todo = list(pending_grid(done))
    print(f"[e10v2] done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("[e10v2] all runs complete.")
        return

    # Group pending work by (market, seed) so each datamodule is loaded once.
    from itertools import groupby
    new_rows = []
    for (market, seed), grp in groupby(todo, key=lambda t: (t[0], t[1])):
        if time.time() - t_start > time_budget:
            break
        grp = list(grp)
        set_seed(seed)
        dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
        dm.make_windows()
        dm.normalize()
        d_in = dm.windows["train"].shape[1]
        for (_, _, base_id, op, arm) in grp:
            if time.time() - t_start > time_budget:
                break
            # Fresh seed per config for reproducible init regardless of resume order
            cfg_seed = abs(hash((market, seed, base_id, op, arm))) % (2**31)
            set_seed(cfg_seed)
            try:
                base = get_expert(base_id, d_in, hidden=256, drop=0.1)
                model = graft_operator(base, op, arm, n_vars=3, lookback=168, horizon=24)
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                trainer = UnifiedTrainer(TRAIN_CFG)
                res = trainer.train_expert(model, dm)
                row = {
                    "market": market, "seed": seed, "base_model": base_id,
                    "operator": op, "arm": arm,
                    "val_mse": res["val_mse"], "test_mse": res["test_mse"],
                    "test_mae": res["test_mae"], "epochs": res["epochs"],
                    "time_sec": res["time_sec"], "n_trainable": n_trainable,
                }
            except Exception as ex:
                row = {
                    "market": market, "seed": seed, "base_model": base_id,
                    "operator": op, "arm": arm,
                    "val_mse": np.nan, "test_mse": np.nan, "test_mae": np.nan,
                    "epochs": 0, "time_sec": 0.0, "n_trainable": -1,
                    "error": str(ex)[:200],
                }
                print(f"  ERROR {market}/{seed}/{base_id}/{op}/{arm}: {ex}", flush=True)
            new_rows.append(row)
            print(f"  {market} s{seed} {base_id} {op:6s} {arm:5s} "
                  f"test_mse={row['test_mse']:.3f} ({row['time_sec']:.1f}s)", flush=True)
            # checkpoint after EVERY run
            df_new = pd.DataFrame([row])
            df_new.to_csv(RUNS_CSV, mode="a", header=not os.path.exists(RUNS_CSV), index=False)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done2 = load_done_keys()
    print(f"[e10v2] session added {len(done2) - len(done)} runs; total done={len(done2)}/360", flush=True)


# ----------------------------- Probe features -----------------------------

def market_probe_features():
    """Seed-independent features from the chronological first-70% train segment."""
    from scipy import fft
    from scipy.stats import kurtosis
    feats = {}
    for market in MARKETS:
        df = pd.read_csv(f"./dataset/epf/{market}.csv")
        df.columns = [c.strip() for c in df.columns]
        price_col = "OT" if "OT" in df.columns else df.columns[-1]
        z_cols = [c for c in df.columns if c not in ["date", price_col]][:2]
        x = df[price_col].values.astype(np.float64)
        n = int(len(x) * 0.7)
        xt = x[:n]
        # kappa-hat: condition number of lag-24 Hankel covariance (log10)
        Lh = 24
        H = np.stack([xt[i:i + Lh] for i in range(0, len(xt) - Lh, 6)])
        cov = np.cov(H.T) + 1e-6 * np.eye(Lh)
        kappa = float(np.log10(np.linalg.cond(cov)))
        # gamma-hat: excess kurtosis (spike heaviness)
        gamma = float(kurtosis(xt))
        # alpha-hat: spectral decay slope on log-log periodogram
        f = np.abs(fft.rfft(xt - xt.mean()))[1:]
        freqs = np.arange(1, len(f) + 1)
        alpha = float(np.polyfit(np.log(freqs), np.log(f + 1e-10), 1)[0])
        # s-hat: seasonal strength = energy share at 24h +- 1 harmonic band
        T = len(xt)
        per_bin = T // 24
        band = np.arange(max(1, per_bin - 1), min(len(f), per_bin + 2))
        s_hat = float(f[band - 1].sum() / (f.sum() + 1e-10))
        # long memory proxy: lag-24 autocorrelation
        acf24 = float(np.corrcoef(xt[:-24], xt[24:])[0, 1])
        # inter-variable correlation strength
        corrs = []
        for zc in z_cols:
            z = df[zc].values.astype(np.float64)[:n]
            corrs.append(abs(np.corrcoef(xt, z)[0, 1]))
        z1, z2 = df[z_cols[0]].values.astype(np.float64)[:n], df[z_cols[1]].values.astype(np.float64)[:n]
        corrs.append(abs(np.corrcoef(z1, z2)[0, 1]))
        corr_strength = float(np.nanmean(corrs))
        feats[market] = {
            "kappa_hat": kappa, "gamma_hat": gamma, "alpha_hat": alpha,
            "s_hat": s_hat, "acf24": acf24, "corr_strength": corr_strength,
        }
    return pd.DataFrame(feats).T.reset_index().rename(columns={"index": "market"})


# ----------------------------- Attribution proxy -----------------------------

def operator_affinity_profiles():
    """Aggregate genome-card affinities of all experts that natively contain each operator."""
    cards = get_all_cards()
    buckets = {"diff": [], "moment": [], "graph": [], "gate": []}
    for c in cards.values():
        tb, sb, rb, gb = c.temporal_basis, c.spatial_basis, c.robust_basis, c.gate_basis
        if "difference" in tb:
            buckets["diff"].append(c)
        if "moment" in tb or any(r in ("quantile", "median", "huber") for r in rb):
            buckets["moment"].append(c)
        if any(s in ("static_low_rank", "dynamic_coupling", "graph_coupling",
                     "cross_section_attention") for s in sb) and sb != ["identity"]:
            buckets["graph"].append(c)
        if any(g in ("volatility_gate", "input_dependent_gate", "spectral_gate") for g in gb):
            buckets["gate"].append(c)
    keys = ["low_freq_decay", "spike_heavy_tail", "long_memory", "strong_periodicity"]
    prof = {}
    for op, cs in buckets.items():
        agg = {k: float(np.mean([c.spectral_affinity.get(k, 0.5) for c in cs])) for k in keys}
        agg["spatial_corr"] = float(np.mean([
            max(c.spatial_affinity.get("graph_coupling", 0.0),
                c.spatial_affinity.get("cross_section_attention", 0.0),
                c.spatial_affinity.get("static_low_rank", 0.0)) for c in cs]))
        agg["n_cards"] = len(cs)
        prof[op] = agg
    return prof


def minmax(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 1e-12 else s * 0.0


# ----------------------------- Phase 2: analysis -----------------------------

def analyze():
    from scipy import stats as sstats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(RESULTS_DIR)
    ensure_dir(FIG_DIR)
    df = pd.read_csv(RUNS_CSV)
    df = df.dropna(subset=["test_mse"])
    print(f"[analyze] valid runs: {len(df)}")

    # ---- ATE per (market, seed, base, op) ----
    pv = df.pivot_table(index=["market", "seed", "base_model", "operator"],
                        columns="arm", values="test_mse").reset_index()
    pv["ate"] = pv["treat"] - pv["ctrl"]
    feats = market_probe_features()
    pv = pv.merge(feats, on="market", how="left")
    pv.to_csv(os.path.join(RESULTS_DIR, "e10v2_ate.csv"), index=False)

    # ---- stratification splits (median across markets) ----
    kap_med = feats["kappa_hat"].median()
    gam_med = feats["gamma_hat"].median()
    pv["kappa_grp"] = np.where(pv["kappa_hat"] >= kap_med, "high_kappa", "low_kappa")
    pv["gamma_grp"] = np.where(pv["gamma_hat"] >= gam_med, "high_spike", "low_spike")
    print(feats.to_string(index=False))
    print("kappa median:", kap_med, "gamma median:", gam_med)

    # ---- global ATE by operator ----
    glob = pv.groupby("operator").agg(
        ate_mean=("ate", "mean"), ate_std=("ate", "std"),
        n=("ate", "count"),
        sign_neg=("ate", lambda s: float((s < 0).mean())),
        treat_mse=("treat", "mean"), ctrl_mse=("ctrl", "mean"),
    ).reset_index()
    glob["sign_consistency"] = glob["sign_neg"].apply(lambda p: max(p, 1 - p))

    # ---- ATE by operator x base (seed/market stats) ----
    opbase = pv.groupby(["operator", "base_model"]).agg(
        ate_mean=("ate", "mean"), ate_std=("ate", "std"), n=("ate", "count"),
        sign_neg=("ate", lambda s: float((s < 0).mean())),
    ).reset_index()

    # ---- stratified ATE + Wilcoxon ----
    strat_rows = []

    def paired_wilcoxon(sub, group_col, hi, lo):
        """Pair by (base_model, seed): mean ATE in hi-group vs lo-group markets."""
        a = sub[sub[group_col] == hi].groupby(["base_model", "seed"])["ate"].mean()
        b = sub[sub[group_col] == lo].groupby(["base_model", "seed"])["ate"].mean()
        j = pd.concat([a, b], axis=1, keys=["hi", "lo"]).dropna()
        if len(j) >= 5:
            stat, p = sstats.wilcoxon(j["hi"], j["lo"])
            stat1, p1 = sstats.wilcoxon(j["hi"], j["lo"], alternative="less")
        else:
            p, p1 = np.nan, np.nan
        return j, p, p1

    # Acceptance 1: diff ~ 0 in low-kappa, < 0 in high-kappa
    d = pv[pv.operator == "diff"]
    jk, pk, pk1 = paired_wilcoxon(d, "kappa_grp", "high_kappa", "low_kappa")
    lo_cells = d[d.kappa_grp == "low_kappa"]["ate"]
    hi_cells = d[d.kappa_grp == "high_kappa"]["ate"]
    _, p_lo0 = sstats.wilcoxon(lo_cells) if len(lo_cells) >= 5 else (np.nan, np.nan)
    _, p_hi0 = sstats.wilcoxon(hi_cells, alternative="less") if len(hi_cells) >= 5 else (np.nan, np.nan)
    acc1 = {
        "ate_low_kappa": lo_cells.mean(), "ate_high_kappa": hi_cells.mean(),
        "wilcoxon_lowk_vs_0_p": p_lo0, "wilcoxon_highk_neg_p": p_hi0,
        "wilcoxon_high_vs_low_p": pk,
        "pass": bool((p_lo0 > 0.05) and (p_hi0 < 0.05) and (pk < 0.05)
                     and hi_cells.mean() < lo_cells.mean()),
    }

    # Acceptance 2: moment more negative in high-spike
    m = pv[pv.operator == "moment"]
    jg, pg, pg1 = paired_wilcoxon(m, "gamma_grp", "high_spike", "low_spike")
    acc2 = {
        "ate_high_spike": m[m.gamma_grp == "high_spike"]["ate"].mean(),
        "ate_low_spike": m[m.gamma_grp == "low_spike"]["ate"].mean(),
        "wilcoxon_high_vs_low_p_twosided": pg,
        "wilcoxon_high_more_neg_p": pg1,
        "pass": bool(pg1 < 0.05),
    }

    # Acceptance 3: graph ATE interacts with inter-variable correlation strength
    g = pv[pv.operator == "graph"]
    gm = g.groupby("market").agg(ate=("ate", "mean"), corr=("corr_strength", "first"))
    rho3, p3 = sstats.spearmanr(gm["corr"], gm["ate"])
    # also cell-level
    rho3c, p3c = sstats.spearmanr(g["corr_strength"], g["ate"])
    acc3 = {"spearman_marketlevel_rho": rho3, "p": p3,
            "spearman_celllevel_rho": rho3c, "p_cell": p3c,
            "pass": bool(rho3 < 0 and p3c < 0.10)}

    # stratified CSV rows (all ops x both grouping schemes)
    for op in OPERATORS:
        for gc, hi, lo in [("kappa_grp", "high_kappa", "low_kappa"),
                           ("gamma_grp", "high_spike", "low_spike")]:
            sub = pv[pv.operator == op]
            jj, pp, _ = paired_wilcoxon(sub, gc, hi, lo)
            for grpval in [hi, lo]:
                cells = sub[sub[gc] == grpval]["ate"]
                strat_rows.append({
                    "operator": op, "stratify_by": gc, "group": grpval,
                    "ate_mean": cells.mean(), "ate_std": cells.std(),
                    "n_cells": len(cells),
                    "ci95_lo": cells.mean() - 1.96 * cells.std() / np.sqrt(len(cells)),
                    "ci95_hi": cells.mean() + 1.96 * cells.std() / np.sqrt(len(cells)),
                    "wilcoxon_paired_hi_vs_lo_p": pp if grpval == hi else np.nan,
                })
    strat = pd.DataFrame(strat_rows)
    strat.to_csv(os.path.join(RESULTS_DIR, "e10v2_stratified.csv"), index=False)

    # ---- Attribution consistency ----
    prof = operator_affinity_profiles()
    f = feats.set_index("market")
    probe_norm = pd.DataFrame({
        "low_freq_decay": minmax(-f["alpha_hat"]),       # steeper decay -> higher
        "spike_heavy_tail": minmax(f["gamma_hat"]),
        "long_memory": minmax(f["acf24"]),
        "strong_periodicity": minmax(f["s_hat"]),
        "spatial_corr": minmax(f["corr_strength"]),
    }, index=f.index)
    keys = ["low_freq_decay", "spike_heavy_tail", "long_memory", "strong_periodicity", "spatial_corr"]
    attrib = []
    for op in OPERATORS:
        for market in MARKETS:
            score = sum(prof[op][k] * probe_norm.loc[market, k] for k in keys)
            attrib.append({"operator": op, "market": market, "attrib_score": score})
    attrib = pd.DataFrame(attrib)
    ate_op_mkt = pv.groupby(["operator", "market"])["ate"].mean().reset_index()
    attrib = attrib.merge(ate_op_mkt, on=["operator", "market"])
    rho_att, p_att = sstats.spearmanr(attrib["attrib_score"], -attrib["ate"])  # higher score -> lower ATE
    attrib.to_csv(os.path.join(RESULTS_DIR, "e10v2_attribution.csv"), index=False)

    # ---- summary CSV ----
    summary = {
        "n_runs_valid": len(df),
        "global_ate": {op: {"mean": float(r.ate_mean), "std": float(r.ate_std),
                            "sign_neg_frac": float(r.sign_neg)}
                       for op, r in glob.set_index("operator").iterrows()},
        "acceptance_1_diff_kappa": acc1,
        "acceptance_2_moment_spike": acc2,
        "acceptance_3_graph_corr": acc3,
        "attribution_spearman": {"rho": float(rho_att), "p": float(p_att), "target": 0.5,
                                 "pass": bool(rho_att >= 0.5)},
        "param_check": df.groupby(["operator", "arm"])["n_trainable"].mean().unstack().to_dict(),
    }
    glob.assign(base_model="ALL").to_csv(os.path.join(RESULTS_DIR, "e10v2_global.csv"), index=False)
    opbase.to_csv(os.path.join(RESULTS_DIR, "e10v2_ate_by_opbase.csv"), index=False)
    summary_flat = []
    for op, r in glob.set_index("operator").iterrows():
        summary_flat.append({"section": "global", "key": op, "value": r.ate_mean,
                             "detail": f"std={r.ate_std:.3f}, sign_neg={r.sign_neg:.2f}"})
    for name, acc in [("acc1_diff_kappa", acc1), ("acc2_moment_spike", acc2), ("acc3_graph_corr", acc3)]:
        for k, v in acc.items():
            summary_flat.append({"section": name, "key": k, "value": v, "detail": ""})
    summary_flat.append({"section": "attribution", "key": "spearman_rho", "value": rho_att,
                         "detail": f"p={p_att:.4g}, target>=0.5"})
    pd.DataFrame(summary_flat).to_csv(os.path.join(RESULTS_DIR, "e10v2_summary.csv"), index=False)

    # ---------------- figures ----------------
    # Fig 1: heatmap ATE by operator x base
    hm = opbase.pivot(index="operator", columns="base_model", values="ate_mean").loc[OPERATORS, BASE_MODELS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(hm.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.nanmax(np.abs(hm.values)), vmax=np.nanmax(np.abs(hm.values)))
    ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
    for i in range(hm.shape[0]):
        for jj in range(hm.shape[1]):
            ax.text(jj, i, f"{hm.values[i, jj]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("E10-v2 operator transplant ATE (MSE_treat - MSE_ctrl), mean over markets x seeds")
    fig.colorbar(im, ax=ax, label="ATE (negative = operator helps)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e10_v2_ate_by_operator_base_heatmap.png"), dpi=150)
    plt.close(fig)

    # Fig 2: stratified forest plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for ax, op in zip(axes.ravel(), OPERATORS):
        sub = strat[strat.operator == op]
        labels = [f"{r['stratify_by'].replace('_grp','')}: {r['group']}" for _, r in sub.iterrows()]
        y = np.arange(len(sub))
        ax.errorbar(sub["ate_mean"], y,
                    xerr=[sub["ate_mean"] - sub["ci95_lo"], sub["ci95_hi"] - sub["ate_mean"]],
                    fmt="o", capsize=4, color="darkred")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"operator = {op}")
        ax.set_xlabel("ATE (95% CI)")
        ax.invert_yaxis()
    fig.suptitle("E10-v2 stratified ATE forest plot (kappa / spike splits)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e10_v2_stratified_ate_forest.png"), dpi=150)
    plt.close(fig)

    # Fig 3: attribution vs realized ATE scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    for op, mk in zip(OPERATORS, ["o", "s", "^", "D"]):
        s = attrib[attrib.operator == op]
        ax.scatter(s["attrib_score"], s["ate"], marker=mk, label=op, s=55, alpha=0.85)
    for _, r in attrib.iterrows():
        ax.annotate(r["market"], (r["attrib_score"], r["ate"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("genome attribution score (operator affinity x market probe)")
    ax.set_ylabel("realized ATE (negative = helps)")
    ax.set_title(f"E10-v2 attribution vs ATE  (Spearman rho={rho_att:.3f}, p={p_att:.3g})")
    ax.legend(title="operator")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e10_v2_attribution_vs_ate_scatter.png"), dpi=150)
    plt.close(fig)

    print(pd.DataFrame(summary_flat).to_string(index=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--time-budget", type=float, default=280.0)
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run_phase(time_budget=args.time_budget)
