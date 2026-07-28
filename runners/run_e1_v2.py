"""E1-v2: complete the E1 gaps.

  A) Third seed 3407: 12 configs x 8 experts, same protocol as original E1
     (T=5000, V=8, H=24, 5 epochs, patience 2, bs 256, lr 1e-4, hidden 128).
     Merged with the original 24 configs into results/e1_v2/e1v2_combined.csv
     (original file untouched). Checkpointed: re-running skips done configs.
  B) Controls on all 36 configs:
       - FFORMA-lite: ~23 window-statistics features -> per-expert Ridge/GBM
         error prediction -> ranking, grouped-CV by dataset (spatial_type x
         seed) to avoid leakage from identical (alpha, kappa) twins.
       - Random-ranking negative control (100 reps/config).
  C) Risk decomposition R = sigma^2 + mismatch + est: nested-regression
     variance shares of MSE across the 36x8 cells.

Outputs:
  results/e1_v2/e1v2_train_seed3407.csv      (checkpoint, raw new runs)
  results/e1_v2/e1v2_combined.csv            (36 configs)
  results/e1_v2/e1v2_baselines_comparison.csv
  results/e1_v2/e1v2_variance_decomposition.csv
  results/figures/e1_v2_score_vs_fforma_comparison.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert, get_all_cards
from src.training.trainer import UnifiedTrainer
from src.probes.input_probe import InputProbe
from src.theory.affinity import SpectralAffinity

EXPERT_IDS = ["M52", "M03", "M01", "M117", "M36", "M17", "M14", "N01"]
ALPHAS = [0.5, 1.0, 2.0]
SPATIALS = ["lowrank", "aligned"]
KAPPAS = [0.0, 0.3]
OUT_DIR = "./results/e1_v2"
CKPT = f"{OUT_DIR}/e1v2_train_seed3407.csv"


class MockDM:
    def __init__(self, d):
        self.windows = d


def probe_feats_for_alpha(alpha):
    """Original E1 probe-feature formula (kept for protocol continuity)."""
    return {"low_freq_ratio": 0.5 + (alpha - 1.0) * 0.2,
            "spec_entropy": alpha, "spike_count": 0.0}


def spectral_rho_and_scores(alpha, rankings):
    """Replicate the original E1 spectral ranking; return rho + per-expert scores."""
    affinity_est = SpectralAffinity()
    cards = get_all_cards()
    pf = probe_feats_for_alpha(alpha)
    scores = {eid: affinity_est.mismatch_score(cards[eid], pf) for eid in EXPERT_IDS}
    pred_rank = sorted(EXPERT_IDS, key=lambda e: scores[e])  # low mismatch = better
    pred_ranks = {e: i + 1 for i, e in enumerate(pred_rank)}
    true_rank = sorted(EXPERT_IDS, key=lambda e: rankings[e])
    true_ranks = {e: i + 1 for i, e in enumerate(true_rank)}
    rho, pval = spearmanr([true_ranks[e] for e in EXPERT_IDS],
                          [pred_ranks[e] for e in EXPERT_IDS])
    return rho, pval, scores


def train_one_config(alpha, st, kappa, seed):
    cfg = SynthConfig(T=5000, V=8, H=24, alpha=alpha, spatial_type=st,
                      kappa_st=kappa, seed=seed)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    dm = MockDM({"train": data["train_inp"], "train_tgt": data["train_tgt"],
                 "val": data["val_inp"], "val_tgt": data["val_tgt"],
                 "test": data["test_inp"], "test_tgt": data["test_tgt"]})
    d_in = data["train_inp"].shape[1]
    rankings = {}
    for eid in EXPERT_IDS:
        set_seed(seed)
        expert = get_expert(eid, d_in, hidden=128, drop=0.1)
        trainer = UnifiedTrainer({"max_epochs": 5, "patience": 2,
                                  "batch_size": 256, "lr": 1e-4})
        try:
            rankings[eid] = trainer.train_expert(expert, dm)["test_mse"]
        except Exception as ex:
            print(f"  [warn] {eid} failed: {ex}")
            rankings[eid] = 999.0
    rho, pval, _ = spectral_rho_and_scores(alpha, rankings)
    rec = {"alpha": alpha, "spatial_type": st, "kappa": kappa, "seed": seed,
           "spearman_rho": rho, "pvalue": pval}
    rec.update({f"mse_{eid}": rankings[eid] for eid in EXPERT_IDS})
    return rec


def phase_train():
    ensure_dir(OUT_DIR)
    done = pd.read_csv(CKPT) if os.path.exists(CKPT) else pd.DataFrame()
    done_keys = set()
    if len(done):
        done_keys = set(zip(done.alpha, done.spatial_type, done.kappa, done.seed))
    recs = []
    for alpha in ALPHAS:
        for st in SPATIALS:
            for kappa in KAPPAS:
                if (alpha, st, kappa, 3407) in done_keys:
                    continue
                rec = train_one_config(alpha, st, kappa, 3407)
                recs.append(rec)
                pd.concat([done, pd.DataFrame(recs)], ignore_index=True).to_csv(CKPT, index=False)
                print(f"[e1v2] seed=3407 alpha={alpha} st={st} kappa={kappa} "
                      f"rho={rec['spearman_rho']:.3f}", flush=True)
    return pd.read_csv(CKPT)


FEAT_NAMES = ["mean", "std", "skew", "kurt", "q05", "q25", "q50", "q75", "q95",
              "spec_entropy", "low_freq_ratio", "dom_freq", "spec_slope",
              "slope", "acf1", "mad", "spike_count", "max_mean_ratio", "volatility"]
FFORMA_FEATS = ["mean", "std", "skew", "kurt", "acf1", "mad", "volatility",
                "low_freq_ratio", "spec_entropy", "spec_slope"]  # ~10 window stats


def config_features(st, seed, n_windows=64):
    """~10 window-statistics features for FFORMA-lite (deterministic);
    plus the full probe dict for the data-driven spectral score."""
    cfg = SynthConfig(T=5000, V=8, H=24, alpha=1.0, spatial_type=st, seed=seed)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    X = data["X"].numpy()
    probe = InputProbe()
    # window-level features on var-0 windows
    L = 24
    rng = np.random.RandomState(0)
    starts = rng.choice(len(X) - L, size=n_windows, replace=False)
    feats = np.stack([probe(X[s:s + L, 0]) for s in starts]).mean(axis=0)
    fdict = dict(zip(FEAT_NAMES, feats))
    x10 = np.array([fdict[k] for k in FFORMA_FEATS], dtype=np.float64)
    # data-driven spectral probe features (from a long var-0 stretch)
    long = probe(X[:1000, 0])
    ldict = dict(zip(FEAT_NAMES, long))
    spec_pf = {"low_freq_ratio": ldict["low_freq_ratio"],
               "spec_entropy": ldict["spec_entropy"],
               "spike_count": ldict["spike_count"]}
    return x10, spec_pf


def phase_analysis(df36):
    """FFORMA-lite + random control + variance decomposition on 36 configs."""
    mse_cols = [f"mse_{e}" for e in EXPERT_IDS]
    Y = df36[mse_cols].values  # (36, 8)
    df36 = df36.copy()
    df36["dataset"] = df36["spatial_type"] + "_" + df36["seed"].astype(str)

    # ---- spectral mismatch score per config x expert (for decomposition) ----
    cards = get_all_cards()
    aff = SpectralAffinity()
    M = np.zeros_like(Y)
    for i, row in df36.iterrows():
        pf = probe_feats_for_alpha(row["alpha"])
        for j, eid in enumerate(EXPERT_IDS):
            M[i, j] = aff.mismatch_score(cards[eid], pf)

    # ---- features per config + data-driven spectral scores ----
    feat_cache = {}
    feat_rows = []
    spec_pf_rows = []
    for _, row in df36.iterrows():
        key = (row["spatial_type"], int(row["seed"]))
        if key not in feat_cache:
            feat_cache[key] = config_features(*key)
        feat_rows.append(feat_cache[key][0])
        spec_pf_rows.append(feat_cache[key][1])
    X = np.stack(feat_rows)

    # data-driven spectral mismatch scores (probe features measured on data)
    M_real = np.zeros_like(Y)
    for i, pf in enumerate(spec_pf_rows):
        for j, eid in enumerate(EXPERT_IDS):
            M_real[i, j] = aff.mismatch_score(cards[eid], pf)

    true_ranks = np.argsort(np.argsort(Y, axis=1), axis=1) + 1  # rank 1 = best

    def rho_of(pred_score):
        # pred_score: (36, 8), higher = worse predicted error
        pr = np.argsort(np.argsort(pred_score, axis=1), axis=1) + 1
        return np.array([spearmanr(true_ranks[i], pr[i]).statistic
                         for i in range(len(df36))])

    # ---- FFORMA-lite: grouped CV by dataset (avoids identical-twin leakage) ----
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    groups = df36["dataset"].values
    uniq = sorted(set(groups))
    Y_log = np.log(Y + 1e-8)
    pred_ridge = np.zeros_like(Y)
    pred_gbm = np.zeros_like(Y)
    for g in uniq:
        te = groups == g
        tr = ~te
        for j in range(len(EXPERT_IDS)):
            r = Ridge(alpha=1.0)
            r.fit(X[tr], Y_log[tr, j])
            pred_ridge[te, j] = r.predict(X[te])
            m = HistGradientBoostingRegressor(max_iter=80, max_depth=3,
                                              learning_rate=0.1, random_state=0)
            m.fit(X[tr], Y_log[tr, j])
            pred_gbm[te, j] = m.predict(X[te])
    rho_ridge = rho_of(pred_ridge)
    rho_gbm = rho_of(pred_gbm)
    rho_spec_real = rho_of(M_real)

    # ---- random negative control: 100 random rankings per config ----
    rng = np.random.RandomState(0)
    rho_rand = []
    for rep in range(100):
        rr = np.array([rng.permutation(len(EXPERT_IDS)) + 1
                       for _ in range(len(df36))])
        rho_rand.append([spearmanr(true_ranks[i], rr[i]).statistic
                         for i in range(len(df36))])
    rho_rand = np.array(rho_rand)  # (100, 36)

    rho_spec = df36["spearman_rho"].values
    comp = pd.DataFrame({
        "config": df36["alpha"].astype(str) + "/" + df36["spatial_type"]
                  + "/k" + df36["kappa"].astype(str) + "/s" + df36["seed"].astype(str),
        "dataset": df36["dataset"],
        "rho_spectral_alphaformula": rho_spec,
        "rho_spectral_realfeat": rho_spec_real,
        "rho_fforma_ridge": rho_ridge,
        "rho_fforma_gbm": rho_gbm,
        "rho_random_mean": rho_rand.mean(axis=0),
    })
    comp.to_csv(f"{OUT_DIR}/e1v2_baselines_comparison.csv", index=False)
    print("\n=== E1-v2 baseline comparison (mean Spearman over 36 configs) ===")
    print(f"  spectral (alpha-formula, original E1): {np.mean(rho_spec):.3f}")
    print(f"  spectral (data-driven probe features): {np.mean(rho_spec_real):.3f}")
    print(f"  FFORMA-lite ridge : {np.mean(rho_ridge):.3f}")
    print(f"  FFORMA-lite GBM   : {np.mean(rho_gbm):.3f}")
    print(f"  random (100 reps) : {rho_rand.mean():.3f} "
          f"[95% CI {np.percentile(rho_rand, 2.5):.3f}, {np.percentile(rho_rand, 97.5):.3f}]")
    # paired tests spectral vs baselines
    from scipy.stats import wilcoxon
    for label, base in [("alpha-formula", rho_spec), ("realfeat", rho_spec_real)]:
        for name, arr in [("fforma_ridge", rho_ridge), ("fforma_gbm", rho_gbm),
                          ("random_mean", rho_rand.mean(axis=0))]:
            d = base - arr
            try:
                p_gt = wilcoxon(d, alternative="greater").pvalue
                p_lt = wilcoxon(d, alternative="less").pvalue
            except Exception:
                p_gt = p_lt = np.nan
            print(f"  spectral({label}) vs {name}: mean diff={d.mean():+.3f} "
                  f"p(greater)={p_gt:.4g} p(less)={p_lt:.4g}")

    # ---- variance decomposition: R = sigma^2 + mismatch + est ----
    # nested regression on all 36x8 cells
    y = Y.flatten()
    config_dummies = np.repeat(pd.get_dummies(df36["dataset"]).values, 8, axis=0)
    m = M.flatten()
    m_real = M_real.flatten()

    def r2_of(design):
        design = np.column_stack([np.ones(len(y)), design])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        return 1.0 - resid.var() / y.var()

    r2_config = r2_of(config_dummies)
    r2_full = r2_of(np.column_stack([config_dummies, m]))
    r2_full_real = r2_of(np.column_stack([config_dummies, m_real]))
    r2_mismatch_only = r2_of(m.reshape(-1, 1))
    total_var = y.var()
    shares = {
        "sigma2_between_dataset_share": r2_config,
        "mismatch_share_of_total_alphaformula": max(0.0, r2_full - r2_config),
        "mismatch_share_of_total_realfeat": max(0.0, r2_full_real - r2_config),
        "mismatch_share_marginal": r2_mismatch_only,
        "est_residual_share": 1.0 - r2_full,
        "r2_config_only": r2_config,
        "r2_config_plus_mismatch": r2_full,
    }
    # within-config (z-scored) mismatch explanatory power
    zY = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-12)
    for tag, MM in [("alphaformula", M), ("realfeat", M_real)]:
        zM = (MM - MM.mean(axis=1, keepdims=True)) / (MM.std(axis=1, keepdims=True) + 1e-12)
        zy, zm = zY.flatten(), zM.flatten()
        b = np.polyfit(zm, zy, 1)
        shares[f"within_config_mismatch_r2_{tag}"] = 1 - (zy - np.polyval(b, zm)).var() / zy.var()
    vd = pd.DataFrame([shares])
    vd.to_csv(f"{OUT_DIR}/e1v2_variance_decomposition.csv", index=False)
    print("\n=== E1-v2 variance decomposition (MSE over 36x8 cells) ===")
    for k, v in shares.items():
        print(f"  {k}: {v:.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    data_plot = [rho_spec, rho_spec_real, rho_ridge, rho_gbm, rho_rand.mean(axis=0)]
    bp = ax.boxplot(data_plot, tick_labels=["Spectral\n(alpha-formula)", "Spectral\n(data-driven)",
                                       "FFORMA-lite\n(ridge)", "FFORMA-lite\n(GBM)",
                                       "Random\n(mean of 100)"],
                    showmeans=True)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Spearman rho vs true expert ranking")
    ax.set_title("E1-v2: ranking prediction methods (36 configs)")
    ax = axes[1]
    ax.scatter(M_real.flatten(), Y.flatten(), s=10, alpha=0.5)
    ax.set_xlabel("spectral mismatch score (data-driven)")
    ax.set_ylabel("test MSE")
    ax.set_title(f"Mismatch vs risk (within-config R^2={shares['within_config_mismatch_r2_realfeat']:.2f})")
    fig.tight_layout()
    ensure_dir("./results/figures")
    fig.savefig("./results/figures/e1_v2_score_vs_fforma_comparison.png",
                dpi=150, bbox_inches="tight")
    print("\nSaved results/figures/e1_v2_score_vs_fforma_comparison.png")


def main():
    new_df = phase_train()
    old = pd.read_csv("./results/e1_synthetic_spectral.csv")
    old = old[old["seed"].isin([2021, 42])]
    combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined.sort_values(["spatial_type", "seed", "alpha", "kappa"]).reset_index(drop=True)
    assert len(combined) == 36, f"expected 36 configs, got {len(combined)}"
    combined.to_csv(f"{OUT_DIR}/e1v2_combined.csv", index=False)
    print(f"[e1v2] combined 36 configs -> {OUT_DIR}/e1v2_combined.csv")
    print(f"[e1v2] mean spectral Spearman (36 cfg): {combined['spearman_rho'].mean():.3f}")
    phase_analysis(combined)


if __name__ == "__main__":
    main()
