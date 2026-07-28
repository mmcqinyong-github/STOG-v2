"""E1 v3: reviewer-driven re-computation on the corrected generator (pure analysis).

Background: the v1 "analytic spectral variant" (mean Spearman 0.664 over 36
configs) was computed on the defective generator where alpha only
parameterized the unused analytic spectrum H* and did NOT shape the data
spectrum. The corrected generator (SynthConfig.alpha_filter=True, E2-v2
design) adds a per-variable 1/f^alpha colored component so alpha genuinely
stratifies the data spectrum. v2 fixed the generator but never re-reported
the analytic variant. This script closes that gap without any training:

  Task 2 (analytic variant): regenerate the 36 e1v2 configs (same grid, same
    seeds) on the corrected generator and score each config with the ANALYTIC
    spectral variant -- probe features derived from the generator's analytic
    spectrum H*(omega) = 1/(1+|omega|^alpha) instead of a data-driven probe.
    Spearman vs the stored measured expert rankings in
    results/e1_v2/e1v2_combined.csv. Compare against 0.664 (old alpha-formula)
    and 0.738 (v2 data-driven probe).
    -> results/e1_v3/e1v3_analytic_variant.csv

  Task 3 (Welch sensitivity): for the same 36 regenerated fields, estimate
    the var-0 spectrum with (a) the periodogram (status quo: |FFT| of the
    first 1000 samples, as in e1v2) and (b) Welch (segment 64, 50% overlap),
    recompute alpha_hat (log-log amplitude slope) and the data-driven
    spectral mismatch score under each estimator, and compare probe->rank
    Spearman, stratified by alpha.
    -> results/e1_v3/e1v3_welch_sensitivity.csv

  Figure: results/figures/e1_v3_analytic_vs_datadriven.png
    panel A: analytic-H* vs alpha-formula (0.664) vs data-driven (0.738)
    panel B: periodogram vs Welch probe->rank Spearman by alpha

Checkpointed: regenerated fields are cached under results/e1_v3/fields/ and
re-running skips existing caches.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from src.utils.common import ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_all_cards
from src.theory.affinity import SpectralAffinity
from src.probes.input_probe import InputProbe

EXPERT_IDS = ["M52", "M03", "M01", "M117", "M36", "M17", "M14", "N01"]
ALPHAS = [0.5, 1.0, 2.0]
SPATIALS = ["lowrank", "aligned"]
KAPPAS = [0.0, 0.3]
SEEDS = [2021, 42, 3407]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "e1_v3")
FIELD_DIR = os.path.join(OUT_DIR, "fields")
FIG_DIR = os.path.join(ROOT, "results", "figures")
E1V2_DIR = os.path.join(ROOT, "results", "e1_v2")
ensure_dir(OUT_DIR); ensure_dir(FIELD_DIR); ensure_dir(FIG_DIR)

PROBE = InputProbe()
FEAT_NAMES = ["mean", "std", "skew", "kurt", "q05", "q25", "q50", "q75", "q95",
              "spec_entropy", "low_freq_ratio", "dom_freq", "spec_slope",
              "slope", "acf1", "mad", "spike_count", "max_mean_ratio", "volatility"]


# ----------------------------------------------------------------------------
# Field cache: corrected generator (alpha_filter=True), same grid/seeds as e1v2
# ----------------------------------------------------------------------------
def field_path(alpha, st, kappa, seed):
    return os.path.join(FIELD_DIR, f"field_a{alpha}_{st}_k{kappa}_s{seed}.npz")


def gen_fields():
    for alpha in ALPHAS:
        for st in SPATIALS:
            for kappa in KAPPAS:
                for seed in SEEDS:
                    p = field_path(alpha, st, kappa, seed)
                    if os.path.exists(p):
                        continue
                    cfg = SynthConfig(T=5000, V=8, H=24, alpha=alpha,
                                      spatial_type=st, kappa_st=kappa, seed=seed,
                                      alpha_filter=True, alpha_strength=1.0)
                    data = SpatioTemporalFieldGenerator(cfg).generate()
                    np.savez(p, x0=data["X"][:, 0].numpy(),
                             alpha=alpha, spatial_type=st, kappa=kappa, seed=seed)
                    print(f"[field] a={alpha} {st} k={kappa} s={seed} cached",
                          flush=True)


def spec_feats_from_amp(amp):
    """InputProbe-style spectral features from an amplitude spectrum."""
    total = amp.sum() + 1e-10
    p = amp / total
    spec_entropy = -np.sum(p * np.log(p + 1e-10))
    low_freq_ratio = amp[:max(1, len(amp) // 4)].sum() / total
    return {"low_freq_ratio": float(low_freq_ratio),
            "spec_entropy": float(spec_entropy)}


def alpha_hat_from_amp(freqs, amp):
    """alpha_hat = -slope of log-log amplitude spectrum (f > 0)."""
    m = freqs > 0
    return float(-np.polyfit(np.log(freqs[m]), np.log(amp[m] + 1e-12), 1)[0])


# ----------------------------------------------------------------------------
# Task 2: analytic spectral variant from H* on the corrected generator
# ----------------------------------------------------------------------------
def analytic_pf(alpha, n=1000):
    """Probe features from the generator's analytic spectrum H* (no data)."""
    cfg = SynthConfig(alpha=alpha)
    H_star = SpatioTemporalFieldGenerator(cfg).true_operator()
    k = np.arange(n // 2)
    freqs = k / n                       # cycles/sample, matches FFT grid
    omega = 2 * np.pi * freqs
    amp = np.array([H_star(w, 0.0) for w in omega])  # temporal amplitude |H*|
    return spec_feats_from_amp(amp)


def phase_analytic(df36, base_cmp):
    cards = get_all_cards()
    aff = SpectralAffinity()
    mse_cols = [f"mse_{e}" for e in EXPERT_IDS]
    Y = df36[mse_cols].values
    true_ranks = np.argsort(np.argsort(Y, axis=1), axis=1) + 1

    pf_cache = {a: analytic_pf(a) for a in ALPHAS}
    rows = []
    rho_new = []
    for i, row in df36.iterrows():
        pf = dict(pf_cache[row["alpha"]]); pf["spike_count"] = 0.0
        scores = np.array([aff.mismatch_score(cards[e], pf) for e in EXPERT_IDS])
        pr = np.argsort(np.argsort(scores)) + 1
        rho = spearmanr(true_ranks[i], pr).statistic
        rho_new.append(rho)
        rows.append({"alpha": row["alpha"], "spatial_type": row["spatial_type"],
                     "kappa": row["kappa"], "seed": row["seed"],
                     "low_freq_ratio_Hstar": pf["low_freq_ratio"],
                     "spec_entropy_Hstar": pf["spec_entropy"],
                     "rho_analytic_hstar_corrected": rho,
                     "rho_alpha_formula_defective": row["spearman_rho"],
                     "rho_datadriven_v2": base_cmp["rho_spectral_realfeat"].iloc[i]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "e1v3_analytic_variant.csv"), index=False)
    rho_new = np.array(rho_new)
    print("\n=== Task 2: analytic spectral variant, corrected generator ===")
    print(f"  analytic-H* (corrected gen) mean rho: {rho_new.mean():.3f} "
          f"(median {np.median(rho_new):.3f})")
    print(f"  old alpha-formula (defective gen)   : {df36['spearman_rho'].mean():.3f}")
    print(f"  data-driven probe (v2)              : {base_cmp['rho_spectral_realfeat'].mean():.3f}")
    for a in ALPHAS:
        m = out["alpha"] == a
        print(f"  alpha={a}: analytic-H* rho={out.loc[m, 'rho_analytic_hstar_corrected'].mean():.3f} "
              f"| alpha-formula={out.loc[m, 'rho_alpha_formula_defective'].mean():.3f} "
              f"| data-driven={out.loc[m, 'rho_datadriven_v2'].mean():.3f}")
    return out


# ----------------------------------------------------------------------------
# Task 3: Welch vs periodogram sensitivity
# ----------------------------------------------------------------------------
def phase_welch(df36):
    cards = get_all_cards()
    aff = SpectralAffinity()
    mse_cols = [f"mse_{e}" for e in EXPERT_IDS]
    Y = df36[mse_cols].values
    true_ranks = np.argsort(np.argsort(Y, axis=1), axis=1) + 1

    rows = []
    for i, row in df36.iterrows():
        d = np.load(field_path(row["alpha"], row["spatial_type"],
                               row["kappa"], int(row["seed"])))
        x0 = d["x0"]
        # time-domain spike count (shared by both estimators)
        td = dict(zip(FEAT_NAMES, PROBE(x0[:1000])))
        spike = td["spike_count"]

        # (a) periodogram, status quo: |FFT| of first 1000 samples
        seg = x0[:1000]
        amp_p = np.abs(np.fft.fft(seg))[:len(seg) // 2]
        freqs_p = np.fft.fftfreq(len(seg))[:len(seg) // 2]
        ah_p = alpha_hat_from_amp(freqs_p, amp_p)
        pf_p = spec_feats_from_amp(amp_p); pf_p["spike_count"] = spike

        # (b) Welch: segment 64, 50% overlap
        f_w, Pxx = welch(x0, fs=1.0, nperseg=64, noverlap=32)
        amp_w = np.sqrt(Pxx)
        ah_w = alpha_hat_from_amp(f_w, amp_w)
        pf_w = spec_feats_from_amp(amp_w); pf_w["spike_count"] = spike

        def rho_of(pf):
            scores = np.array([aff.mismatch_score(cards[e], pf) for e in EXPERT_IDS])
            pr = np.argsort(np.argsort(scores)) + 1
            return spearmanr(true_ranks[i], pr).statistic

        rho_p, rho_w = rho_of(pf_p), rho_of(pf_w)
        rows.append({"alpha": row["alpha"], "spatial_type": row["spatial_type"],
                     "kappa": row["kappa"], "seed": int(row["seed"]),
                     "alpha_hat_periodogram": ah_p, "alpha_hat_welch": ah_w,
                     "low_freq_ratio_periodogram": pf_p["low_freq_ratio"],
                     "low_freq_ratio_welch": pf_w["low_freq_ratio"],
                     "spec_entropy_periodogram": pf_p["spec_entropy"],
                     "spec_entropy_welch": pf_w["spec_entropy"],
                     "rho_periodogram": rho_p, "rho_welch": rho_w,
                     "rho_diff_welch_minus_perio": rho_w - rho_p})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "e1v3_welch_sensitivity.csv"), index=False)
    print("\n=== Task 3: Welch vs periodogram sensitivity ===")
    print(f"  probe->rank Spearman: periodogram {out['rho_periodogram'].mean():.3f} "
          f"vs Welch {out['rho_welch'].mean():.3f} "
          f"(mean diff {out['rho_diff_welch_minus_perio'].mean():+.3f})")
    print("  alpha_hat (true -> perio / welch):")
    for a in ALPHAS:
        m = out["alpha"] == a
        print(f"  alpha={a}: alpha_hat perio={out.loc[m,'alpha_hat_periodogram'].mean():.2f} "
              f"welch={out.loc[m,'alpha_hat_welch'].mean():.2f} | "
              f"rho perio={out.loc[m,'rho_periodogram'].mean():.3f} "
              f"welch={out.loc[m,'rho_welch'].mean():.3f}")
    return out


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(an, we, df36, base_cmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    ax = axes[0]
    data_plot = [an["rho_analytic_hstar_corrected"].values,
                 df36["spearman_rho"].values,
                 base_cmp["rho_spectral_realfeat"].values]
    ax.boxplot(data_plot,
               tick_labels=["Analytic H*\n(corrected gen)",
                            "Alpha-formula\n(defective gen, 0.664)",
                            "Data-driven\n(v2, 0.738)"],
               showmeans=True)
    for j, vals in enumerate(data_plot):
        ax.scatter(np.full(len(vals), j + 1) + np.random.RandomState(0).uniform(-0.08, 0.08, len(vals)),
                   vals, s=14, alpha=0.5, c="steelblue", zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Spearman $\\rho$ vs measured expert ranking")
    ax.set_title(f"E1 v3: analytic variant re-computed on corrected generator\n"
                 f"mean $\\rho$={data_plot[0].mean():.3f} (n=36 configs)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(ALPHAS))
    mp = [we.loc[we["alpha"] == a, "rho_periodogram"].mean() for a in ALPHAS]
    mw = [we.loc[we["alpha"] == a, "rho_welch"].mean() for a in ALPHAS]
    sp = [we.loc[we["alpha"] == a, "rho_periodogram"].std() for a in ALPHAS]
    sw = [we.loc[we["alpha"] == a, "rho_welch"].std() for a in ALPHAS]
    ax.bar(x - 0.2, mp, width=0.4, yerr=sp, capsize=4, label="periodogram (status quo)",
           color="gray")
    ax.bar(x + 0.2, mw, width=0.4, yerr=sw, capsize=4,
           label="Welch (seg 64, 50% overlap)", color="teal")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\alpha$={a}" for a in ALPHAS])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("probe$\\to$rank Spearman $\\rho$")
    ax.set_title(f"E1 v3: Welch sensitivity by $\\alpha$\n"
                 f"overall perio={we['rho_periodogram'].mean():.3f} "
                 f"welch={we['rho_welch'].mean():.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e1_v3_analytic_vs_datadriven.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nfigure saved: results/figures/e1_v3_analytic_vs_datadriven.png")


def main():
    gen_fields()
    df36 = pd.read_csv(os.path.join(E1V2_DIR, "e1v2_combined.csv"))
    base_cmp = pd.read_csv(os.path.join(E1V2_DIR, "e1v2_baselines_comparison.csv"))
    assert len(df36) == 36 and len(base_cmp) == 36
    an = phase_analytic(df36, base_cmp)
    we = phase_welch(df36)
    make_figure(an, we, df36, base_cmp)


if __name__ == "__main__":
    main()
