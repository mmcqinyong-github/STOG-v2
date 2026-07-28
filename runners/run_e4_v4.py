"""E4 v4: amplitude-balanced re-test of Theorem 4 (reviewer-driven).

E4 v3 found regime 1 amplitude ~6x regime 0, which put Theorem 4 outside its
assumptions (gating benefit was driven by MSE amplitude dominance, not regime
complementarity; v3 S4 R^2=0.096, oracle R^2=0.342). v4 regenerates the same
12 fields (delta x seed) with the generator's new `amplitude_balance=True`
switch (regime-1 coefficient matrix rescaled so both regimes emit comparable
amplitude; measured ratio printed by the generator and cached per field) and
re-runs the v3 protocol unchanged otherwise:
  - regime-specialized pool (M52_r0/M52_r1/M03_r0/M03_r1, TRUE regime labels)
    + generalist controls (M52_gen/M03_gen), same tiered purity rule,
    same per-expert gradient-step budget and same-regime early stopping;
  - S1 static simplex, S4 dual-score window routing, S0 oracle;
  - benefit ∝ (1-delta) regression, pre-registered threshold R^2 >= 0.7.

Protocol note (regime observability): under amplitude balance the energy-GMM
regime detector of v2/v3 is unidentifiable BY DESIGN (it keyed on the 6x
amplitude gap). Since v3 already constructed specialists from TRUE regime
labels, v4 stays inside the theorem's conditions and also derives the window
mixedness m_hat and delta_hat from the TRUE regime sequence:
  m_hat(i) = 2*min(f_i, 1-f_i), f_i = true regime-1 fraction of input window
  delta_hat = empirical transition rate of the true regime sequence.
This is an oracle-regime-label, amplitude-balanced, in-scope test of Theorem 4.

Outputs (results/e4_v4/):
  cache/field_*.npz, e4v4_runs.csv, e4v4_summary.csv, e4v4_diagnostics.csv,
  e4v4_vs_v3_comparison.csv
  results/figures/e4_v4_balanced_benefit_vs_delta.png

Usage:
  python run_e4_v4.py --fields 0.1,2021 0.1,42   # trains experts, caches preds
  python run_e4_v4.py --combine                  # metrics, pairing, CSVs, figure
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress, spearmanr, wilcoxon

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

import run_e4_v2 as v2  # reuse: true_window_regime_frac, strategies, PROBE
import run_e4_v3 as v3  # reuse: span_regime_frac, TIERS, pool definition

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "e4_v4")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
FIG_DIR = os.path.join(ROOT, "results", "figures")
V3_DIR = os.path.join(ROOT, "results", "e4_v3")
ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR); ensure_dir(FIG_DIR)

H = v2.H
V = v2.V
EXPERT_NAMES = v3.EXPERT_NAMES
TIERS = v3.TIERS
MIN_CLASS = v3.MIN_CLASS
DELTAS = v2.DELTAS
SEEDS = v2.SEEDS


# ----------------------------------------------------------------------------
# Field runner: same as v3 but amplitude-balanced fields + true-label mixedness
# ----------------------------------------------------------------------------
def run_field(delta, seed):
    tag = f"d{delta}_s{seed}"
    cache_path = os.path.join(CACHE_DIR, f"field_{tag}.npz")
    if os.path.exists(cache_path):
        print(f"[field {tag}] cache exists, skip")
        return
    t0 = time.time()
    cfg = SynthConfig(T=5000, V=V, H=H, alpha=1.0, delta=delta, seed=seed,
                      amplitude_balance=True)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    amp_ratio = float(data["amplitude_ratio_r1_over_r0"])
    n_train = data["train_inp"].shape[0]; n_val = data["val_inp"].shape[0]
    n = n_train + n_val + data["test_inp"].shape[0]

    inp_all = torch.cat([data["train_inp"], data["val_inp"], data["test_inp"]]).numpy()
    tr_inp, va_inp, te_inp = inp_all[:n_train], inp_all[n_train:n_train+n_val], inp_all[n_train+n_val:]

    # ---- TRUE-label regime stats (energy GMM unidentifiable under balance) --
    regimes = data["regimes"]
    fr_tr, fr_va, fr_te = v2.true_window_regime_frac(regimes, n_train, n_val, n)

    def true_mixedness(reg_slice, n_win):
        """m_hat from TRUE regime-1 fraction over each input window (len H)."""
        f = np.array([reg_slice[i:i + H].mean() for i in range(n_win)])
        return 2.0 * np.minimum(f, 1.0 - f)

    m_tr = true_mixedness(regimes[:n_train], n_train)
    m_va = true_mixedness(regimes[n_train:n_train + n_val], n_val)
    m_te = true_mixedness(regimes[n_train + n_val:], n - n_train - n_val)
    q_bar = float((regimes[1:] != regimes[:-1]).mean())  # true transition rate
    delta_hat = q_bar  # unbiased under the known Markov switching family

    # ---- probe features (same as v2/v3) --------------------------------------
    def probe_feats(inp):
        v0 = inp.reshape(inp.shape[0], H, V)[:, :, 0]
        return np.stack([v2.PROBE(v0[i]) for i in range(inp.shape[0])])
    z_tr, z_va, z_te = probe_feats(tr_inp), probe_feats(va_inp), probe_feats(te_inp)

    # ---- regime-specialized train split (TRUE labels, v3 tiered rule) --------
    fr_span_tr = v3.span_regime_frac(regimes, n_train)
    m0 = m1 = None
    rule = None
    for tname, lo, hi in TIERS:
        c0 = fr_span_tr <= lo if lo < 0.5 else fr_span_tr < 0.5
        c1 = fr_span_tr >= hi
        if c0.sum() >= MIN_CLASS and c1.sum() >= MIN_CLASS:
            m0, m1, rule = c0, c1, tname
            break
    assert m0 is not None
    pur_r0 = float(fr_span_tr[m0].mean())
    pur_r1 = float(fr_span_tr[m1].mean())
    print(f"[field {tag}] rule={rule}  n_r0={int(m0.sum())} (f1={pur_r0:.3f})  "
          f"n_r1={int(m1.sum())} (f1={pur_r1:.3f})  amp_ratio={amp_ratio:.3f}")

    class MockDM:
        def __init__(self, d): self.windows = d

    full = {
        "train": data["train_inp"], "train_tgt": data["train_tgt"],
        "val": data["val_inp"], "val_tgt": data["val_tgt"],
        "test": data["test_inp"], "test_tgt": data["test_tgt"],
    }
    m0_t = torch.tensor(m0); m1_t = torch.tensor(m1)

    fr_span_va = v3.span_regime_frac(regimes[n_train:], n_val)
    lo_t, hi_t = {"pure_0.25": (0.25, 0.75), "lean_0.35": (0.35, 0.65),
                  "tilt_0.45": (0.45, 0.55), "majority_0.5": (0.5, 0.5)}[rule]
    va0 = fr_span_va <= lo_t if lo_t < 0.5 else fr_span_va < 0.5
    va1 = fr_span_va >= hi_t
    if va0.sum() < 30: va0 = np.ones(n_val, dtype=bool)
    if va1.sum() < 30: va1 = np.ones(n_val, dtype=bool)
    va0_t, va1_t = torch.tensor(va0), torch.tensor(va1)

    ptr_list, pv_list, pt_list, val_mses = [], [], [], []
    for name in EXPERT_NAMES:
        arch, kind = name.split("_")
        set_seed(seed)
        expert = get_expert(arch, data["train_inp"].shape[1], hidden=128)
        if kind == "r0":
            dm = MockDM({**full,
                         "train": data["train_inp"][m0_t],
                         "train_tgt": data["train_tgt"][m0_t],
                         "val": data["val_inp"][va0_t],
                         "val_tgt": data["val_tgt"][va0_t]})
        elif kind == "r1":
            dm = MockDM({**full,
                         "train": data["train_inp"][m1_t],
                         "train_tgt": data["train_tgt"][m1_t],
                         "val": data["val_inp"][va1_t],
                         "val_tgt": data["val_tgt"][va1_t]})
        else:
            dm = MockDM(full)
        n_sub = dm.windows["train"].shape[0]
        bs = int(np.clip(n_sub // 14, 32, 256))
        trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2,
                                  "batch_size": bs, "lr": 1e-4})
        try:
            res = trainer.train_expert(expert, dm)
            expert.eval()
            dev = next(expert.parameters()).device
            with torch.no_grad():
                pr = expert(data["train_inp"].to(dev)).cpu()
                pv = expert(data["val_inp"].to(dev)).cpu()
                pt = expert(data["test_inp"].to(dev)).cpu()
            # FULL-val MSE for strategy static scores (comparable across experts)
            val_mses.append(float(((pv - data["val_tgt"]) ** 2).mean()))
        except Exception as ex:
            print(f"  [field {tag}] expert {name} failed: {ex}")
            pr = torch.zeros_like(data["train_tgt"])
            pv = torch.zeros_like(data["val_tgt"]); pt = torch.zeros_like(data["test_tgt"])
            val_mses.append(999.0)
        ptr_list.append(pr.numpy()); pv_list.append(pv.numpy()); pt_list.append(pt.numpy())
        print(f"  [field {tag}] {name} done ({time.time()-t0:.0f}s)")

    np.savez(cache_path,
             delta=delta, seed=seed, delta_hat=delta_hat, amp_ratio=amp_ratio,
             q_bar=q_bar, m_bar=float(m_tr.mean()),
             split_rule=rule, n_r0=int(m0.sum()), n_r1=int(m1.sum()),
             pur_r0=pur_r0, pur_r1=pur_r1,
             pr=np.stack(ptr_list), pv=np.stack(pv_list), pt=np.stack(pt_list),
             ytr=data["train_tgt"].numpy(),
             yv=data["val_tgt"].numpy(), yt=data["test_tgt"].numpy(),
             m_va=m_va, m_te=m_te, m_tr=m_tr,
             fr_va=fr_va, fr_te=fr_te, fr_tr=fr_tr,
             z_va=z_va, z_te=z_te, z_tr=z_tr,
             val_mses=np.array(val_mses))
    print(f"[field {tag}] saved ({time.time()-t0:.0f}s)  "
          f"delta_hat={delta_hat:.3f} amp_ratio={amp_ratio:.3f}")


# ----------------------------------------------------------------------------
# Combine: strategies + pairing vs v3 (unbalanced) + CSVs + figure
# ----------------------------------------------------------------------------
def combine():
    fields = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("field_"))
    if not fields:
        print("no cached fields"); return
    v3df = pd.read_csv(os.path.join(V3_DIR, "e4v3_runs.csv"))

    rows, diag_rows = [], []
    for fname in fields:
        d = np.load(os.path.join(CACHE_DIR, fname))
        delta, seed = float(d["delta"]), int(d["seed"])
        pr = torch.tensor(d["pr"])
        pv = torch.tensor(d["pv"]); pt = torch.tensor(d["pt"])
        ytr = torch.tensor(d["ytr"])
        yv = torch.tensor(d["yv"]); yt = torch.tensor(d["yt"])
        ztr = torch.tensor(d["z_tr"], dtype=torch.float32)
        zva = torch.tensor(d["z_va"], dtype=torch.float32)
        zte = torch.tensor(d["z_te"], dtype=torch.float32)
        mtr = torch.tensor(d["m_tr"], dtype=torch.float32)
        mva = torch.tensor(d["m_va"], dtype=torch.float32)
        mte = torch.tensor(d["m_te"], dtype=torch.float32)
        val_mses = d["val_mses"]
        mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
        ztr, zva, zte = (ztr - mu) / sd, (zva - mu) / sd, (zte - mu) / sd

        E = pv.shape[0]
        w_eq = torch.full((pt.shape[1], E), 1.0 / E)
        mse_eq = v2.mse_t(v2.combine_w(pt, w_eq), yt)

        w1, pred1 = v2.strat_S1_static(pv, yv, pt, yt)
        mse1 = v2.mse_t(pred1, yt)
        w4, pred4, beta4, dyncoef4 = v2.strat_S4_dual(
            ztr, mtr, pr, ytr, zva, mva, pv, yv, zte, mte, pt, val_mses, seed)
        mse4 = v2.mse_t(pred4, yt)

        per_te = ((pt - yt.unsqueeze(0)) ** 2).mean(dim=2)         # (E, n)
        mse0 = float(per_te.min(dim=0).values.mean())

        # ---- regime alignment diagnostic (affinity-based, self-contained) ----
        fr_tr, fr_te = d["fr_tr"], d["fr_te"]
        per_tr = ((pr - ytr.unsqueeze(0)) ** 2).mean(dim=2)
        best_tr = per_tr.argmin(dim=0).numpy()
        aff = np.array([fr_tr[best_tr == e].mean() if (best_tr == e).any()
                        else fr_tr.mean() for e in range(E)])
        best = per_te.argmin(dim=0).numpy()
        rho_align = spearmanr(fr_te, aff[best]).statistic
        w4n = w4.numpy()
        rho_w4 = spearmanr(fr_te, w4n @ aff).statistic

        gen_idx = [i for i, nm in enumerate(EXPERT_NAMES) if "gen" in nm]
        spec_idx = [i for i in range(E) if i not in gen_idx]
        mse_gen_best = float(per_te[gen_idx].min(dim=0).values.mean())
        mse_spec_best = float(per_te[spec_idx].min(dim=0).values.mean())

        for sname, mse_s in [("S1_static", mse1), ("S4_dual_score", mse4),
                              ("S0_oracle", mse0)]:
            rows.append({"delta": delta, "seed": seed, "strategy": sname,
                         "one_minus_delta": 1 - delta, "test_mse": mse_s,
                         "mse_equal": mse_eq, "mse_S1": mse1,
                         "benefit_vs_static": mse1 - mse_s,
                         "benefit_vs_equal": mse_eq - mse_s,
                         "S4_beta": beta4, "S4_dyn_coef_mean": dyncoef4,
                         "amp_ratio": float(d["amp_ratio"]),
                         "split_rule": str(d["split_rule"]),
                         "n_r0": int(d["n_r0"]), "n_r1": int(d["n_r1"]),
                         "pur_r0": float(d["pur_r0"]), "pur_r1": float(d["pur_r1"])})
        diag_rows.append({"delta": delta, "seed": seed,
                          "amp_ratio": float(d["amp_ratio"]),
                          "split_rule": str(d["split_rule"]),
                          "pur_r0": float(d["pur_r0"]), "pur_r1": float(d["pur_r1"]),
                          "rho_best_expert_regime_v4": rho_align,
                          "aff_spread_v4": float(aff.max() - aff.min()),
                          "rho_S4w_affinity_regime": rho_w4,
                          "mse_best_specialist": mse_spec_best,
                          "mse_best_generalist": mse_gen_best,
                          "oracle_spec_minus_gen": mse_gen_best - mse_spec_best,
                          "S4_beta": beta4,
                          "mse_S1": mse1, "mse_S4": mse4,
                          "S4_minus_S1_mse": mse4 - mse1})
        print(f"[combine] {fname}: S1={mse1:.4f} S4={mse4:.4f} oracle={mse0:.4f} "
              f"align={rho_align:+.3f} beta={beta4:.2f} amp={float(d['amp_ratio']):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "e4v4_runs.csv"), index=False)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(os.path.join(OUT_DIR, "e4v4_diagnostics.csv"), index=False)

    # ---- paired comparison vs v3 (same field, unbalanced generator) ----------
    cmp_rows = []
    for (delta, seed), g in df.groupby(["delta", "seed"]):
        v3g = v3df[(v3df["delta"] == delta) & (v3df["seed"] == seed)]
        row = {"delta": delta, "seed": seed}
        for s in ["S1_static", "S4_dual_score", "S0_oracle"]:
            row[f"v4_{s}_mse"] = float(g[g["strategy"] == s]["test_mse"].iloc[0])
            row[f"v4_{s}_benefit"] = float(g[g["strategy"] == s]["benefit_vs_static"].iloc[0])
            if len(v3g):
                row[f"v3_{s}_mse"] = float(v3g[v3g["strategy"] == s]["test_mse"].iloc[0])
                row[f"v3_{s}_benefit"] = float(v3g[v3g["strategy"] == s]["benefit_vs_static"].iloc[0])
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, "e4v4_vs_v3_comparison.csv"), index=False)

    # ---- summary --------------------------------------------------------------
    summ = []
    for sname, g in df.groupby("strategy"):
        if g["benefit_vs_static"].std() < 1e-12:
            continue
        lr_ = linregress(g["one_minus_delta"], g["benefit_vs_static"])
        rel = g["benefit_vs_static"] / g["mse_S1"]
        lrr = linregress(g["one_minus_delta"], rel)
        summ.append({"metric": "benefit_vs_static_R2", "strategy": sname, "value": lr_.rvalue ** 2})
        summ.append({"metric": "benefit_vs_static_slope", "strategy": sname, "value": lr_.slope})
        summ.append({"metric": "benefit_vs_static_p", "strategy": sname, "value": lr_.pvalue})
        summ.append({"metric": "benefit_vs_static_mean", "strategy": sname,
                     "value": g["benefit_vs_static"].mean()})
        summ.append({"metric": "rel_benefit_vs_static_R2", "strategy": sname,
                     "value": lrr.rvalue ** 2})
        summ.append({"metric": "rel_benefit_vs_static_p", "strategy": sname,
                     "value": lrr.pvalue})
    # v3 (unbalanced) R2 for reference
    for sname in ["S4_dual_score", "S0_oracle"]:
        g = v3df[v3df["strategy"] == sname]
        lr_ = linregress(g["one_minus_delta"], g["benefit_vs_static"])
        summ.append({"metric": "v3_unbalanced_benefit_vs_static_R2", "strategy": sname,
                     "value": lr_.rvalue ** 2})
        diff = cmp_df[f"v4_{sname}_benefit"] - cmp_df[f"v3_{sname}_benefit"]
        try:
            p = wilcoxon(cmp_df[f"v4_{sname}_benefit"], cmp_df[f"v3_{sname}_benefit"],
                         alternative="greater").pvalue
        except Exception:
            p = float("nan")
        summ.append({"metric": "paired_wilcoxon_p_v4_gt_v3", "strategy": sname, "value": p})
        summ.append({"metric": "paired_benefit_gain_mean", "strategy": sname, "value": diff.mean()})
    # diagnostics
    summ.append({"metric": "amplitude_ratio_mean", "strategy": "verification",
                 "value": diag["amp_ratio"].mean()})
    summ.append({"metric": "amplitude_ratio_max", "strategy": "verification",
                 "value": diag["amp_ratio"].max()})
    summ.append({"metric": "rho_best_expert_regime_mean", "strategy": "diagnostic",
                 "value": diag["rho_best_expert_regime_v4"].mean()})
    summ.append({"metric": "rho_S4w_affinity_regime_mean", "strategy": "S4_dual_score",
                 "value": diag["rho_S4w_affinity_regime"].mean()})
    summ.append({"metric": "oracle_spec_minus_gen_mean", "strategy": "diagnostic",
                 "value": diag["oracle_spec_minus_gen"].mean()})
    summ.append({"metric": "trainset_purity_gap_mean", "strategy": "diagnostic",
                 "value": (diag["pur_r1"] - diag["pur_r0"]).mean()})
    # H4-v4 verdict (pre-registered R^2 >= 0.7, amplitude-balanced scope)
    r2_s4 = [s["value"] for s in summ
             if s["metric"] == "benefit_vs_static_R2" and s["strategy"] == "S4_dual_score"][0]
    r2_or = [s["value"] for s in summ
             if s["metric"] == "benefit_vs_static_R2" and s["strategy"] == "S0_oracle"][0]
    verdict = (r2_s4 >= 0.7) or (r2_or >= 0.7)
    summ.append({"metric": "H4_v4_amplitude_balanced_verified", "strategy": "verdict",
                 "value": float(verdict)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(OUT_DIR, "e4v4_summary.csv"), index=False)

    make_figure(df, v3df)
    print("\n==== E4 v4 SUMMARY (amplitude-balanced) ====")
    print(sdf.to_string(index=False))
    print(f"\nH4-v4 verdict: {'VERIFIED (R2>=0.7)' if verdict else 'NOT verified'} "
          f"(S4 R2={r2_s4:.3f}, oracle R2={r2_or:.3f})")
    print(f"amplitude ratio: mean={diag['amp_ratio'].mean():.3f} "
          f"max={diag['amp_ratio'].max():.3f}")


def make_figure(df, v3df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, s, ttl in zip(axes, ["S4_dual_score", "S0_oracle"],
                          ["S4 dual-score routing", "S0 oracle (per-window best)"]):
        for dfx, lab, col, mk in [(v3df, "v3 unbalanced (amp ratio ~6)", "gray", "o"),
                                  (df, "v4 amplitude-balanced (ratio ~1)", "crimson", "s")]:
            g = dfx[dfx["strategy"] == s]
            x, y = g["one_minus_delta"].values, g["benefit_vs_static"].values
            ax.scatter(x, y, c=col, marker=mk, s=48, alpha=0.8, edgecolors="k",
                       lw=0.4, label=lab)
            if np.std(y) > 1e-12:
                lr_ = linregress(x, y)
                xs = np.linspace(0.05, 0.95, 50)
                ax.plot(xs, lr_.intercept + lr_.slope * xs, color=col, ls="--", lw=1.6,
                        label=f"{lab.split(' (')[0]} fit: $R^2$={lr_.rvalue**2:.3f}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"regime stability $1-\delta$")
        ax.set_ylabel("gating benefit vs static (MSE reduction)")
        ax.set_title(ttl)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("E4 v4: gating benefit $\\propto (1-\\delta)$ under amplitude balance "
                 "(Theorem-4 in-scope) vs v3 unbalanced (12 fields each)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v4_balanced_benefit_vs_delta.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figure saved: results/figures/e4_v4_balanced_benefit_vs_delta.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--fields", nargs="*", default=[], help="entries like 0.1,2021")
    args = ap.parse_args()
    for f in args.fields:
        d, s = f.split(",")
        run_field(float(d), int(s))
    if args.combine:
        combine()


if __name__ == "__main__":
    main()
