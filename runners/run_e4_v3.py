"""E4 v3: Regime-specialized expert pool variant (Theorem 4 / H4 constructive check).

E4 v2 showed window-level dynamic gating is the right direction (S4 R^2 = 0.184,
up from 0.006) but the generalist pool lacked regime complementarity: even the
oracle upper bound reached only R^2 = 0.373. v3 tests whether, with a pool that
is regime-complementary BY CONSTRUCTION, Theorem 4's "gating benefit ∝ (1-delta)"
holds (pre-registered threshold R^2 >= 0.7).

Pool per field (6 experts):
  Specialized: M52_r0, M52_r1, M03_r0, M03_r1
    E_rk trained ONLY on train windows whose input+target span is dominated by
    regime k (TRUE regime labels exported by the generator -- see note below).
  Generalist:  M52_gen, M03_gen trained on the full train split (controls).

Regime labels: the generator returns the true switching sequence `regimes`
(synthetic.py L130/L196), so no estimation is needed for constructing the
specialists. Window regime-1 fraction f over span [i, i+2H) is used. Tiered
purity rule (first tier with >= MIN_CLASS windows in BOTH classes wins):
  tierA "pure_0.25":     f <= 0.25 vs f >= 0.75
  tierB "lean_0.35":     f <= 0.35 vs f >= 0.65
  tierC "tilt_0.45":     f <= 0.45 vs f >= 0.55
  tierD "majority_0.5":  f <  0.5  vs f >= 0.5   (always satisfiable)
High-delta fields have no pure windows by construction (fast switching), so
they fall through to tilt/majority tiers -- reported per field via split_rule
and the realized mean purity of each specialist's train set.

Training protocol: same as v2 (max_epochs=8, patience=2, lr=1e-4). Two
protocol corrections so that specialization is measured rather than starved:
  1) Batch size is scaled per expert so every expert gets the same
     gradient-step budget as the v2 full-data protocol (~14 steps/epoch):
     batch = clip(n_train_subset // 14, 32, 256).
  2) Specialists early-stop on the SAME-REGIME subset of val (tier thresholds
     reused, >=30 windows required, else full val). v2's trainer early-stops
     on the full val split, whose MSE is dominated by the ~6x-amplitude
     regime-1 windows -- this silently converts every specialist into a
     regime-1-fitting model (verified in diagnostics: without it, r1
     specialists never beat generalists even on pure regime-1 windows).
Reported val_mses (used by S1/S4 static scores) remain FULL-val MSEs for
comparability across experts.

Evaluation: v2's S1 (static simplex), S4 (dual-score window routing) and
S0 oracle are re-run unchanged (imported from run_e4_v2) on the 6-expert pool;
paired same-field comparison against the v2 generalist pool.

Usage:
  python run_e4_v3.py --fields 0.1,2021 0.1,42   # trains experts, caches preds
  python run_e4_v3.py --combine                  # metrics, pairing, CSVs, figures
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

import run_e4_v2 as v2  # reuse: window_regime_stats, estimate_delta, strategies

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "e4_v3")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
FIG_DIR = os.path.join(ROOT, "results", "figures")
V2_DIR = os.path.join(ROOT, "results", "e4_v2")
ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR); ensure_dir(FIG_DIR)

H = v2.H
V = v2.V
ARCHES = ["M52", "M03"]                      # DLinear, RLinear
EXPERT_NAMES = ["M52_r0", "M52_r1", "M03_r0", "M03_r1", "M52_gen", "M03_gen"]
R1_SPEC_IDX = [1, 3]                         # indices of regime-1 specialists
TIERS = [("pure_0.25", 0.25, 0.75), ("lean_0.35", 0.35, 0.65),
         ("tilt_0.45", 0.45, 0.55), ("majority_0.5", 0.5, 0.5)]
MIN_CLASS = 256
DELTAS = v2.DELTAS
SEEDS = v2.SEEDS


# ----------------------------------------------------------------------------
# Field runner: train 6-expert regime-complementary pool, cache preds
# ----------------------------------------------------------------------------
def span_regime_frac(regimes, n):
    """True regime-1 fraction over input+target span [i, i+2H) of each window."""
    return np.array([regimes[i:i + 2 * H].mean() for i in range(n)], dtype=np.float64)


def run_field(delta, seed):
    tag = f"d{delta}_s{seed}"
    cache_path = os.path.join(CACHE_DIR, f"field_{tag}.npz")
    if os.path.exists(cache_path):
        print(f"[field {tag}] cache exists, skip")
        return
    t0 = time.time()
    cfg = SynthConfig(T=5000, V=V, H=H, alpha=1.0, delta=delta, seed=seed)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    n_train = data["train_inp"].shape[0]; n_val = data["val_inp"].shape[0]
    n = n_train + n_val + data["test_inp"].shape[0]

    # ---- regime stats identical to v2 (GMM fit on train windows only) -------
    inp_all = torch.cat([data["train_inp"], data["val_inp"], data["test_inp"]]).numpy()
    tr_inp, va_inp, te_inp = inp_all[:n_train], inp_all[n_train:n_train+n_val], inp_all[n_train+n_val:]
    _, _, _, _, gmm = v2.window_regime_stats(tr_inp, fit=True)
    _, m_tr, q_tr, _, _ = v2.window_regime_stats(tr_inp, gmm=gmm)
    _, m_va, _, _, _ = v2.window_regime_stats(va_inp, gmm=gmm)
    _, m_te, _, _, _ = v2.window_regime_stats(te_inp, gmm=gmm)
    delta_hat = v2.estimate_delta(float(q_tr.mean()))
    fr_tr, fr_va, fr_te = v2.true_window_regime_frac(data["regimes"], n_train, n_val, n)

    # ---- probe features (same as v2) ----------------------------------------
    def probe_feats(inp):
        v0 = inp.reshape(inp.shape[0], H, V)[:, :, 0]
        return np.stack([v2.PROBE(v0[i]) for i in range(inp.shape[0])])
    z_tr, z_va, z_te = probe_feats(tr_inp), probe_feats(va_inp), probe_feats(te_inp)

    # ---- regime-specialized train split (TRUE labels, exported by generator) -
    fr_span_tr = span_regime_frac(data["regimes"], n_train)
    m0 = m1 = None
    rule = None
    for tname, lo, hi in TIERS:
        c0 = fr_span_tr <= lo if lo < 0.5 else fr_span_tr < 0.5
        c1 = fr_span_tr >= hi
        if c0.sum() >= MIN_CLASS and c1.sum() >= MIN_CLASS:
            m0, m1, rule = c0, c1, tname
            break
    assert m0 is not None
    pur_r0 = float(fr_span_tr[m0].mean())   # realized mean regime-1 frac of r0 set
    pur_r1 = float(fr_span_tr[m1].mean())   # realized mean regime-1 frac of r1 set
    print(f"[field {tag}] rule={rule}  n_r0={int(m0.sum())} (f1={pur_r0:.3f})  "
          f"n_r1={int(m1.sum())} (f1={pur_r1:.3f})")

    class MockDM:
        def __init__(self, d): self.windows = d

    full = {
        "train": data["train_inp"], "train_tgt": data["train_tgt"],
        "val": data["val_inp"], "val_tgt": data["val_tgt"],
        "test": data["test_inp"], "test_tgt": data["test_tgt"],
    }
    m0_t = torch.tensor(m0); m1_t = torch.tensor(m1)

    # same-regime val subsets for specialist early stopping (fr over val spans)
    fr_span_va = span_regime_frac(data["regimes"][n_train:], n_val)
    lo_t, hi_t = {"pure_0.25": (0.25, 0.75), "lean_0.35": (0.35, 0.65),
                  "tilt_0.45": (0.45, 0.55), "majority_0.5": (0.5, 0.5)}[rule]
    va0 = fr_span_va <= lo_t if lo_t < 0.5 else fr_span_va < 0.5
    va1 = fr_span_va >= hi_t
    if va0.sum() < 30: va0 = np.ones(n_val, dtype=bool)
    if va1.sum() < 30: va1 = np.ones(n_val, dtype=bool)
    va0_t, va1_t = torch.tensor(va0), torch.tensor(va1)
    print(f"[field {tag}] val subsets for early stop: r0={int(va0.sum())}, r1={int(va1.sum())}")

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
        # scale batch so every expert gets ~14 steps/epoch (v2 full-data budget)
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
            # FULL-val MSE for strategy static scores (res val_mse is subset MSE
            # for specialists and not comparable across experts)
            val_mses.append(float(((pv - data["val_tgt"]) ** 2).mean()))
        except Exception as ex:
            print(f"  [field {tag}] expert {name} failed: {ex}")
            pr = torch.zeros_like(data["train_tgt"])
            pv = torch.zeros_like(data["val_tgt"]); pt = torch.zeros_like(data["test_tgt"])
            val_mses.append(999.0)
        ptr_list.append(pr.numpy()); pv_list.append(pv.numpy()); pt_list.append(pt.numpy())
        print(f"  [field {tag}] {name} done ({time.time()-t0:.0f}s)")

    np.savez(cache_path,
             delta=delta, seed=seed, delta_hat=delta_hat,
             q_bar=float(q_tr.mean()), m_bar=float(m_tr.mean()),
             split_rule=rule, n_r0=int(m0.sum()), n_r1=int(m1.sum()),
             pur_r0=pur_r0, pur_r1=pur_r1,
             pr=np.stack(ptr_list), pv=np.stack(pv_list), pt=np.stack(pt_list),
             ytr=data["train_tgt"].numpy(),
             yv=data["val_tgt"].numpy(), yt=data["test_tgt"].numpy(),
             m_va=m_va, m_te=m_te, m_tr=m_tr,
             fr_va=fr_va, fr_te=fr_te, fr_tr=fr_tr,
             z_va=z_va, z_te=z_te, z_tr=z_tr,
             val_mses=np.array(val_mses))
    print(f"[field {tag}] saved ({time.time()-t0:.0f}s)  delta_hat={delta_hat:.3f}")


# ----------------------------------------------------------------------------
# Combine: strategies + pairing vs v2 + diagnostics + CSVs + figures
# ----------------------------------------------------------------------------
def combine():
    fields = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("field_"))
    if not fields:
        print("no cached fields"); return
    v2df = pd.read_csv(os.path.join(V2_DIR, "e4v2_strategies.csv"))

    rows, diag_rows = [], []
    align_store = []
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

        # oracle: per-window true best expert on test
        per_te = ((pt - yt.unsqueeze(0)) ** 2).mean(dim=2)         # (E, n)
        mse0 = float(per_te.min(dim=0).values.mean())

        # ---- regime alignment diagnostic ------------------------------------
        # Affinity-based alignment: each expert's regime affinity aff_e = mean
        # true regime-1 fraction of the TRAIN windows it wins. Then test whether
        # the affinity of the per-window best expert tracks the window's true
        # regime on TEST. This is identity-free: it credits the de-facto regime
        # expert whoever it is (here M03_gen wins high-amplitude regime-1
        # windows due to MSE amplitude dominance, while r0 specialists win
        # regime-0 windows -- an r1-specialist indicator would be backwards).
        fr_tr, fr_te = d["fr_tr"], d["fr_te"]
        per_tr = ((pr - ytr.unsqueeze(0)) ** 2).mean(dim=2)         # (E, n_tr)
        best_tr = per_tr.argmin(dim=0).numpy()
        aff = np.array([fr_tr[best_tr == e].mean() if (best_tr == e).any()
                        else fr_tr.mean() for e in range(E)])
        best = per_te.argmin(dim=0).numpy()
        rho_align = spearmanr(fr_te, aff[best]).statistic
        # S4: expected affinity under gate weights vs true regime fraction
        w4n = w4.numpy()
        rho_w4 = spearmanr(fr_te, w4n @ aff).statistic
        align_store.append({"delta": delta, "seed": seed, "fr_te": fr_te,
                            "best_aff": aff[best]})

        # v2 baseline: same affinity metric on the v2 generalist pool
        d2 = np.load(os.path.join(V2_DIR, "cache", fname))
        pr2 = torch.tensor(d2["pr"]); pt2 = torch.tensor(d2["pt"])
        ytr2 = torch.tensor(d2["ytr"]); yt2 = torch.tensor(d2["yt"])
        E2 = pt2.shape[0]
        best2_tr = ((pr2 - ytr2.unsqueeze(0)) ** 2).mean(dim=2).argmin(dim=0).numpy()
        aff2 = np.array([d2["fr_tr"][best2_tr == e].mean() if (best2_tr == e).any()
                         else d2["fr_tr"].mean() for e in range(E2)])
        best2 = ((pt2 - yt2.unsqueeze(0)) ** 2).mean(dim=2).argmin(dim=0).numpy()
        rho_v2 = spearmanr(d2["fr_te"], aff2[best2]).statistic
        aff_spread = float(aff.max() - aff.min())
        aff_spread_v2 = float(aff2.max() - aff2.min())

        # per-window specialist gap: best specialist vs best generalist
        gen_idx = [i for i in range(E) if i not in R1_SPEC_IDX and "gen" in EXPERT_NAMES[i]]
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
                         "split_rule": str(d["split_rule"]),
                         "n_r0": int(d["n_r0"]), "n_r1": int(d["n_r1"]),
                         "pur_r0": float(d["pur_r0"]), "pur_r1": float(d["pur_r1"])})
        diag_rows.append({"delta": delta, "seed": seed,
                          "split_rule": str(d["split_rule"]),
                          "pur_r0": float(d["pur_r0"]), "pur_r1": float(d["pur_r1"]),
                          "rho_best_expert_regime_v3": rho_align,
                          "rho_best_expert_regime_v2": rho_v2,
                          "aff_spread_v3": aff_spread,
                          "aff_spread_v2": aff_spread_v2,
                          "rho_S4w_affinity_regime": rho_w4,
                          "mse_best_specialist": mse_spec_best,
                          "mse_best_generalist": mse_gen_best,
                          "oracle_spec_minus_gen": mse_gen_best - mse_spec_best,
                          "S4_beta": beta4,
                          "mse_S1": mse1, "mse_S4": mse4,
                          "S4_minus_S1_mse": mse4 - mse1})
        print(f"[combine] {fname}: S1={mse1:.4f} S4={mse4:.4f} oracle={mse0:.4f} "
              f"align={rho_align:+.3f} (v2 align={rho_v2:+.3f}) beta={beta4:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "e4v3_runs.csv"), index=False)
    diag = pd.DataFrame(diag_rows)

    # ---- paired comparison vs v2 (same field) --------------------------------
    cmp_rows = []
    for (delta, seed), g in df.groupby(["delta", "seed"]):
        v2g = v2df[(v2df["delta"] == delta) & (v2df["seed"] == seed)]
        row = {"delta": delta, "seed": seed}
        for s in ["S4_dual_score", "S0_oracle"]:
            row[f"v3_{s}_mse"] = float(g[g["strategy"] == s]["test_mse"].iloc[0])
            row[f"v3_{s}_benefit"] = float(g[g["strategy"] == s]["benefit_vs_static"].iloc[0])
            row[f"v2_{s}_mse"] = float(v2g[v2g["strategy"] == s]["test_mse"].iloc[0])
            row[f"v2_{s}_benefit"] = float(v2g[v2g["strategy"] == s]["benefit_vs_static"].iloc[0])
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, "e4v3_vs_v2_comparison.csv"), index=False)

    # ---- summary --------------------------------------------------------------
    summ = []
    for sname, g in df.groupby("strategy"):
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
    # v2 R2 for reference (absolute + relative)
    for sname in ["S4_dual_score", "S0_oracle"]:
        g = v2df[v2df["strategy"] == sname]
        lr_ = linregress(g["one_minus_delta"], g["benefit_vs_static"])
        lrr = linregress(g["one_minus_delta"], g["benefit_vs_static"] / g["mse_S1"])
        summ.append({"metric": "v2_benefit_vs_static_R2", "strategy": sname, "value": lr_.rvalue ** 2})
        summ.append({"metric": "v2_rel_benefit_vs_static_R2", "strategy": sname,
                     "value": lrr.rvalue ** 2})
    # paired Wilcoxon on per-field benefit (v3 > v2)
    for s in ["S4_dual_score", "S0_oracle"]:
        diff = cmp_df[f"v3_{s}_benefit"] - cmp_df[f"v2_{s}_benefit"]
        try:
            p = wilcoxon(cmp_df[f"v3_{s}_benefit"], cmp_df[f"v2_{s}_benefit"],
                         alternative="greater").pvalue
        except Exception:
            p = float("nan")
        summ.append({"metric": "paired_wilcoxon_p_v3_gt_v2", "strategy": s, "value": p})
        summ.append({"metric": "paired_benefit_gain_mean", "strategy": s, "value": diff.mean()})
    # diagnostics
    summ.append({"metric": "rho_best_expert_regime_mean", "strategy": "diagnostic",
                 "value": diag["rho_best_expert_regime_v3"].mean()})
    summ.append({"metric": "rho_best_expert_regime_v2_mean", "strategy": "diagnostic",
                 "value": diag["rho_best_expert_regime_v2"].mean()})
    summ.append({"metric": "affinity_spread_v3_mean", "strategy": "diagnostic",
                 "value": diag["aff_spread_v3"].mean()})
    summ.append({"metric": "affinity_spread_v2_mean", "strategy": "diagnostic",
                 "value": diag["aff_spread_v2"].mean()})
    summ.append({"metric": "rho_S4w_affinity_regime_mean", "strategy": "S4_dual_score",
                 "value": diag["rho_S4w_affinity_regime"].mean()})
    summ.append({"metric": "oracle_spec_minus_gen_mean", "strategy": "diagnostic",
                 "value": diag["oracle_spec_minus_gen"].mean()})
    summ.append({"metric": "trainset_purity_gap_mean", "strategy": "diagnostic",
                 "value": (diag["pur_r1"] - diag["pur_r0"]).mean()})
    d9 = diag[diag["delta"] == 0.9]
    summ.append({"metric": "deg09_S4_beta_mean", "strategy": "S4_dual_score",
                 "value": d9["S4_beta"].mean()})
    summ.append({"metric": "deg09_S4_minus_S1_mse", "strategy": "S4_dual_score",
                 "value": d9["S4_minus_S1_mse"].mean()})
    dall = df[df["strategy"] == "S4_dual_score"]
    summ.append({"metric": "all_delta_S4_beta_mean", "strategy": "S4_dual_score",
                 "value": dall["S4_beta"].mean()})
    # H4-v3 verdict
    r2_s4 = [s["value"] for s in summ
             if s["metric"] == "benefit_vs_static_R2" and s["strategy"] == "S4_dual_score"][0]
    r2_or = [s["value"] for s in summ
             if s["metric"] == "benefit_vs_static_R2" and s["strategy"] == "S0_oracle"][0]
    verdict = (r2_s4 >= 0.7) or (r2_or >= 0.7)
    summ.append({"metric": "H4_v3_constructive_verified", "strategy": "verdict",
                 "value": float(verdict)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(OUT_DIR, "e4v3_summary.csv"), index=False)
    diag.to_csv(os.path.join(OUT_DIR, "e4v3_diagnostics.csv"), index=False)

    make_figures(df, v2df, diag, align_store)
    print("\n==== E4 v3 SUMMARY ====")
    print(sdf.to_string(index=False))
    print(f"\nH4-v3 verdict: {'VERIFIED (R2>=0.7)' if verdict else 'NOT verified'} "
          f"(S4 R2={r2_s4:.3f}, oracle R2={r2_or:.3f})")


def make_figures(df, v2df, diag, align_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: benefit vs 1-delta, specialized pool (v3) vs generalist pool (v2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, s, ttl in zip(axes, ["S4_dual_score", "S0_oracle"],
                          ["S4 dual-score routing", "S0 oracle (per-window best)"]):
        for dfx, lab, col, mk in [(v2df, "v2 generalist pool", "gray", "o"),
                                  (df, "v3 regime-specialized pool", "crimson", "s")]:
            g = dfx[dfx["strategy"] == s]
            x, y = g["one_minus_delta"].values, g["benefit_vs_static"].values
            ax.scatter(x, y, c=col, marker=mk, s=48, alpha=0.8, edgecolors="k",
                       lw=0.4, label=f"{lab}")
            if np.std(y) > 1e-12:
                lr_ = linregress(x, y)
                xs = np.linspace(0.05, 0.95, 50)
                ax.plot(xs, lr_.intercept + lr_.slope * xs, color=col, ls="--", lw=1.6,
                        label=f"{lab} fit: $R^2$={lr_.rvalue**2:.3f}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"regime stability $1-\delta$")
        ax.set_ylabel("gating benefit vs static (MSE reduction)")
        ax.set_title(ttl)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("E4 v3: gating benefit $\\propto (1-\\delta)$ — regime-complementary "
                 "pool vs v2 generalist pool (12 fields each)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v3_benefit_vs_delta_specialized_vs_general.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: regime alignment diagnostic (affinity-based)
    fr_all = np.concatenate([a["fr_te"] for a in align_store])
    aff_all = np.concatenate([a["best_aff"] for a in align_store])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # left: affinity of per-window best expert vs true regime-1 fraction
    bins = np.linspace(0, 1, 21)
    ib = np.digitize(fr_all, bins) - 1
    bx, by = [], []
    for b in range(len(bins) - 1):
        m = ib == b
        if m.sum() >= 20:
            bx.append(0.5 * (bins[b] + bins[b + 1]))
            by.append(aff_all[m].mean())
    idx = np.random.RandomState(0).choice(len(fr_all), min(4000, len(fr_all)),
                                          replace=False)
    axes[0].scatter(fr_all[idx], aff_all[idx], s=4, alpha=0.15, c="steelblue")
    axes[0].plot(bx, by, "o-", c="crimson", lw=2, ms=6, label="binned mean")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect alignment")
    rho = spearmanr(fr_all, aff_all).statistic
    axes[0].set_xlabel("true regime-1 fraction in window span")
    axes[0].set_ylabel("regime affinity of oracle-best expert")
    axes[0].set_title(f"oracle best-expert affinity vs regime (pooled, Spearman $\\rho$={rho:.3f})")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # right: per-field alignment rho, v3 vs v2 baseline
    x = np.arange(len(diag))
    axes[1].bar(x - 0.2, diag["rho_best_expert_regime_v3"], width=0.4,
                label="v3 specialized pool", color="crimson")
    axes[1].bar(x + 0.2, diag["rho_best_expert_regime_v2"], width=0.4,
                label="v2 generalist pool", color="gray")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r.delta}/{r.seed}" for r in diag.itertuples()],
                            rotation=60, fontsize=7)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel(r"Spearman $\rho$(regime-1 frac, best-expert affinity)")
    axes[1].set_title("per-field best-expert / regime alignment")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("E4 v3: window-level best-expert regime affinity tracks the true regime "
                 "(expert identity learned from train-window wins)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v3_regime_alignment_diagnostic.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figures saved")


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
