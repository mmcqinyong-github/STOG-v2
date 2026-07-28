"""E10-v3: gate control-arm redesign + random-residual placebo (reviewer-driven).

Addresses two reviewer criticisms of E10-v2:
  (1) the v2 gate CTRL trained MLP weights on a zero input (capacity not
      equivalent). v3 adds G2 (frozen gate MLP, scalar-bias-only learnable)
      and G3 (single learnable scalar gate, no MLP).
  (2) no random-residual placebo arm: v3 adds R1 (frozen random projection
      residual branch + learnable scalar coefficient, param-matched to the
      gate MLP) and R2 (same branch, fully trainable) to separate
      "gate semantics help" from "any residual branch helps".

Design: 3 bases (M52 DLinear / M17 ModernTCN / M50 PatchTST) x 5 arms
(g1 gate-TREAT bridge / g2 / g3 / r1 / r2) x 5 markets x 3 seeds = 225 runs.
Training protocol identical to v2 (max_epochs=10, patience=3, lr=1e-4).

Checkpoint/resume: every finished run is appended to
results/e10_v3/e10v3_runs.csv; re-running skips rows already present.

Usage:
  python run_e10_v3.py                 # run pending experiments (time-budgeted)
  python run_e10_v3.py --time-budget 280
  python run_e10_v3.py --analyze       # summaries + v2 re-aggregation + figure
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, graft_gate_v3
from src.training.trainer import UnifiedTrainer

RESULTS_DIR = "./results/e10_v3"
RUNS_CSV = os.path.join(RESULTS_DIR, "e10v3_runs.csv")
V2_RUNS_CSV = "./results/e10_v2/e10v2_runs.csv"
FIG_DIR = "./results/figures"

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
BASE_MODELS = ["M52", "M17", "M50"]
ARMS = ["g1", "g2", "g3", "r1", "r2"]
ARM_LABELS = {
    "g1": "gate TREAT (bridge)",
    "g2": "gate CTRL-v3a (frozen MLP + scalar bias)",
    "g3": "gate CTRL-v3b (scalar gate)",
    "r1": "random residual placebo (frozen)",
    "r2": "random residual (trainable)",
}

TRAIN_CFG = {"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4}


# ----------------------------- Phase 1: runs -----------------------------

def pending_grid(done_keys):
    for market in MARKETS:
        for seed in SEEDS:
            for base in BASE_MODELS:
                for arm in ARMS:
                    if (market, seed, base, arm) not in done_keys:
                        yield market, seed, base, arm


def load_done_keys():
    if not os.path.exists(RUNS_CSV):
        return set()
    df = pd.read_csv(RUNS_CSV)
    return set(zip(df.market, df.seed, df.base_model, df.arm))


def run_phase(time_budget=280.0):
    ensure_dir(RESULTS_DIR)
    t_start = time.time()
    done = load_done_keys()
    todo = list(pending_grid(done))
    print(f"[e10v3] done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("[e10v3] all runs complete.")
        return

    from itertools import groupby
    for (market, seed), grp in groupby(todo, key=lambda t: (t[0], t[1])):
        if time.time() - t_start > time_budget:
            break
        grp = list(grp)
        set_seed(seed)
        dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
        dm.make_windows()
        dm.normalize()
        d_in = dm.windows["train"].shape[1]
        for (_, _, base_id, arm) in grp:
            if time.time() - t_start > time_budget:
                break
            cfg_seed = abs(hash(("e10v3", market, seed, base_id, arm))) % (2**31)
            set_seed(cfg_seed)
            try:
                base = get_expert(base_id, d_in, hidden=256, drop=0.1)
                model = graft_gate_v3(base, arm, n_vars=3, lookback=168, horizon=24)
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                trainer = UnifiedTrainer(TRAIN_CFG)
                res = trainer.train_expert(model, dm)
                row = {
                    "market": market, "seed": seed, "base_model": base_id,
                    "arm": arm, "arm_label": ARM_LABELS[arm],
                    "val_mse": res["val_mse"], "test_mse": res["test_mse"],
                    "test_mae": res["test_mae"], "epochs": res["epochs"],
                    "time_sec": res["time_sec"], "n_trainable": n_trainable,
                }
            except Exception as ex:
                err = " ".join(str(ex).split())[:200].replace(",", ";")
                row = {
                    "market": market, "seed": seed, "base_model": base_id,
                    "arm": arm, "arm_label": ARM_LABELS[arm],
                    "val_mse": np.nan, "test_mse": np.nan, "test_mae": np.nan,
                    "epochs": 0, "time_sec": 0.0, "n_trainable": -1,
                    "error": err,
                }
                print(f"  ERROR {market}/{seed}/{base_id}/{arm}: {err}", flush=True)
            print(f"  {market} s{seed} {base_id} {arm} "
                  f"test_mse={row['test_mse']:.3f} ({row['time_sec']:.1f}s)", flush=True)
            pd.DataFrame([row]).to_csv(
                RUNS_CSV, mode="a", header=not os.path.exists(RUNS_CSV), index=False)
            if "error" in row and "CUDA" in row["error"]:
                # CUDA context is poisoned after a device-side error; exit so the
                # next driver session resumes in a fresh process.
                print("[e10v3] fatal CUDA error -> exit for fresh-process resume", flush=True)
                sys.exit(3)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done2 = load_done_keys()
    print(f"[e10v3] session added {len(done2) - len(done)} runs; "
          f"total done={len(done2)}/225", flush=True)


# ----------------------------- Phase 2: analysis -----------------------------

def wilcoxon_safe(x, alternative="two-sided"):
    from scipy import stats as sstats
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    try:
        return float(sstats.wilcoxon(x, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(RESULTS_DIR)
    ensure_dir(FIG_DIR)
    df = pd.read_csv(RUNS_CSV).dropna(subset=["test_mse"])
    print(f"[analyze] valid v3 runs: {len(df)}")

    # ---- paired table: gate TREAT (g1) vs every control/placebo arm ----
    pv = df.pivot_table(index=["market", "seed", "base_model"],
                        columns="arm", values="test_mse").reset_index()
    comparisons = [("g2", "frozen-MLP CTRL"), ("g3", "scalar-gate CTRL"),
                   ("r1", "random-residual placebo (frozen)"),
                   ("r2", "random residual (trainable)")]
    rows = []
    for other, label in comparisons:
        sub = pv.dropna(subset=["g1", other]).copy()
        sub["ate_abs"] = sub["g1"] - sub[other]                 # <0: gate better
        sub["ate_rel"] = sub["ate_abs"] / sub[other]
        sub["dlog"] = np.log(sub["g1"]) - np.log(sub[other])
        # overall
        rows.append({
            "comparison": f"g1_vs_{other}", "other_arm_label": label,
            "scope": "ALL", "n_pairs": len(sub),
            "ate_abs_mean": sub["ate_abs"].mean(),
            "ate_rel_mean": sub["ate_rel"].mean(),
            "ate_rel_std": sub["ate_rel"].std(),
            "ci95_lo": sub["ate_rel"].mean() - 1.96 * sub["ate_rel"].std() / np.sqrt(len(sub)),
            "ci95_hi": sub["ate_rel"].mean() + 1.96 * sub["ate_rel"].std() / np.sqrt(len(sub)),
            "dlog_mean": sub["dlog"].mean(),
            "frac_gate_better": float((sub["ate_abs"] < 0).mean()),
            "wilcoxon_p_twosided": wilcoxon_safe(sub["ate_abs"]),
            "wilcoxon_p_gate_better": wilcoxon_safe(sub["ate_abs"], alternative="less"),
        })
        # per base model
        for b in BASE_MODELS:
            sb = sub[sub.base_model == b]
            if len(sb) < 5:
                continue
            rows.append({
                "comparison": f"g1_vs_{other}", "other_arm_label": label,
                "scope": b, "n_pairs": len(sb),
                "ate_abs_mean": sb["ate_abs"].mean(),
                "ate_rel_mean": sb["ate_rel"].mean(),
                "ate_rel_std": sb["ate_rel"].std(),
                "ci95_lo": sb["ate_rel"].mean() - 1.96 * sb["ate_rel"].std() / np.sqrt(len(sb)),
                "ci95_hi": sb["ate_rel"].mean() + 1.96 * sb["ate_rel"].std() / np.sqrt(len(sb)),
                "dlog_mean": sb["dlog"].mean(),
                "frac_gate_better": float((sb["ate_abs"] < 0).mean()),
                "wilcoxon_p_twosided": wilcoxon_safe(sb["ate_abs"]),
                "wilcoxon_p_gate_better": wilcoxon_safe(sb["ate_abs"], alternative="less"),
            })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "e10v3_gate_vs_placebo.csv"), index=False)

    # ---- key verdict: gate semantics vs any-residual-branch ----
    key = {}
    for cname in ["g1_vs_g2", "g1_vs_g3", "g1_vs_r1", "g1_vs_r2"]:
        r = comp_df[(comp_df.comparison == cname) & (comp_df.scope == "ALL")].iloc[0]
        key[cname] = {"ate_rel": r.ate_rel_mean, "p": r.wilcoxon_p_twosided,
                      "frac_gate_better": r.frac_gate_better}
    # gate semantics holds iff gate beats BOTH the frozen placebo (r1) and the
    # capacity-matched trainable residual (r2) significantly
    verdict_semantics = bool(key["g1_vs_r1"]["ate_rel"] < 0 and key["g1_vs_r1"]["p"] < 0.05
                             and key["g1_vs_r2"]["ate_rel"] < 0 and key["g1_vs_r2"]["p"] < 0.05)
    # any-residual story holds if gate is NOT better than the trainable random branch
    verdict_residual = bool(not (key["g1_vs_r2"]["ate_rel"] < 0 and key["g1_vs_r2"]["p"] < 0.05))

    # ---- per-arm global means for context ----
    arm_glob = df.groupby("arm").agg(
        test_mse_mean=("test_mse", "mean"), test_mse_std=("test_mse", "std"),
        n=("test_mse", "count"), n_trainable=("n_trainable", "mean")).reset_index()

    # ---- summary CSV ----
    summary_rows = []
    for _, r in arm_glob.iterrows():
        summary_rows.append({"section": "arm_global", "key": r.arm,
                             "value": r.test_mse_mean,
                             "detail": f"std={r.test_mse_std:.3f}, n={int(r.n)}, "
                                       f"trainable={int(r.n_trainable)}"})
    for _, r in comp_df[comp_df.scope == "ALL"].iterrows():
        summary_rows.append({"section": "gate_vs", "key": r.comparison,
                             "value": r.ate_rel_mean,
                             "detail": f"rel ATE (g1-other)/other; CI95=[{r.ci95_lo:.4f},{r.ci95_hi:.4f}], "
                                       f"p={r.wilcoxon_p_twosided:.4g}, "
                                       f"frac_gate_better={r.frac_gate_better:.2f}"})
    summary_rows.append({"section": "verdict", "key": "gate_semantics_holds",
                         "value": verdict_semantics,
                         "detail": "gate significantly better than BOTH r1 and r2"})
    summary_rows.append({"section": "verdict", "key": "any_residual_branch_story",
                         "value": verdict_residual,
                         "detail": "gate NOT significantly better than trainable random residual (r2)"})
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "e10v3_summary.csv"), index=False)

    # ---------------- Phase 3: v2 relative-scale re-aggregation ----------------
    v2 = pd.read_csv(V2_RUNS_CSV).dropna(subset=["test_mse"])
    pv2 = v2.pivot_table(index=["market", "seed", "base_model", "operator"],
                         columns="arm", values="test_mse").reset_index()
    pv2["rel_ate"] = (pv2["treat"] - pv2["ctrl"]) / pv2["ctrl"]
    pv2["dlog_mse"] = np.log(pv2["treat"]) - np.log(pv2["ctrl"])
    # market-level mean first (removes market-scale dominance), then average
    # across markets with equal weights; CI over n=5 market means.
    mkt = pv2.groupby(["operator", "market"]).agg(
        rel_ate=("rel_ate", "mean"), dlog=("dlog_mse", "mean")).reset_index()
    rel_rows = []
    for op in ["diff", "moment", "graph", "gate"]:
        s = mkt[mkt.operator == op]
        n = len(s)
        rel_rows.append({
            "operator": op, "n_markets": n,
            "rel_ate_mean": s["rel_ate"].mean(),
            "rel_ate_ci95_lo": s["rel_ate"].mean() - 1.96 * s["rel_ate"].std() / np.sqrt(n),
            "rel_ate_ci95_hi": s["rel_ate"].mean() + 1.96 * s["rel_ate"].std() / np.sqrt(n),
            "dlog_mean": s["dlog"].mean(),
            "dlog_ci95_lo": s["dlog"].mean() - 1.96 * s["dlog"].std() / np.sqrt(n),
            "dlog_ci95_hi": s["dlog"].mean() + 1.96 * s["dlog"].std() / np.sqrt(n),
            "frac_cells_negative": float((pv2[pv2.operator == op]["rel_ate"] < 0).mean()),
        })
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(os.path.join(RESULTS_DIR, "e10v2_relative_ate.csv"), index=False)

    # ---------------- forest figure ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot_df = comp_df[comp_df.scope == "ALL"].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = ["darkred" if (r.ate_rel_mean < 0 and r.wilcoxon_p_twosided < 0.05)
              else ("darkgreen" if (r.ate_rel_mean > 0 and r.wilcoxon_p_twosided < 0.05)
                    else "gray")
              for _, r in plot_df.iterrows()]
    ax.errorbar(plot_df["ate_rel_mean"], y,
                xerr=[plot_df["ate_rel_mean"] - plot_df["ci95_lo"],
                      plot_df["ci95_hi"] - plot_df["ate_rel_mean"]],
                fmt="none", capsize=5, ecolor="steelblue", lw=2)
    for yi, (_, r), c in zip(y, plot_df.iterrows(), colors):
        ax.plot(r.ate_rel_mean, yi, "o", color=c, ms=10, mec="black")
        ax.annotate(f"p={r.wilcoxon_p_twosided:.3g}",
                    (r.ci95_hi, yi), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, va="center")
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"gate TREAT vs {r.other_arm_label}" for _, r in plot_df.iterrows()],
                       fontsize=10)
    ax.set_xlabel("relative ATE = (MSE_gate - MSE_other) / MSE_other   (<0: gate better)")
    ax.set_title("E10-v3 gate vs redesigned controls / random placebos\n"
                 "(paired n=45; red = gate significantly better)", fontsize=11)
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e10_v3_gate_vs_placebo_forest.png"), dpi=150)
    plt.close(fig)

    print(comp_df[comp_df.scope == "ALL"].to_string(index=False))
    print(rel_df.to_string(index=False))
    print("verdict gate_semantics_holds:", verdict_semantics,
          "| any_residual_branch_story:", verdict_residual)
    return {"key": key, "verdict_semantics": verdict_semantics,
            "verdict_residual": verdict_residual, "rel": rel_df}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--time-budget", type=float, default=280.0)
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run_phase(time_budget=args.time_budget)
