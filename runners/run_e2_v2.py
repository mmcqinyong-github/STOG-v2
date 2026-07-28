"""E2-v2: condition-number / difference-branch redesign.

Old E2 flaw: alpha never entered data generation, so alpha strata had zero
variance. Here each field uses alpha_filter=True, alpha_pure=True (pure
1/f^alpha colored field, generator seed fixed at 42 -> only alpha varies).

Per field (T=5000, V=8, H=24, lookback L=24):
  - kappa(Sigma_x) and kappa(Sigma_dx) per variable, kappa_ratio =
    log(mean kappa_x / mean kappa_dx)
  - train M52 (DLinear, difference/decomposition branch) and M03 (RLinear,
    linear baseline), 3 train seeds averaged
  - diff benefit (prescribed) = MSE(M03) - MSE(M52)
  - mechanism ablation (architecture-controlled): same M03 trained in the
    difference domain (input dx, predict dy, reintegrate); benefit_delta =
    MSE(M03 raw) - MSE(M03 delta)

Tests:
  - Spearman(kappa_ratio, benefit), pre-registered >= 0.5
  - monotonicity: per-variable kappa_ratio at alpha=2.0 vs alpha=0.5
    (Mann-Whitney)

Outputs:
  results/e2_v2/e2v2_alpha_scan.csv   (per-alpha rows + per-variable kappas)
  results/e2_v2/e2v2_summary.csv
  results/figures/e2_v2_kappa_ratio_vs_diff_benefit.png
Checkpointed per alpha: re-running skips completed alphas.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

ALPHAS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
GEN_SEED = 42
TRAIN_SEEDS = [42, 2021, 3407]
OUT_DIR = "./results/e2_v2"
CKPT = f"{OUT_DIR}/e2v2_alpha_scan.csv"
L = 24
V = 8


class MockDM:
    def __init__(self, d):
        self.windows = d


def kappa_stats(X):
    kx, kdx = [], []
    for v in range(V):
        s = X[:, v]
        w = np.array([s[i:i + L] for i in range(0, len(s) - L, 5)])
        kx.append(np.linalg.cond(np.cov(w.T)))
        d = np.diff(s)
        wd = np.array([d[i:i + L] for i in range(0, len(d) - L, 5)])
        kdx.append(np.linalg.cond(np.cov(wd.T)))
    kx, kdx = np.array(kx), np.array(kdx)
    return kx, kdx, float(np.log(kx.mean() / kdx.mean()))


def train_avg(eid, d_in, dm, device_check=None):
    mses = []
    for ts in TRAIN_SEEDS:
        set_seed(ts)
        ex = get_expert(eid, d_in, hidden=128, drop=0.1)
        tr = UnifiedTrainer({"max_epochs": 5, "patience": 2,
                             "batch_size": 256, "lr": 1e-4})
        mses.append(tr.train_expert(ex, dm)["test_mse"])
    return float(np.mean(mses))


def diff_domain_mse(eid, data):
    """Train eid in the difference domain; return y-domain test MSE."""
    def d_inp(t):
        a = t.numpy().reshape(-1, L, V)
        return torch.tensor(np.diff(a, axis=1).reshape(len(t), -1),
                            dtype=torch.float32)

    def d_tgt(inp, tgt):
        a = inp.numpy().reshape(-1, L, V)
        y0 = torch.tensor(a[:, -1, 0], dtype=torch.float32).unsqueeze(1)
        return torch.diff(tgt, dim=1, prepend=y0)

    dm = MockDM({})
    for split in ["train", "val", "test"]:
        dm.windows[split] = d_inp(data[f"{split}_inp"])
        dm.windows[f"{split}_tgt"] = d_tgt(data[f"{split}_inp"], data[f"{split}_tgt"])
    mses_y = []
    for ts in TRAIN_SEEDS:
        set_seed(ts)
        ex = get_expert(eid, dm.windows["train"].shape[1], hidden=128, drop=0.1)
        tr = UnifiedTrainer({"max_epochs": 5, "patience": 2,
                             "batch_size": 256, "lr": 1e-4})
        tr.train_expert(ex, dm)
        ex.eval()
        with torch.no_grad():
            dyhat = ex(dm.windows["test"].to(tr.device)).cpu().numpy()
        y0 = data["test_inp"].numpy().reshape(-1, L, V)[:, -1, 0]
        yhat = y0[:, None] + np.cumsum(dyhat, axis=1)
        ytrue = data["test_tgt"].numpy()
        mses_y.append(float(((yhat - ytrue) ** 2).mean()))
    return float(np.mean(mses_y))


def run_alpha(alpha):
    cfg = SynthConfig(T=5000, V=V, H=24, alpha=alpha, seed=GEN_SEED,
                      alpha_filter=True, alpha_pure=True)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    X = data["X"].numpy()
    kx, kdx, kratio = kappa_stats(X)
    dm = MockDM({"train": data["train_inp"], "train_tgt": data["train_tgt"],
                 "val": data["val_inp"], "val_tgt": data["val_tgt"],
                 "test": data["test_inp"], "test_tgt": data["test_tgt"]})
    d_in = data["train_inp"].shape[1]
    mse_m03 = train_avg("M03", d_in, dm)
    mse_m52 = train_avg("M52", d_in, dm)
    mse_m03_diff = diff_domain_mse("M03", data)
    rec = {
        "alpha": alpha, "gen_seed": GEN_SEED,
        "kappa_x_mean": float(kx.mean()), "kappa_dx_mean": float(kdx.mean()),
        "kappa_ratio": kratio,
        **{f"kappa_x_v{v}": float(kx[v]) for v in range(V)},
        **{f"kappa_dx_v{v}": float(kdx[v]) for v in range(V)},
        "mse_M03": mse_m03, "mse_M52": mse_m52,
        "benefit_m52": mse_m03 - mse_m52,
        "mse_M03_diffdomain": mse_m03_diff,
        "benefit_delta": mse_m03 - mse_m03_diff,
    }
    print(f"[e2v2] alpha={alpha}: kappa_ratio={kratio:.3f} "
          f"benefit_m52={rec['benefit_m52']:+.4f} benefit_delta={rec['benefit_delta']:+.4f}",
          flush=True)
    return rec


def main():
    ensure_dir(OUT_DIR)
    done = pd.read_csv(CKPT) if os.path.exists(CKPT) else pd.DataFrame()
    done_alphas = set(done.alpha) if len(done) else set()
    for alpha in ALPHAS:
        if alpha in done_alphas:
            continue
        rec = run_alpha(alpha)
        done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
        done.to_csv(CKPT, index=False)
    df = pd.read_csv(CKPT).sort_values("alpha").reset_index(drop=True)

    # ---- tests ----
    kr = df["kappa_ratio"].values
    b52 = df["benefit_m52"].values
    bd = df["benefit_delta"].values
    rho52, p52 = spearmanr(kr, b52)
    rhod, pd_ = spearmanr(kr, bd)
    rho_alpha_d, pa_d = spearmanr(df["alpha"].values, bd)
    # monotonicity: per-variable kappa ratio at alpha=2.0 vs alpha=0.5
    row05 = df[df.alpha == 0.5].iloc[0]
    row20 = df[df.alpha == 2.0].iloc[0]
    kr05 = np.log(np.array([row05[f"kappa_x_v{v}"] for v in range(V)]) /
                  np.array([row05[f"kappa_dx_v{v}"] for v in range(V)]))
    kr20 = np.log(np.array([row20[f"kappa_x_v{v}"] for v in range(V)]) /
                  np.array([row20[f"kappa_dx_v{v}"] for v in range(V)]))
    u, pu = mannwhitneyu(kr20, kr05, alternative="greater")
    summary = pd.DataFrame([{
        "spearman_kratio_vs_benefit_m52": rho52, "pvalue_m52": p52,
        "spearman_kratio_vs_benefit_delta": rhod, "pvalue_delta": pd_,
        "spearman_alpha_vs_benefit_delta": rho_alpha_d, "pvalue_alpha_delta": pa_d,
        "kratio_alpha0.5_mean": float(kr05.mean()),
        "kratio_alpha2.0_mean": float(kr20.mean()),
        "mannwhitney_alpha2.0_gt_0.5_p": pu,
        "preregistered_threshold": 0.5,
        "m52_contrast_pass": bool(rho52 >= 0.5),
        "delta_ablation_pass": bool(rhod >= 0.5),
        "monotonicity_pass": bool(pu < 0.05),
    }])
    summary.to_csv(f"{OUT_DIR}/e2v2_summary.csv", index=False)
    print("\n=== E2-v2 summary ===")
    print(summary.T.to_string())

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(kr, b52, c=df["alpha"], cmap="viridis", s=60)
    for _, r in df.iterrows():
        ax.annotate(f"a={r.alpha}", (r.kappa_ratio, r.benefit_m52),
                    fontsize=8, alpha=0.7)
    ax.set_xlabel("kappa_ratio = log(kappa_x / kappa_dx)")
    ax.set_ylabel("diff benefit = MSE(M03) - MSE(M52)")
    ax.set_title(f"M52-vs-M03 contrast (Spearman={rho52:.2f}, p={p52:.3f})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax = axes[1]
    ax.scatter(kr, bd, c=df["alpha"], cmap="viridis", s=60)
    for _, r in df.iterrows():
        ax.annotate(f"a={r.alpha}", (r.kappa_ratio, r.benefit_delta),
                    fontsize=8, alpha=0.7)
    ax.set_xlabel("kappa_ratio = log(kappa_x / kappa_dx)")
    ax.set_ylabel("benefit_delta = MSE(M03 raw) - MSE(M03 delta-domain)")
    ax.set_title(f"delta-domain ablation (Spearman={rhod:.2f}, p={pd_:.3f})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    ensure_dir("./results/figures")
    fig.savefig("./results/figures/e2_v2_kappa_ratio_vs_diff_benefit.png",
                dpi=150, bbox_inches="tight")
    print("Saved results/figures/e2_v2_kappa_ratio_vs_diff_benefit.png")


if __name__ == "__main__":
    main()
