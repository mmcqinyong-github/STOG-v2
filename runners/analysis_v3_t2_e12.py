#!/usr/bin/env python3
"""Analysis v3 / Task 2: quantify three previously qualitative E12 phase-diagram
claims.

  (a) Region purity      : kNN (k=5, LOO) majority-class proportion on the
                           (pc1,pc2) embedding, at expert and family level,
                           with a permutation null.
  (b) Boundary fidelity  : markets placed in the (alpha_hat, kappa_hat) plane
                           estimated from saved probe features (meta_*.npz);
                           theoretical separatrix alpha_hat=1 (the qualitative
                           smooth/rough boundary from the E1 grid alpha in
                           {0.5,1,2}) vs the measured decision boundary
                           (logistic regression on the 15 cells). Hit rate +
                           signed distances.
  (c) Vector-field smoothness: label-change rate vs pairwise distance curve,
                           exponential fit p(d) = a*exp(-d/lam) + c.

Caveat surfaced throughout: e12_phase_data.csv has only 15 cells
(5 markets x 3 seeds), 2 winning experts, and its embedding is PCA over
hand-crafted market profile features (see run_e12_analysis.py L34-93).
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

OUT = "results/analysis_v3"
os.makedirs(OUT, exist_ok=True)

ph = pd.read_csv("results/figures/e12_phase_data.csv")
X = ph[["pc1", "pc2"]].to_numpy()
labels = ph["best_expert"].to_numpy()
families = ph["expert_family"].to_numpy()
markets = ph["market"].to_numpy()
n = len(ph)
rng = np.random.default_rng(0)

# ---------------- (a) region purity: kNN k=5 LOO majority ----------------
D = squareform(pdist(X))

def knn_purity(y, k=5):
    props = []
    for i in range(n):
        nn = np.argsort(D[i])[1:k + 1]          # exclude self
        vals, cnts = np.unique(y[nn], return_counts=True)
        props.append(cnts.max() / k)
    return float(np.mean(props)), np.array(props)

def null_purity(y, k=5, B=2000):
    nulls = []
    for _ in range(B):
        nulls.append(knn_purity(rng.permutation(y), k)[0])
    return np.array(nulls)

pur_exp, pur_exp_i = knn_purity(labels)
pur_fam, pur_fam_i = knn_purity(families)
null_exp = null_purity(labels)
null_fam = null_purity(families)
p_exp = float((null_exp >= pur_exp).mean())
p_fam = float((null_fam >= pur_fam).mean())

# ---------------- (b) boundary fidelity in (alpha_hat, kappa_hat) ---------
rows = []
for m in sorted(set(markets)):
    a_hats, k_hats = [], []
    for seed in [2021, 42, 3407]:
        meta = np.load(f"results/preds/meta_{m}_{seed}.npz")
        fn = list(meta["feat_names"])
        a_hats.append(float(-meta["feat_test"][:, fn.index("spec_decay")].mean()))
        k_hats.append(float(meta["feat_test"][:, fn.index("kurt")].mean()))
    wins = labels[markets == m]
    best = pd.Series(wins).mode()[0]
    rows.append({"market": m, "alpha_hat": np.mean(a_hats),
                 "kappa_hat": np.mean(k_hats), "best_expert": best,
                 "n_seeds_best": int((wins == best).sum())})
mk = pd.DataFrame(rows)

# measured decision boundary: logistic regression on 15 cells in (a,k) space
cell_ak = mk.set_index("market").loc[markets][["alpha_hat", "kappa_hat"]].to_numpy()
y_bin = (labels == "M47").astype(int)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e4).fit(cell_ak, y_bin)
w = lr.coef_[0]; b = lr.intercept_[0]
norm = np.hypot(*w)
dist_measured = (cell_ak @ w + b) / norm          # signed distance, + => M47 side

# theoretical separatrix: alpha_hat = 1  (E1 grid midpoint claim)
dist_theory = mk.set_index("market").loc[markets]["alpha_hat"].to_numpy() - 1.0
# rule: rough spectra (alpha_hat > 1) -> decomposition M47; smooth -> attention M63
pred_theory = np.where(dist_theory > 0, "M47", "M63")
hit_theory = float((pred_theory == labels).mean())
pred_meas = np.where(dist_measured > 0, "M47", "M63")
hit_measured = float((pred_meas == labels).mean())

mk["dist_to_theory_a1"] = mk["alpha_hat"] - 1.0
md = pd.DataFrame({"market": markets, "best_expert": labels,
                   "signed_dist_measured": dist_measured,
                   "signed_dist_theory_a1": dist_theory})
mk_dist = md.groupby("market").agg(
    best_expert=("best_expert", lambda s: s.mode()[0]),
    mean_abs_dist_measured=("signed_dist_measured", lambda s: np.abs(s).mean()),
    mean_signed_dist_theory=("signed_dist_theory_a1", "mean")).reset_index()

# ---------------- (c) vector-field smoothness ----------------
def smoothness(y):
    iu = np.triu_indices(n, 1)
    d = D[iu]
    chg = (y[iu[0]] != y[iu[1]]).astype(float)
    edges = np.quantile(d, np.linspace(0, 1, 6))
    edges[0] = 0
    bins, rates, cnts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (d >= lo) & (d <= hi if hi == edges[-1] else d < hi)
        bins.append((lo + hi) / 2); rates.append(chg[msk].mean()); cnts.append(int(msk.sum()))
    bins, rates = np.array(bins), np.array(rates)
    def f(x, a, lam, c):
        return a * np.exp(-x / lam) + c
    try:
        popt, _ = curve_fit(f, bins, rates, p0=[rates[0], np.median(d), 0.2],
                            bounds=([0, 1e-6, 0], [1.5, 1e4, 1]), maxfev=20000)
        pred = f(bins, *popt)
        r2 = 1 - np.sum((rates - pred) ** 2) / max(np.sum((rates - rates.mean()) ** 2), 1e-12)
        lam = popt[1]
    except Exception:
        lam, r2 = np.nan, np.nan
    return bins, rates, cnts, lam, r2, float(chg.mean())

b_e, r_e, c_e, lam_e, r2_e, base_e = smoothness(labels)
b_f, r_f, c_f, lam_f, r2_f, base_f = smoothness(families)

# ---------------- write e12_quantified.csv ----------------
with open(os.path.join(OUT, "e12_quantified.csv"), "w") as f:
    f.write("# === (a) region purity (kNN k=5 LOO majority proportion) ===\n")
    pd.DataFrame({
        "level": ["expert", "family"],
        "purity": [pur_exp, pur_fam],
        "null_mean": [null_exp.mean(), null_fam.mean()],
        "null_std": [null_exp.std(), null_fam.std()],
        "p_value_perm2000": [p_exp, p_fam],
        "n_cells": [n, n],
    }).to_csv(f, index=False)
    f.write("\n# === per-cell purity (expert level) ===\n")
    pd.DataFrame({"market": markets, "best_expert": labels,
                  "family": families, "knn5_majority_prop": pur_exp_i,
                  "knn5_majority_prop_family": pur_fam_i}).to_csv(f, index=False)
    f.write("\n# === (b) markets in (alpha_hat,kappa_hat) plane ===\n")
    mk.to_csv(f, index=False)
    f.write(f"\n# theory alpha_hat=1 hit_rate={hit_theory:.3f}; "
            f"measured logistic boundary hit_rate={hit_measured:.3f}\n")
    f.write("\n# === (b) per-market signed distances to boundaries ===\n")
    mk_dist.to_csv(f, index=False)
    f.write("\n# === (c) label-change rate vs distance (bins) ===\n")
    pd.DataFrame({
        "bin_center": np.r_[b_e, b_f],
        "level": ["expert"] * len(b_e) + ["family"] * len(b_f),
        "change_rate": np.r_[r_e, r_f],
        "n_pairs": np.r_[c_e, c_f],
    }).to_csv(f, index=False)
    f.write(f"\n# exp fit: expert lam={lam_e:.1f} R2={r2_e:.3f} "
            f"base_rate={base_e:.3f}; family lam={lam_f:.1f} R2={r2_f:.3f} "
            f"base_rate={base_f:.3f}\n")

print(f"(a) purity expert={pur_exp:.3f} (null {null_exp.mean():.3f}±{null_exp.std():.3f}, p={p_exp:.3f})")
print(f"    purity family={pur_fam:.3f} (null {null_fam.mean():.3f}±{null_fam.std():.3f}, p={p_fam:.3f})")
print(mk.to_string(index=False))
print(f"(b) theory a_hat=1 hit={hit_theory:.3f}; measured logistic hit={hit_measured:.3f}")
print(mk_dist.to_string(index=False))
print(f"(c) expert: lam={lam_e:.1f} R2={r2_e:.3f} rates={np.round(r_e,3)}")
print(f"    family: lam={lam_f:.1f} R2={r2_f:.3f} rates={np.round(r_f,3)}")
