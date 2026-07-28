"""E5 figures: fusion CRPS bar, manifold family scatter, quantile fan example."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.common import ensure_dir

ensure_dir("./results/figures")
sns.set_style("whitegrid")

EXPERTS_12 = ["M03", "M52", "M47", "M63", "M17", "M14",
              "M50", "M18", "M31", "M55", "M233", "M89"]
TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
FUSION_ORDER = ["F1_Vincentization", "F2_LinearPool", "F3_MedianPool",
                "F4_OutputWeighted", "OracleBestSingle"]


# ---------- Fig 1: CRPS comparison bar ----------
df = pd.read_csv("./results/e5/e5_fusion_comparison.csv")
q = df[df.fusion.isin(FUSION_ORDER)].copy()
q["fusion"] = pd.Categorical(q["fusion"], FUSION_ORDER, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
sns.barplot(data=q, x="fusion", y="crps", errorbar="sd", ax=axes[0],
            color="#4C9BD6", edgecolor="black")
axes[0].set_title("E5 fusion geometry: test CRPS (5-quantile approx.)\nmean ± sd over NP/PJM/DE × 3 seeds")
axes[0].set_xlabel(""); axes[0].set_ylabel("CRPS ↓")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(data=q, x="fusion", y="mse", errorbar="sd", ax=axes[1],
            color="#7BBF7B", edgecolor="black")
f5 = df[df.fusion == "F5_HiddenRidge"]
axes[1].bar(["F5_HiddenRidge"], [f5.mse.mean()], yerr=[f5.mse.std()],
            color="#E8A33D", edgecolor="black", capsize=4)
axes[1].set_title("Point MSE of median (F5: hidden ridge fusion,\npoint-only, lightweight approximation)")
axes[1].set_xlabel(""); axes[1].set_ylabel("MSE ↓")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("./results/figures/e5_fusion_crps_comparison_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: manifold scatter ----------
z = np.load("./results/e5/e5_embedding_coords.npz")
coords, fam, mkt, exp = z["coords"], z["family"], z["market"], z["expert"]
dfc = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                    "family": fam, "market": mkt, "expert": exp})
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.scatterplot(data=dfc, x="x", y="y", hue="family", style="market",
                s=70, ax=ax, alpha=0.85)
# label experts once (first market occurrence)
for e in pd.unique(exp):
    sub = dfc[dfc.expert == e].iloc[0]
    ax.annotate(e, (sub.x, sub.y), fontsize=7, alpha=0.75,
                xytext=(3, 3), textcoords="offset points")
ax.set_title("E5 expert manifold: PHATE 2D embedding of inter-expert prediction distances\n"
             "(19 experts × 5 markets × 3 seeds, GPA-aligned; color = coarse family)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("./results/figures/e5_manifold_embedding_family_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3: quantile fan example ----------
market, seed = "DE", 2021
meta = np.load(f"./results/preds/meta_{market}_{seed}.npz")
yt = meta["test_true"]
# pick a high-volatility window for visual interest
vol = yt.std(axis=1)
widx = int(np.argsort(vol)[-12])
Qt = np.stack([np.load(f"./results/preds_quantile/{market}_{e}_{seed}.npz")["test_quant"][widx]
               for e in EXPERTS_12])  # (E,24,5)
f1 = Qt.mean(axis=0)  # (24,5)
t = np.arange(24)
fig, ax = plt.subplots(figsize=(10, 5.5))
for e in range(len(EXPERTS_12)):
    ax.plot(t, Qt[e, :, 2], color="gray", alpha=0.35, lw=0.9)
ax.fill_between(t, f1[:, 0], f1[:, 4], color="#4C9BD6", alpha=0.25, label="F1 10–90% band")
ax.fill_between(t, f1[:, 1], f1[:, 3], color="#4C9BD6", alpha=0.40, label="F1 25–75% band")
ax.plot(t, f1[:, 2], color="#145A8D", lw=2.2, label="F1 median (W2/Vincentization)")
ax.plot(t, yt[widx], color="black", lw=2.0, ls="--", label="observed")
ax.set_title(f"E5 quantile fan example — {market} market, test window #{widx} (24h ahead)\n"
             f"grey lines: 12 expert medians; bands: W2-fused quantiles")
ax.set_xlabel("hour ahead"); ax.set_ylabel("price")
ax.legend()
fig.tight_layout()
fig.savefig("./results/figures/e5_quantile_fan_example.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("figures saved:", os.listdir("./results/figures")[-3:])
