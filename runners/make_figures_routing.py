"""Generate routing experiment figures from results/e6_routing CSVs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/figures"
R = "./results/e6_routing"
os.makedirs(OUT, exist_ok=True)

main_df = pd.read_csv(f"{R}/routing_main.csv")
summ = pd.read_csv(f"{R}/routing_summary.csv")
abl = pd.read_csv(f"{R}/routing_ablation.csv")
spear = pd.read_csv(f"{R}/probe_rank_spearman.csv")

ORDER = ["B20_oracle", "R_full", "B1_best_single", "B5_static_top3",
         "B8_fforma_lite", "B4_val_weighted", "B3_avg_ensemble", "B7_random_top3"]
LABEL = {"B20_oracle": "Oracle (B20)", "R_full": "MetaMorph-Lite",
         "B1_best_single": "Best-single (B1)", "B5_static_top3": "Static Top3 (B5)",
         "B8_fforma_lite": "FFORMA-lite (B8)", "B4_val_weighted": "Val-weighted (B4)",
         "B3_avg_ensemble": "Avg ensemble (B3)", "B7_random_top3": "Random Top3 (B7)",
         "R_dyn_only": "w/o static prior", "R_K1": "K=1", "R_K5": "K=5",
         "R_top3_mean": "mean fusion", "R_ridge": "ridge scorer",
         "R_no_probe": "no-probe (B4)", "R_card_only": "card-only"}
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]

# 1) methods comparison bar (5-market mean MSE, normalized per market for readability)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
per = main_df.pivot_table(index="market", columns="method", values="test_mse")
per = per[[m for m in ORDER if m in per.columns]]
per.rename(columns=LABEL).plot(kind="bar", ax=axes[0])
axes[0].set_ylabel("Test MSE (mean over seeds)")
axes[0].set_title("Per-market test MSE by method")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(fontsize=7)
norm = per.div(per["B1_best_single"], axis=0)
norm.rename(columns=LABEL).plot(kind="bar", ax=axes[1])
axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
axes[1].set_ylabel("Test MSE / B1 (per market)")
axes[1].set_title("MSE relative to best-single")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/e6_routing_methods_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2) improvement vs B1 (%), per market + average
imp = (1 - per.div(per["B1_best_single"], axis=0)) * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(imp.columns))
for i, m in enumerate(MARKETS):
    ax.bar(x + i * 0.15, imp.loc[m], width=0.15, label=m)
avg = imp.mean(axis=0)
ax.plot(x + 0.3, avg, "kD--", label="5-market avg", zorder=5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x + 0.3)
ax.set_xticklabels([LABEL.get(c, c) for c in imp.columns], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Improvement vs B1 (%)  (positive = better)")
ax.set_title("Routing methods: relative improvement over best-single expert")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/e6_routing_vs_bestsingle_improvement.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) ablation bar
fig, ax = plt.subplots(figsize=(8, 4.5))
a = abl.pivot_table(index="market", columns="method", values="test_mse")
anorm = a.div(a["R_full"], axis=0).mean(axis=0).sort_values()
anorm.index = [LABEL.get(c, c) for c in anorm.index]
anorm.plot(kind="barh", ax=ax, color="steelblue")
ax.axvline(1.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Test MSE / R_full (5-market mean of per-market ratio)")
ax.set_title("MetaMorph-Lite ablations")
fig.tight_layout()
fig.savefig(f"{OUT}/e6_routing_ablation_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4) probe->rank spearman scatter
fig, ax = plt.subplots(figsize=(6, 5.5))
for m in MARKETS:
    s = spear[spear.market == m]
    ax.scatter(s["spearman_fforma"], s["spearman_probe_only"], label=m, s=45)
lim = [0, 1]
ax.plot(lim, lim, "k--", lw=0.8)
ax.axhline(0.25, color="r", ls=":", lw=0.8, label="target 0.25")
ax.set_xlabel("FFORMA-lite (B8) per-window rank Spearman")
ax.set_ylabel("Probe (5 feats) per-window rank Spearman")
ax.set_title("Probe-predicted expert ranking vs realized ranking")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/e6_routing_probe_rank_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("figures saved:", [f for f in os.listdir(OUT) if f.startswith("e6_routing")])
