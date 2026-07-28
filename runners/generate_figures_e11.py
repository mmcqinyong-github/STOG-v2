import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
})

# Load data
df = pd.read_csv("./results/e11_crossdomain.csv")
corr = pd.read_csv("./results/e11_domain_rank_correlation.csv", index_col=0)
lodo = pd.read_csv("./results/e11_lodo_spearman.csv")

# Fig 1: Cross-domain rank correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='RdYlBu_r', aspect='auto', vmin=-0.3, vmax=1.0)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(val) > 0.6 else 'black')
ax.set_title('E11: Cross-Domain Expert Rank Correlation', fontweight='bold', fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman ρ')
fig.tight_layout()
fig.savefig('./results/figures/e11_full_domain_correlation_heatmap.png')
plt.close(fig)
print("Saved e11_full_domain_correlation_heatmap.png")

# Fig 2: LODO bar chart
fig, ax = plt.subplots(figsize=(7, 4))
summary = lodo.groupby('held_out')['spearman_rho'].agg(['mean', 'std']).reset_index()
x = np.arange(len(summary))
ax.bar(x, summary['mean'], yerr=summary['std'], capsize=4, color='#457B9D',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axhline(0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='ρ=0.5 threshold')
ax.set_xticks(x)
ax.set_xticklabels(summary['held_out'], rotation=45, ha='right')
ax.set_ylabel('LODO Spearman ρ')
ax.set_title('E11: Leave-One-Domain-Out Rank Generalization', fontweight='bold', fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('./results/figures/e11_full_lodo_bar.png')
plt.close(fig)
print("Saved e11_full_lodo_bar.png")

# Fig 3: Per-domain best expert (no universal champion)
fig, ax = plt.subplots(figsize=(8, 4))
domain_best = df.groupby(['domain', 'expert_id'])['test_mse'].mean().reset_index()
best_per_domain = domain_best.loc[domain_best.groupby('domain')['test_mse'].idxmin()]
domains = best_per_domain['domain'].values
experts = best_per_domain['expert_id'].values
colors = plt.cm.tab20(np.linspace(0, 1, len(domains)))
ax.bar(range(len(domains)), best_per_domain['test_mse'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(domains)))
ax.set_xticklabels(domains, rotation=45, ha='right')
for i, (d, e) in enumerate(zip(domains, experts)):
    ax.text(i, best_per_domain['test_mse'].iloc[i], e, ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_ylabel('Best Test MSE')
ax.set_title('E11: Best Expert Varies by Domain (No Universal Champion)', fontweight='bold', fontsize=11)
fig.tight_layout()
fig.savefig('./results/figures/e11_full_best_expert_per_domain.png')
plt.close(fig)
print("Saved e11_full_best_expert_per_domain.png")

print("All E11 figures generated!")
