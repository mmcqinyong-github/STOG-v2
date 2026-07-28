"""Generate Nature-style figures for E8, E9, E10 full results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
})

os.makedirs("./results/figures", exist_ok=True)

# ===================== E8 =====================
print("Loading E8...")
df8 = pd.read_csv("./results/e8_stress_test.csv")
df8v = df8[df8['test_mse'] < 9000].copy()
stress_df = df8v[df8v['axis'] != 'baseline'].copy()
summary = stress_df.groupby(['axis', 'param'])['degradation'].agg(['mean', 'std']).reset_index()

# E8-1: Faceted bar
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
axes_info = [
    ('missingness', ['0.1_mcar','0.25_mcar','0.375_block','0.5_block'],
     ['MCAR 10%','MCAR 25%','Block 37.5%','Block 50%'], '#E63946', 'Missingness'),
    ('lookback', ['L_96','L_168','L_336','L_720'],
     ['L=96','L=168','L=336','L=720'], '#2A9D8F', 'Lookback'),
    ('corruption', ['noise_0.25','noise_0.5','noise_1.0','cov_missing_0.25','cov_missing_0.5'],
     ['Noise 0.25','Noise 0.5','Noise 1.0','CovMiss 0.25','CovMiss 0.5'], '#457B9D', 'Corruption'),
]
for ax, (axis, params, labels, color, title) in zip(axes, axes_info):
    sub = summary[summary['axis']==axis].set_index('param').reindex(params).reset_index()
    x = np.arange(len(sub))
    ax.bar(x, sub['mean'], 0.6, yerr=sub['std'], capsize=3, color=color,
           edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Mean Degradation' if axis=='missingness' else '')
fig.suptitle('E8: Three-Axis Stress Degradation', fontweight='bold', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('./results/figures/e8_full_degradation_faceted_bar.png')
plt.close(fig)
print("  Saved e8_full_degradation_faceted_bar.png")

# E8-2: Market × Axis heatmap
fig, ax = plt.subplots(figsize=(6, 4))
heatmap_data = stress_df.groupby(['market','axis'])['degradation'].median().unstack()
heatmap_data = heatmap_data[['missingness','lookback','corruption']]
im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto', vmin=-0.1, vmax=1.0)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(['Missingness','Lookback','Corruption'])
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.iloc[i,j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                color='white' if val > 0.5 else 'black')
ax.set_title('E8: Median Degradation by Market × Stress Axis', fontweight='bold', fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Median Degradation')
fig.tight_layout()
fig.savefig('./results/figures/e8_full_market_axis_heatmap.png')
plt.close(fig)
print("  Saved e8_full_market_axis_heatmap.png")

# E8-3: Expert robustness ranking
fig, ax = plt.subplots(figsize=(7, 6))
exp_robust = stress_df.groupby('expert_id')['degradation'].mean().sort_values()
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(exp_robust)))
ax.barh(range(len(exp_robust)), exp_robust.values, color=colors, edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(exp_robust)))
ax.set_yticklabels(exp_robust.index, fontsize=7)
ax.set_xlabel('Mean Degradation Rate')
ax.set_title('E8: Expert Robustness Ranking (Lower = More Robust)', fontweight='bold', fontsize=11)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.savefig('./results/figures/e8_full_expert_robustness_ranking.png')
plt.close(fig)
print("  Saved e8_full_expert_robustness_ranking.png")

# E8-4: Degradation violin
fig, ax = plt.subplots(figsize=(6, 4))
parts = ax.violinplot([stress_df[stress_df['axis']==a]['degradation'].clip(-1,5).values
                        for a in ['missingness','lookback','corruption']],
                       positions=[1,2,3], showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], ['#E63946','#2A9D8F','#457B9D']):
    pc.set_facecolor(color); pc.set_alpha(0.6); pc.set_edgecolor('black')
ax.set_xticks([1,2,3]); ax.set_xticklabels(['Missingness','Lookback','Corruption'])
ax.set_ylabel('Degradation Rate (clipped at 5)')
ax.set_title('E8: Degradation Distribution by Stress Axis', fontweight='bold', fontsize=11)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.savefig('./results/figures/e8_full_degradation_violin.png')
plt.close(fig)
print("  Saved e8_full_degradation_violin.png")

# ===================== E9 =====================
print("Loading E9...")
df9 = pd.read_csv("./results/e9_incremental.csv")

# E9-1: Cumulative regret curves
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for idx, market in enumerate(['NP','DE']):
    ax = axes[idx]
    for strat, color, label in [('fixed','#E63946','Fixed Best'),
                                 ('hedge','#457B9D','Hedge'),
                                 ('ctx_hedge','#2A9D8F','Contextual Hedge')]:
        curves = []
        for seed in [2021,42,3407]:
            sub = df9[(df9['market']==market)&(df9['seed']==seed)&(df9['strategy']==strat)]
            if len(sub)>0: curves.append(sub['cum_regret'].values)
        if curves:
            max_len = max(len(c) for c in curves)
            padded = [np.pad(c, (0, max_len-len(c)), mode='edge') for c in curves]
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            months = np.arange(len(mean_curve))
            ax.plot(months, mean_curve, color=color, label=label, linewidth=1.5)
            ax.fill_between(months, mean_curve-std_curve, mean_curve+std_curve, color=color, alpha=0.15)
    ax.set_xlabel('Month'); ax.set_ylabel('Cumulative Regret')
    ax.set_title(f'{market} Market', fontweight='bold')
    ax.legend(frameon=False, fontsize=7)
    ax.set_xticks(range(12))
fig.suptitle('E9: Cumulative Regret over Streaming Batches', fontweight='bold', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('./results/figures/e9_full_cumulative_regret_curves.png')
plt.close(fig)
print("  Saved e9_full_cumulative_regret_curves.png")

# E9-2: Final regret bar
fig, ax = plt.subplots(figsize=(5, 4))
final_regret = df9.groupby(['market','strategy','seed'])['cum_regret'].last().reset_index()
summary9 = final_regret.groupby(['market','strategy'])['cum_regret'].agg(['mean','std']).reset_index()
x = np.arange(2); width = 0.25
for i, (strat, color, label) in enumerate([('fixed','#E63946','Fixed'),
                                            ('hedge','#457B9D','Hedge'),
                                            ('ctx_hedge','#2A9D8F','Ctx Hedge')]):
    vals = summary9[summary9['strategy']==strat]
    means = [vals[vals['market']=='NP']['mean'].values[0] if len(vals[vals['market']=='NP']) else 0,
             vals[vals['market']=='DE']['mean'].values[0] if len(vals[vals['market']=='DE']) else 0]
    stds = [vals[vals['market']=='NP']['std'].values[0] if len(vals[vals['market']=='NP']) else 0,
            vals[vals['market']=='DE']['std'].values[0] if len(vals[vals['market']=='DE']) else 0]
    ax.bar(x + i*width - width, means, width, yerr=stds, capsize=3,
           label=label, color=color, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(['NP','DE'])
ax.set_ylabel('Final Cumulative Regret')
ax.set_title('E9: Strategy Comparison by Market', fontweight='bold', fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('./results/figures/e9_full_strategy_comparison_bar.png')
plt.close(fig)
print("  Saved e9_full_strategy_comparison_bar.png")

# ===================== E10 =====================
print("Loading E10...")
df10 = pd.read_csv("./results/e10_operator_ate.csv")

# E10-1: ATE forest plot
fig, ax = plt.subplots(figsize=(7, 6))
summary10 = df10.groupby(['operator','base_model'])['ate'].agg(['mean','std','count']).reset_index()
summary10 = summary10.sort_values(['operator','mean'])
y_pos = 0
colors = {'diff':'#E63946','moment':'#F4A261','graph':'#2A9D8F','gate':'#457B9D'}
for op in ['diff','moment','graph','gate']:
    sub = summary10[summary10['operator']==op]
    for _, row in sub.iterrows():
        ax.errorbar(row['mean'], y_pos, xerr=row['std']*1.96/np.sqrt(row['count']),
                   fmt='o', color=colors[op], markersize=5, capsize=3, elinewidth=1)
        y_pos += 1
    y_pos += 0.5
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('ATE (ΔMSE = Treat − Control)')
ax.set_ylabel('Base Model × Operator')
ax.set_title('E10: Operator Transplant ATE Forest Plot', fontweight='bold', fontsize=11)
legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c], markersize=8, label=c) for c in colors]
ax.legend(handles=legend_elements, frameon=False, title='Operator')
fig.tight_layout()
fig.savefig('./results/figures/e10_full_ate_forest_plot.png')
plt.close(fig)
print("  Saved e10_full_ate_forest_plot.png")

# E10-2: ATE heatmap
fig, ax = plt.subplots(figsize=(5, 4))
hm = df10.groupby(['operator','base_model'])['ate'].mean().unstack()
im = ax.imshow(hm.values, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, fontsize=8)
ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
for i in range(len(hm.index)):
    for j in range(len(hm.columns)):
        val = hm.iloc[i,j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8,
                color='white' if abs(val)>25 else 'black')
ax.set_title('E10: Mean ATE (Operator × Base Model)', fontweight='bold', fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('ATE (ΔMSE)')
fig.tight_layout()
fig.savefig('./results/figures/e10_full_ate_heatmap.png')
plt.close(fig)
print("  Saved e10_full_ate_heatmap.png")

# E10-3: ATE boxplot
fig, ax = plt.subplots(figsize=(5, 4))
bp_data = [df10[df10['operator']==op]['ate'].clip(-100,50).values for op in ['diff','moment','graph','gate']]
bp = ax.boxplot(bp_data, tick_labels=['Diff','Moment','Graph','Gate'],
                patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], ['#E63946','#F4A261','#2A9D8F','#457B9D']):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_ylabel('ATE (ΔMSE)')
ax.set_title('E10: ATE Distribution by Operator', fontweight='bold', fontsize=11)
fig.tight_layout()
fig.savefig('./results/figures/e10_full_ate_boxplot.png')
plt.close(fig)
print("  Saved e10_full_ate_boxplot.png")

print("\nAll figures generated!")
