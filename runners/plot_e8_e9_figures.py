"""Generate figures for E8 and E9."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Nature style
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.4,
})

OUTDIR = "./results/figures"
os.makedirs(OUTDIR, exist_ok=True)
C_GRAD = sns.color_palette("viridis", 12)
C_WARM = sns.color_palette("plasma", 8)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {path}")


# E8 Figures
print("=== E8 Figures ===")
df8 = pd.read_csv("./results/e8_stress_test.csv")

# Fig: Degradation by axis - grouped bar
fig, ax = plt.subplots(figsize=(8, 5))
summary = df8.groupby(['axis', 'param', 'expert_id'])['degradation'].mean().reset_index()
axes_names = summary['axis'].unique()
x = np.arange(len(axes_names))
width = 0.25
for idx, eid in enumerate(summary['expert_id'].unique()):
    sub = summary[summary['expert_id'] == eid]
    vals = []
    for ax_name in axes_names:
        v = sub[sub['axis'] == ax_name]['degradation'].mean()
        vals.append(v)
    ax.bar(x + idx*width, vals, width, label=eid, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(x + width)
ax.set_xticklabels(['Missingness', 'Noise', 'Truncation'])
ax.set_ylabel('Degradation Rate', fontsize=11)
ax.set_title('E8: Expert Degradation Under Three Stress Axes', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='gray', title='Expert')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='gray', lw=0.8, alpha=0.5)
save(fig, 'e8_fig14_stress_degradation_grouped_bar.png')

# Fig: Detailed stress parameter radar-like dot plot (novel)
fig, ax = plt.subplots(figsize=(8, 5))
params = df8['param'].unique()
x = np.arange(len(params))
for idx, eid in enumerate(df8['expert_id'].unique()):
    sub = df8[df8['expert_id'] == eid]
    means = [sub[sub['param']==p]['degradation'].mean() for p in params]
    ax.plot(x, means, marker='o', label=eid, color=C_GRAD[idx*2+2], lw=2, markersize=7,
            markeredgecolor='white', markeredgewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(params, rotation=30, ha='right')
ax.set_ylabel('Degradation Rate', fontsize=11)
ax.set_title('E8: Expert Degradation Curves by Stress Parameter', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='gray')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='gray', lw=0.8, alpha=0.5)
save(fig, 'e8_fig15_stress_degradation_trajectory.png')


# E9 Figures
print("\n=== E9 Figures ===")
df9 = pd.read_csv("./results/e9_incremental.csv")

# Fig: Cumulative regret curves
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df9['batch'], df9['ctx_cum'], marker='s', label='Contextual Hedge (probe prior)',
        color='#3498DB', lw=2.5, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.plot(df9['batch'], df9['hedge_cum'], marker='o', label='Post-hoc Hedge (no prior)',
        color='#E67E22', lw=2.5, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.plot(df9['batch'], df9['static_cum'], marker='^', label='Static Best-Single',
        color='#95A5A6', lw=2.5, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel('Batch (Time)', fontsize=11)
ax.set_ylabel('Cumulative Regret', fontsize=11)
ax.set_title('E9: Cumulative Regret of Incremental Strategies', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='gray', loc='upper left')
ax.grid(True, alpha=0.3)
# Add annotation
final_ctx = df9['ctx_cum'].iloc[-1]
final_hedge = df9['hedge_cum'].iloc[-1]
final_static = df9['static_cum'].iloc[-1]
ax.annotate(f'Ctx: {final_ctx:.2f}', xy=(df9['batch'].iloc[-1], final_ctx),
            xytext=(df9['batch'].iloc[-1]-3, final_ctx+0.5),
            fontsize=9, fontweight='bold', color='#3498DB',
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1))
ax.annotate(f'Hedge: {final_hedge:.2f}', xy=(df9['batch'].iloc[-1], final_hedge),
            xytext=(df9['batch'].iloc[-1]-3, final_hedge+0.5),
            fontsize=9, fontweight='bold', color='#E67E22',
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1))
save(fig, 'e9_fig16_cumulative_regret_comparison.png')

# Fig: Per-batch regret (novel)
fig, ax = plt.subplots(figsize=(7, 5))
width = 0.25
x = np.arange(len(df9))
ax.bar(x - width, df9['ctx_regret'], width, label='Contextual Hedge', color='#3498DB', alpha=0.8, edgecolor='white')
ax.bar(x, df9['hedge_regret'], width, label='Post-hoc Hedge', color='#E67E22', alpha=0.8, edgecolor='white')
ax.bar(x + width, df9['static_regret'], width, label='Static Best', color='#95A5A6', alpha=0.8, edgecolor='white')
ax.set_xlabel('Batch', fontsize=11)
ax.set_ylabel('Per-Batch Regret', fontsize=11)
ax.set_title('E9: Per-Batch Regret Distribution', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='gray')
ax.grid(True, alpha=0.3, axis='y')
save(fig, 'e9_fig17_perbatch_regret_distribution.png')

print("\n✅ E8/E9 figures generated!")
