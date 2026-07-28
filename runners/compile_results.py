"""Compile all experiment results into summary statistics."""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress

results_dir = "./results"
summary = {}

# E1: Synthetic Spectral
if os.path.exists(f"{results_dir}/e1_synthetic_spectral.csv"):
    df = pd.read_csv(f"{results_dir}/e1_synthetic_spectral.csv")
    summary["E1"] = {
        "n_configs": len(df),
        "mean_rho": df["spearman_rho"].mean(),
        "std_rho": df["spearman_rho"].std(),
        "mean_pvalue": df["pvalue"].mean(),
        "rho_by_alpha": df.groupby("alpha")["spearman_rho"].mean().to_dict(),
        "rho_by_spatial": df.groupby("spatial_type")["spearman_rho"].mean().to_dict(),
    }

# E2: Condition Number
if os.path.exists(f"{results_dir}/e2_condition_number.csv"):
    df = pd.read_csv(f"{results_dir}/e2_condition_number.csv")
    summary["E2"] = {
        "n_samples": len(df),
        "mean_kappa_x": df["kappa_x"].mean(),
        "mean_kappa_dx": df["kappa_dx"].mean(),
        "mean_kappa_ratio": df["kappa_ratio"].mean(),
        "ratio_by_alpha": df.groupby("alpha")["kappa_ratio"].mean().to_dict(),
    }

# E3: Heavy-tail
if os.path.exists(f"{results_dir}/e3_heavytail.csv"):
    df = pd.read_csv(f"{results_dir}/e3_heavytail.csv")
    deg_summary = df.groupby(["group", "spike_rate"])["degradation"].agg(["mean", "std"]).reset_index()
    summary["E3"] = {
        "n_runs": len(df),
        "degradation_summary": deg_summary.to_dict('records'),
        "robust_vs_raw": deg_summary.pivot(index="spike_rate", columns="group", values="mean").to_dict(),
    }

# E4: Regime Overlap
if os.path.exists(f"{results_dir}/e4_regime_overlap.csv"):
    df = pd.read_csv(f"{results_dir}/e4_regime_overlap.csv")
    if not df["benefit"].isna().all() and len(df) > 2:
        slope, intercept, r_value, p_value, std_err = linregress(df["one_minus_delta"], df["benefit"])
        summary["E4"] = {
            "n_samples": len(df),
            "r_squared": r_value**2,
            "slope": slope,
            "pvalue": p_value,
            "mean_benefit": df["benefit"].mean(),
            "benefit_by_delta": df.groupby("delta")["benefit"].mean().to_dict(),
        }
    else:
        summary["E4"] = {"note": "Insufficient variation for regression"}

# E6: EPF Main
if os.path.exists(f"{results_dir}/e6_epf_main.csv"):
    df = pd.read_csv(f"{results_dir}/e6_epf_main.csv")
    df_valid = df[df["test_mse"] < 900]
    summary["E6"] = {
        "n_total_runs": len(df),
        "n_successful": len(df_valid),
        "n_failed": len(df) - len(df_valid),
        "markets": df["market"].unique().tolist(),
        "experts": df["expert_id"].unique().tolist(),
        "seeds": df["seed"].unique().tolist(),
    }
    # Per-market best expert
    market_best = df_valid.loc[df_valid.groupby("market")["test_mse"].idxmin()][["market", "expert_id", "test_mse", "test_mae"]]
    summary["E6"]["market_best"] = market_best.to_dict('records')
    # Overall ranking
    overall = df_valid.groupby("expert_id")["test_mse"].mean().sort_values()
    summary["E6"]["overall_ranking"] = overall.to_dict()
    # Per-market ranking table
    rank_table = df_valid.groupby(["market", "expert_id"])["test_mse"].mean().unstack(level=0)
    summary["E6"]["rank_table_csv"] = rank_table.to_csv()

# Print summary
for exp, data in summary.items():
    print(f"\n{'='*40}")
    print(f"{exp} Summary")
    print(f"{'='*40}")
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{len(v)} items]")
        elif isinstance(v, dict) and len(v) > 5:
            print(f"  {k}: { {k2: round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in list(v.items())[:5]} }...")
        else:
            print(f"  {k}: {v}")

# Save to JSON for report generation
import json
with open("./results/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSummary saved to ./results/summary.json")
