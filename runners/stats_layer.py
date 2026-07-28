"""STOG-MetaMorph 统计层：对 results/*.csv 统一计算置信区间、显著性检验与 FDR 校正。

只读已有 CSV，输出到 results/stats/。纯数据分析，零训练。
运行: /c/code/T/Scripts/python.exe stats_layer.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260727)
N_BOOT = 10_000
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RESULTS, "stats")
os.makedirs(OUT, exist_ok=True)

FDR_FAMILY = []  # (comparison_id, raw_p)


def fisher_ci(r, n, alpha=0.05):
    """Spearman/Pearson 相关系数的 Fisher z 置信区间。"""
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def boot_ci_mean(x, n_boot=N_BOOT, alpha=0.05):
    """均值 percentile bootstrap CI（向量化）。"""
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def winsorized_mean(x, lo=0.01, hi=0.99):
    x = np.asarray(x, dtype=float)
    a, b = np.percentile(x, lo * 100), np.percentile(x, hi * 100)
    return float(np.clip(x, a, b).mean())


# ============================== E1 ==============================
def run_e1():
    df = pd.read_csv(os.path.join(RESULTS, "e1_synthetic_spectral.csv"))
    n_ranked = len([c for c in df.columns if c.startswith("mse_")])  # 被排序的专家数
    rows = []
    for _, r in df.iterrows():
        lo, hi = fisher_ci(r.spearman_rho, n_ranked)
        rows.append({
            "level": "config", "alpha": r.alpha, "spatial_type": r.spatial_type,
            "kappa": r.kappa, "seed": r.seed, "spearman_rho": r.spearman_rho,
            "ci_lo": lo, "ci_hi": hi, "p_value": r.pvalue, "sig_p05": bool(r.pvalue < 0.05),
        })
        FDR_FAMILY.append((f"E1|a={r.alpha}|{r.spatial_type}|k={r.kappa}|s={r.seed}", r.pvalue))
    share = float((df.pvalue < 0.05).mean())
    for a, sub in df.groupby("alpha"):
        lo, hi = boot_ci_mean(sub.spearman_rho.values)
        rows.append({
            "level": "alpha_summary", "alpha": a, "spatial_type": "", "kappa": "", "seed": "",
            "spearman_rho": sub.spearman_rho.mean(), "ci_lo": lo, "ci_hi": hi,
            "p_value": "", "sig_p05": f"{(sub.pvalue < 0.05).mean():.3f}",
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e1_stats.csv"), index=False)
    return {"share_sig": share,
            "alpha_means": df.groupby("alpha").spearman_rho.mean().round(3).to_dict()}


# ============================== E3 ==============================
def run_e3():
    df = pd.read_csv(os.path.join(RESULTS, "e3_heavytail.csv"))
    # robust 与 raw 使用不同专家 -> 按 spike 配置聚合后配对
    key = ["spike_rate", "spike_amp", "seed"]
    agg = df.groupby(key + ["group"])[["mse_spike", "degradation"]].mean().reset_index()
    piv_d = agg.pivot_table(index=key, columns="group", values="degradation").dropna()
    piv_m = agg.pivot_table(index=key, columns="group", values="mse_spike").dropna()

    w_d = stats.wilcoxon(piv_d["robust"], piv_d["raw"])
    w_m = stats.wilcoxon(piv_m["robust"], piv_m["raw"])
    FDR_FAMILY.append(("E3|wilcoxon_degradation|robust_vs_raw", w_d.pvalue))
    FDR_FAMILY.append(("E3|wilcoxon_mse_spike|robust_vs_raw", w_m.pvalue))

    rel = (piv_d["raw"] - piv_d["robust"]) / piv_d["raw"]  # 相对改善（degradation）
    rel_m = (piv_m["raw"] - piv_m["robust"]) / piv_m["raw"]
    idx = RNG.integers(0, len(rel), size=(N_BOOT, len(rel)))
    ci_rel = np.percentile(rel.values[idx].mean(axis=1), [2.5, 97.5])
    ci_rel_m = np.percentile(rel_m.values[idx].mean(axis=1), [2.5, 97.5])

    pair_rows = piv_d.reset_index().rename(columns={"robust": "degr_robust", "raw": "degr_raw"})
    pair_rows["rel_improvement"] = rel.values
    summary = pd.DataFrame([{
        "level": "summary", "n_pairs": len(piv_d),
        "degr_robust_mean": piv_d["robust"].mean(), "degr_raw_mean": piv_d["raw"].mean(),
        "rel_improvement_mean": rel.mean(), "rel_ci_lo": ci_rel[0], "rel_ci_hi": ci_rel[1],
        "rel_mse_mean": rel_m.mean(), "rel_mse_ci_lo": ci_rel_m[0], "rel_mse_ci_hi": ci_rel_m[1],
        "wilcoxon_stat_degr": w_d.statistic, "wilcoxon_p_degr": w_d.pvalue,
        "wilcoxon_stat_mse": w_m.statistic, "wilcoxon_p_mse": w_m.pvalue,
    }])
    pair_rows["level"] = "pair"
    out = pd.concat([pair_rows, summary], ignore_index=True)
    out.to_csv(os.path.join(OUT, "e3_stats.csv"), index=False)
    return {"n_pairs": len(piv_d), "p_degr": w_d.pvalue, "p_mse": w_m.pvalue,
            "rel_mean": float(rel.mean()), "rel_ci": [float(ci_rel[0]), float(ci_rel[1])]}


# ============================== E6 ==============================
def run_e6():
    df = pd.read_csv(os.path.join(RESULTS, "e6_epf_main.csv"))

    # (1) 冠亚军 market 级配对 Wilcoxon（n=5 市场）
    mm = df.groupby(["market", "expert_id"]).test_mse.mean().unstack()
    a, b = mm["M63"], mm["M47"]
    w = stats.wilcoxon(a, b)
    FDR_FAMILY.append(("E6|wilcoxon_market|M63_vs_M47", w.pvalue))

    # (2) 每专家 5 市场均值的 (market,seed) 分层 block bootstrap CI
    experts = sorted(df.expert_id.unique())
    markets = sorted(df.market.unique())
    ci_rows = []
    for e in experts:
        sub = df[df.expert_id == e]
        per_ms = sub.groupby(["market", "seed"]).test_mse.mean()  # 每 block 一个值
        market_vals = {m: per_ms.loc[m].values for m in markets}
        base_stat = float(np.mean([v.mean() for v in market_vals.values()]))
        # 分层 block bootstrap：每个市场内有放回重采样种子
        boot = np.empty(N_BOOT)
        arr = {m: market_vals[m] for m in markets}
        for m in markets:
            arr[m] = arr[m][RNG.integers(0, len(arr[m]), size=(N_BOOT, len(arr[m])))].mean(axis=1)
        boot = np.mean(np.vstack([arr[m] for m in markets]), axis=0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci_rows.append({"section": "expert_ci", "expert_id": e, "market_mean_mse": base_stat,
                        "ci_lo": lo, "ci_hi": hi})
    ci_df = pd.DataFrame(ci_rows).sort_values("market_mean_mse")

    # (3) TOST 等效检验：log MSE 配对差，margin = 1% MSE
    pair = df[df.expert_id.isin(["M63", "M47"])].pivot_table(
        index=["market", "seed"], columns="expert_id", values="test_mse").dropna()
    d = np.log(pair["M63"]) - np.log(pair["M47"])
    n = len(d)
    md, sd = d.mean(), d.std(ddof=1)
    margin = np.log(1.01)
    se = sd / np.sqrt(n)
    t1 = (md - (-margin)) / se
    p1 = 1 - stats.t.cdf(t1, df=n - 1)
    t2 = (margin - md) / se
    p2 = 1 - stats.t.cdf(t2, df=n - 1)
    tost_p = max(p1, p2)
    ci90 = (md - stats.t.ppf(0.95, n - 1) * se, md + stats.t.ppf(0.95, n - 1) * se)
    FDR_FAMILY.append(("E6|tost_max_p|M63_vs_M47_logMSE_margin1pct", tost_p))

    top = pd.DataFrame([{
        "section": "wilcoxon_M63_vs_M47", "expert_id": "", "market_mean_mse": "",
        "ci_lo": f"n_markets=5 (功效不足)", "ci_hi": f"stat={w.statistic:.1f}, p={w.pvalue:.4f}",
    }, {
        "section": "tost_M63_vs_M47", "expert_id": "",
        "market_mean_mse": f"mean_log_diff={md:.5f}",
        "ci_lo": f"90%CI=[{ci90[0]:.5f},{ci90[1]:.5f}]",
        "ci_hi": f"margin=±{margin:.5f}, TOST_p={tost_p:.4f}, equivalent={tost_p < 0.05}",
    }])
    out = pd.concat([top, ci_df], ignore_index=True)
    out.to_csv(os.path.join(OUT, "e6_stats.csv"), index=False)
    return {"wilcoxon_p": w.pvalue, "tost_p": float(tost_p), "tost_equiv": bool(tost_p < 0.05),
            "mean_log_diff": float(md), "ci90": [float(ci90[0]), float(ci90[1])],
            "best_expert": ci_df.iloc[0].expert_id,
            "best_ci": [float(ci_df.iloc[0].market_mean_mse), float(ci_df.iloc[0].ci_lo), float(ci_df.iloc[0].ci_hi)],
            "m63_row": ci_df[ci_df.expert_id == "M63"].iloc[0].tolist(),
            "m47_row": ci_df[ci_df.expert_id == "M47"].iloc[0].tolist()}


# ============================== E8 ==============================
def run_e8():
    df = pd.read_csv(os.path.join(RESULTS, "e8_stress_test.csv"))
    s = df[df.axis != "baseline"].copy()
    rows = []
    for (ax, pa), sub in s.groupby(["axis", "param"]):
        d = sub.degradation.dropna().values
        q1, q3 = np.percentile(d, [25, 75])
        rows.append({
            "section": "axis_config", "axis": ax, "param": pa, "n": len(d),
            "mean": d.mean(), "std": d.std(), "median": np.median(d),
            "iqr": q3 - q1, "q1": q1, "q3": q3,
            "winsor_mean_1_99": winsorized_mean(d),
            "n_gt10": int((d > 10).sum()), "share_gt10": float((d > 10).mean()),
            "n_sentinel_base1": int((sub.baseline_mse == 1.0).sum()),
        })
    top10 = s.nlargest(10, "degradation")[
        ["market", "expert_id", "axis", "param", "baseline_mse", "test_mse", "degradation"]]
    top10 = top10.assign(section="top10_degradation").rename(columns={"param": "param"})
    for c in ["n", "mean", "std", "median", "iqr", "q1", "q3", "winsor_mean_1_99", "n_gt10", "share_gt10", "n_sentinel_base1"]:
        top10[c] = ""
    out = pd.concat([pd.DataFrame(rows), top10[pd.DataFrame(rows).columns.tolist() + ["market", "expert_id", "baseline_mse", "test_mse", "degradation"]].reindex(columns=list(pd.DataFrame(rows).columns) + ["market", "expert_id", "baseline_mse", "test_mse", "degradation"])], ignore_index=True)
    out.to_csv(os.path.join(OUT, "e8_stats.csv"), index=False)
    miss = s[s.axis == "missingness"].degradation
    return {"miss_mean": float(miss.mean()), "miss_median": float(miss.median()),
            "n_gt10_miss": int((miss > 10).sum()), "share_gt10_miss": float((miss > 10).mean()),
            "top10": top10[["market", "expert_id", "axis", "param", "degradation"]].values.tolist()}


# ============================== E9 ==============================
def run_e9():
    df = pd.read_csv(os.path.join(RESULTS, "e9_incremental.csv"))
    final = df[df.month == df.month.max()].pivot_table(
        index=["market", "seed"], columns="strategy", values="cum_regret").reset_index()
    rows = []
    for strat in ["fixed", "hedge", "ctx_hedge"]:
        for mkt, sub in final.groupby("market"):
            vals = np.asarray(sub[strat].values, dtype=float)
            lo, hi = boot_ci_mean(vals)
            rows.append({"section": "final_cum_regret", "market": mkt, "strategy": strat,
                         "n_seeds": len(vals), "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi,
                         "values": ";".join(f"{v:.3f}" for v in vals)})
    # DE 敏感性：剔除 top-1% 极端月度 regret 后重算累计（诊断，不改原始数据）
    de = df[df.market == "DE"].copy()
    thresh = np.percentile(de.regret, 99)
    de_f = de[de.regret <= thresh]
    sens = (de_f.sort_values(["strategy", "seed", "month"])
            .groupby(["strategy", "seed"]).regret.sum().reset_index())
    for strat, sub in sens.groupby("strategy"):
        vals = sub.regret.values
        lo, hi = boot_ci_mean(vals)
        rows.append({"section": "DE_drop_top1pct", "market": "DE", "strategy": strat,
                     "n_seeds": len(vals), "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi,
                     "values": ";".join(f"{v:.3f}" for v in vals)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e9_stats.csv"), index=False)
    np_means = {r.strategy: r["mean"] for _, r in out[(out.section == "final_cum_regret") & (out.market == "NP")].iterrows()}
    de_means = {r.strategy: r["mean"] for _, r in out[(out.section == "final_cum_regret") & (out.market == "DE")].iterrows()}
    de_sens = {r.strategy: r["mean"] for _, r in out[out.section == "DE_drop_top1pct"].iterrows()}
    return {"np_means": np_means, "de_means": de_means, "de_sens": de_sens,
            "thresh": float(thresh)}


# ============================== E11 ==============================
def run_e11():
    df = pd.read_csv(os.path.join(RESULTS, "e11_lodo_spearman.csv"))
    epf_internal = ["NP", "PJM", "BE", "FR", "DE"]
    rows = []
    for _, r in df.iterrows():
        lo, hi = fisher_ci(r.spearman_rho, int(r.n_experts))
        rows.append({"level": "row", "held_out": r.held_out, "seed": r.seed,
                     "spearman_rho": r.spearman_rho, "ci_lo": lo, "ci_hi": hi,
                     "p_value": r.p_value, "group": "EPF内部" if r.held_out in epf_internal else "外部"})
        FDR_FAMILY.append((f"E11|{r.held_out}|s={r.seed}", r.p_value))
    for dom, sub in df.groupby("held_out"):
        lo, hi = boot_ci_mean(sub.spearman_rho.values)
        rows.append({"level": "domain_summary", "held_out": dom, "seed": "",
                     "spearman_rho": sub.spearman_rho.mean(), "ci_lo": lo, "ci_hi": hi,
                     "p_value": "", "group": "EPF内部" if dom in epf_internal else "外部"})
    for gname, gset in [("EPF内部5域", epf_internal), ("外部3域", ["ETTh1", "Weather", "Exchange"])]:
        sub = df[df.held_out.isin(gset)]
        lo, hi = boot_ci_mean(sub.spearman_rho.values)
        rows.append({"level": "group_summary", "held_out": gname, "seed": "",
                     "spearman_rho": sub.spearman_rho.mean(), "ci_lo": lo, "ci_hi": hi,
                     "p_value": "", "group": gname})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e11_stats.csv"), index=False)
    return {
        "epf_mean": float(df[df.held_out.isin(epf_internal)].spearman_rho.mean()),
        "ext_mean": float(df[~df.held_out.isin(epf_internal)].spearman_rho.mean()),
        "domain_means": df.groupby("held_out").spearman_rho.mean().round(3).to_dict(),
    }


# ============================== FDR ==============================
def run_fdr():
    ids = [x[0] for x in FDR_FAMILY]
    ps = np.array([x[1] for x in FDR_FAMILY], dtype=float)
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ps)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qvals = np.empty(m)
    qvals[order] = q
    out = pd.DataFrame({"comparison": ids, "p_raw": ps, "q_bh": qvals,
                        "sig_q05": qvals < 0.05})
    out.to_csv(os.path.join(OUT, "fdr_qvalues.csv"), index=False)
    return {"n_comparisons": m, "n_sig_raw": int((ps < 0.05).sum()),
            "n_sig_q": int((qvals < 0.05).sum()),
            "nonsig": out[~out.sig_q05].comparison.tolist()}


if __name__ == "__main__":
    import json
    res = {}
    for name, fn in [("e1", run_e1), ("e3", run_e3), ("e6", run_e6),
                     ("e8", run_e8), ("e9", run_e9), ("e11", run_e11)]:
        res[name] = fn()
        print(f"[done] {name}")
    res["fdr"] = run_fdr()
    print("[done] fdr")
    with open(os.path.join(OUT, "_run_results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:3000])
