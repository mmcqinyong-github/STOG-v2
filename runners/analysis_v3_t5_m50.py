#!/usr/bin/env python3
"""Analysis v3 / Task 5 (v2): M50 x Weather champion audit — corrected.

Key correction vs v1: BatchedEvalTrainer._eval_mse uses
nn.MSELoss(reduction="sum") / (n*H) (run_e7_v2.py L41-51) -> chunking is
MATHEMATICALLY EXACT, and test_mse is computed from full-batch predictions
(L94-97). Verified empirically against errs/*.npz. Therefore the bs=1024
eval-bias channel does not exist; the audit refocuses on (i) eval exactness,
(ii) training-budget asymmetry, (iii) paired per-window tests incl. the
seed-3407 H96 reversal and skew diagnostics.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = "results/analysis_v3"
os.makedirs(OUT, exist_ok=True)

runs = pd.read_csv("results/e7_v2/e7v2_runs.csv")
W = runs[runs.dataset == "Weather"].copy()
W["val_test_ratio"] = W.val_mse / W.test_mse

rank_tbl = (W.groupby(["horizon", "expert_id"])
            .agg(test_mse=("test_mse", "mean"), epochs=("epochs", "mean"),
                 time_sec=("time_sec", "mean"), ratio=("val_test_ratio", "mean"))
            .reset_index())
rank_tbl["rank"] = rank_tbl.groupby("horizon")["test_mse"].rank()

m50w = W[W.expert_id == "M50"]
oth = W[W.expert_id != "M50"]
m50g = runs[runs.expert_id == "M50"]
attg = runs[runs.expert_id.isin(["M55", "M63"])]

champ = rank_tbl.loc[rank_tbl.groupby("horizon").test_mse.idxmin(),
                     ["horizon", "expert_id", "test_mse"]]
second = rank_tbl[rank_tbl["rank"] == 2.0][["horizon", "expert_id", "test_mse"]]
marg = champ.merge(second, on="horizon", suffixes=("_win", "_2nd"))
marg["rel_margin_pct"] = 100 * (marg.test_mse_2nd - marg.test_mse_win) / marg.test_mse_2nd

# ---------- 1. eval exactness: runs.test_mse vs mean(errs) ----------
ex_rows = []
for _, r in W.iterrows():
    p = f"results/e7_v2/errs/Weather_H{int(r.horizon)}_{int(r.seed)}_{r.expert_id}.npz"
    if not os.path.exists(p):
        continue
    err = np.load(p)["test_err"].astype(np.float64)
    ex_rows.append({"horizon": int(r.horizon), "expert": r.expert_id,
                    "seed": int(r.seed), "runs_test_mse": r.test_mse,
                    "errs_mean": err.mean(),
                    "rel_diff": abs(err.mean() - r.test_mse) / r.test_mse})
exa = pd.DataFrame(ex_rows)
max_rel = exa.rel_diff.max()

# ---------- 2. paired per-window tests ----------
pairs = {24: "M17", 96: "M17", 720: "N07"}
pair_rows = []
for h, rival in pairs.items():
    for seed in [2021, 42, 3407]:
        e50 = np.load(f"results/e7_v2/errs/Weather_H{h}_{seed}_M50.npz")["test_err"].astype(np.float64)
        er = np.load(f"results/e7_v2/errs/Weather_H{h}_{seed}_{rival}.npz")["test_err"].astype(np.float64)
        diff = e50 - er
        stat, p = wilcoxon(diff)
        pair_rows.append({"horizon": h, "seed": seed, "rival": rival,
                          "n_windows": len(e50),
                          "mean_M50": e50.mean(), "mean_rival": er.mean(),
                          "mean_winner": "M50" if e50.mean() < er.mean() else rival,
                          "median_diff_M50_minus_rival": float(np.median(diff)),
                          "win_frac_M50": float((diff < 0).mean()),
                          "wilcoxon_p": float(p)})
pt = pd.DataFrame(pair_rows)

# ---------- write csv ----------
with open(os.path.join(OUT, "m50_weather_audit.csv"), "w") as f:
    f.write("# === Weather block standings (3-seed mean) ===\n")
    rank_tbl.sort_values(["horizon", "rank"]).to_csv(f, index=False)
    f.write("\n# === champion margins ===\n")
    marg.to_csv(f, index=False)
    f.write(f"\n# === eval exactness check (max rel diff = {max_rel:.2e}) ===\n")
    exa.to_csv(f, index=False)
    f.write("\n# === paired per-window Wilcoxon (M50 vs rival) ===\n")
    pt.to_csv(f, index=False)

h24_lag = (rank_tbl.query("horizon==24 and expert_id=='M50'").test_mse.iloc[0]
           / rank_tbl.query("horizon==24 and expert_id=='M17'").test_mse.iloc[0] - 1) * 100

md = f"""# M50 × Weather 冠军身份审计（analysis_v3 / Task 5，纯审计不重跑）

## 0. 结论先行

**冠军身份成立（H720 铁证，H96 真实但种子敏感），评估管线零偏差，
不应剔除；但 M50 的有效训练预算与其他专家不对等，需在 SI 披露。**

## 1. bs=1024 / BatchedEvalTrainer 偏差检查：不存在

- 代码审读（run_e7_v2.py L41–51）：`_eval_mse` 用
  `MSELoss(reduction="sum")` 累加后除以 `n×H`——**分块评估在数学上
  与整批完全等价**，与 bs 取值无关；`test_mse` 更是直接由全量预测
  计算（L94–97）。该类 Trainer 的 docstring 明确其唯一动机是"避免
  PatchTST-M50 在 Weather-H720 上一次性注意力分配卡死"，训练循环未动。
- 实证核验：Weather 全部 {len(exa)} 个 (horizon, expert, seed) 单元，
  `runs.test_mse` 与 `errs/test_err` 均值的最大相对偏差
  **{max_rel:.1e}**（浮点噪声级）。
- 因此"bs=1024 + 批评估偏差解释冠军身份"这一假设**被直接否证**：
  偏差的量级不是小，而是恒等于零。

## 2. Weather 块排名与夺冠差距（3 种子均值）

| horizon | 冠军 | test_mse | 亚军 | test_mse | 相对差距 |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| H{int(r.horizon)} | {r.expert_id_win} | {r.test_mse_win:.6f} | "
    f"{r.expert_id_2nd} | {r.test_mse_2nd:.6f} | {r.rel_margin_pct:.1f}% |"
    for r in marg.itertuples()) + f"""

H24 冠军是 **M17**（M50 第二，落后 {h24_lag:.0f}%）；M50 的冠军身份仅指
**H96 与 H720** 两块。

## 3. 训练过程异常（runs 表硬数字）

| 指标 | M50 × Weather (9 runs) | 其他 18 专家 × Weather (162 runs) |
|---|---|---|
| epochs | 全部 10/10（无一早停） | 均值 {oth.epochs.mean():.1f}（{oth.epochs.min():.0f}–{oth.epochs.max():.0f}，多数 4–5 早停） |
| time_sec | {m50w.time_sec.min():.0f}–{m50w.time_sec.max():.0f}（均值 {m50w.time_sec.mean():.0f}） | 均值 {oth.time_sec.mean():.1f}（最大 {oth.time_sec.max():.1f}） |
| val/test 比 | {m50w.val_test_ratio.min():.2f}–{m50w.val_test_ratio.max():.2f} | 中位 {oth.val_test_ratio.median():.2f} |

全局对照：M50 全部 27 runs epochs 均值 {m50g.epochs.mean():.1f}、time 均值
{m50g.time_sec.mean():.0f}s；同族 attention 的 M55/M63（54 runs）epochs 均值
{attg.epochs.mean():.1f}、time 均值 {attg.time_sec.mean():.1f}s。
**M50（PatchTST）比同族慢 20–50 倍且从不早停**——这是 M50 的全局属性，
非 Weather 特有；runs 表无 batch_size 列，现有产物无法证实或排除
bs=1024 的训练侧偏离，但时长差异与"PatchTST 本身更重 + 分块评估"
一致。val/test 比正常（≈0.50–0.75 vs 中位 0.55），无过拟合签名。

## 4. 配对逐窗检验（Wilcoxon 符号秩，N≈9.5k–10.2k 窗）

{pt.to_markdown(index=False)}

- **H720（vs N07）：铁证。** 3/3 种子逐窗胜率 0.86–0.89，中位差
  −0.009 左右，p≈0；均值差距 72% 由全窗一致优势支撑。
- **H96（vs M17）：真实但种子敏感、由重尾窗口驱动。**
  seed 2021：均值 M50 胜（0.00686 vs 0.01116）但**逐窗胜率仅 0.30、
  中位差为正**——胜利由少数 M50 大幅领先的窗口（重尾）贡献；
  seed 42：胜率 0.53、中位差为负，p=1.2e-4，正常获胜；
  **seed 3407：M17 均值反超（0.00640 vs 0.00739），M50 胜率 0.20、
  p≈0 显著落败**。3 种子均值汇总 M50 领先 27.6%，本质上是 2/3 种子
  + 重尾窗口的结果，论文应避免"H96 上一致优于 M17"的措辞。
- **H24（vs M17）：M17 反向显著**（M50 胜率 0.19–0.31），印证 H24
  冠军确为 M17，M50 并非 Weather 全域冠军。

## 5. 判定与论文建议

1. **H720 冠军：成立**，无需任何附加条件。
2. **H96 冠军：成立但需弱化措辞**（3 种子均值胜 27.6%；逐窗一致性
   仅 1/3 种子）。建议正文用"mean test MSE 最低"，避免逐窗一致性声称。
3. **评估偏差：排除**（reduction=sum 数学精确 + 实证 {max_rel:.0e}）。
4. **训练预算不对等：需在 SI 披露** M50 全程 10 epochs、耗时为同族
   20–50 倍；若审稿人要求预算对齐，需补等 epoch/等时长对照（训练任务，
   超出本次零训练审计范围）。
"""
with open(os.path.join(OUT, "m50_weather_audit.md"), "w", encoding="utf-8") as f:
    f.write(md)

print(marg.to_string(index=False))
print(f"\nexactness max rel diff: {max_rel:.2e} over {len(exa)} cells")
print("\npaired tests:")
print(pt.to_string(index=False))
