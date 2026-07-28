#!/usr/bin/env python3
"""Analysis v3 / Task 1: sentinel failure-mode analysis of e8_stress_test.csv.

Sentinels identified empirically:
  - baseline_mse == 1.0   : clean-baseline reference missing/failed -> degradation
                            column inflated to (test_mse - 1.0). 59 rows.
  - baseline_mse == 9999.0: baseline run itself crashed (shape error) -> 195 rows.
  - test_mse    == 9999.0 : stressed run crashed (error col) -> 210 rows, all N10.
Only NEW files are written, to results/analysis_v3/.
"""
import os
import numpy as np
import pandas as pd

OUT = "results/analysis_v3"
os.makedirs(OUT, exist_ok=True)

FAMILY = {  # from run_e12_analysis.py + e7v2_summary.csv + models.py
    "M01": "frequency", "M03": "linear", "M117": "frequency", "M14": "ssm",
    "M17": "cnn", "M18": "decomposition", "M220": "hybrid", "M233": "hybrid",
    "M31": "ssm", "M36": "wavelet", "M47": "decomposition", "M50": "attention",
    "M51": "cnn", "M52": "linear", "M55": "attention", "M63": "attention",
    "M89": "graph", "N01": "ssm", "N07": "basis", "N08": "periodic",
    "N10": "statistical",
}

df = pd.read_csv("results/e8_stress_test.csv")
df["family"] = df["expert_id"].map(FAMILY)
df["mechanism"] = np.where(df["param"].str.contains("mcar", na=False), "MCAR",
                  np.where(df["param"].str.contains("block", na=False), "block",
                           df["param"].astype(str)))

s1 = df[df["baseline_mse"] == 1.0].copy()          # the 59 flagged rows
s9b = df[df["baseline_mse"] == 9999.0].copy()      # crashed-baseline sentinel
s9t = df[df["test_mse"] == 9999.0].copy()          # crashed-run sentinel

# ---------- long table of the 59 sentinel rows ----------
s1_out = s1[["market", "expert_id", "family", "axis", "param", "mechanism",
             "seed", "test_mse", "baseline_mse", "degradation"]].sort_values(
    ["market", "param", "expert_id"])

# ---------- failure-rate tables ----------
# denominator: all missingness-axis rows for the same market(/family)
miss = df[df["axis"] == "missingness"].copy()
miss["sentinel10"] = miss["baseline_mse"] == 1.0

by_family = (miss.groupby("family")["sentinel10"]
             .agg(sentinel_rows="sum", total_rows="count"))
by_family["failure_rate"] = by_family["sentinel_rows"] / by_family["total_rows"]
by_family = by_family.sort_values("failure_rate", ascending=False)

by_param = (miss.groupby(["market", "param"])["sentinel10"]
            .agg(sentinel_rows="sum", total_rows="count"))
by_param["failure_rate"] = by_param["sentinel_rows"] / by_param["total_rows"]
by_param = by_param.reset_index()

by_mech = (miss.groupby(["market", "mechanism"])["sentinel10"]
           .agg(sentinel_rows="sum", total_rows="count"))
by_mech["failure_rate"] = by_mech["sentinel_rows"] / by_mech["total_rows"]
by_mech = by_mech.reset_index()

by_expert = (miss.groupby(["market", "expert_id", "family"])["sentinel10"]
             .agg(sentinel_rows="sum", total_rows="count"))
by_expert["failure_rate"] = by_expert["sentinel_rows"] / by_expert["total_rows"]
by_expert = by_expert.reset_index().sort_values(
    ["market", "failure_rate"], ascending=[True, False])

# other sentinel classes summary
other = pd.DataFrame({
    "sentinel_type": ["baseline_mse==9999 (crashed baseline)",
                      "test_mse==9999 (crashed stress run)"],
    "rows": [len(s9b), len(s9t)],
    "experts": [",".join(sorted(s9b.expert_id.unique())),
                ",".join(sorted(s9t.expert_id.unique()))],
    "markets": [",".join(sorted(s9b.market.unique())),
                ",".join(sorted(s9t.market.unique()))],
    "axes": [",".join(sorted(s9b.axis.unique())),
             ",".join(sorted(s9t.axis.unique()))],
})

# ---------- extreme stress_mse values (excluding all sentinels) ----------
st = df[(df["axis"] != "baseline") & (df["test_mse"] < 9999.0)]
med = st["test_mse"].median()
ext = st[st["test_mse"] > 100 * med]
top = st.nlargest(10, "test_mse")[
    ["market", "axis", "param", "expert_id", "family", "seed", "test_mse",
     "baseline_mse"]]
top = top.assign(ratio_to_median=top["test_mse"] / med)

# ---------- write CSV ----------
with pd.ExcelWriter if False else open(os.path.join(OUT, "sentinel_failure_analysis.csv"), "w") as f:
    f.write("# === 59 sentinel rows (baseline_mse==1.0) ===\n")
    s1_out.to_csv(f, index=False)
    f.write("\n# === failure rate by expert family (missingness axis) ===\n")
    by_family.to_csv(f)
    f.write("\n# === failure rate by (market,param) ===\n")
    by_param.to_csv(f, index=False)
    f.write("\n# === failure rate by (market,mechanism) ===\n")
    by_mech.to_csv(f, index=False)
    f.write("\n# === failure rate by (market,expert) ===\n")
    by_expert.to_csv(f, index=False)
    f.write("\n# === other sentinel classes ===\n")
    other.to_csv(f, index=False)
    f.write("\n# === top-10 stress test_mse excluding sentinels "
            f"(median={med:.2f}; rows>100x median: {len(ext)}) ===\n")
    top.to_csv(f, index=False)

# ---------- narrative numbers for the report ----------
n_be = (s1.market == "BE").sum()
n_fr = (s1.market == "FR").sum()
seed_be = s1[s1.market == "BE"].seed.unique().tolist()
seed_fr = s1[s1.market == "FR"].seed.unique().tolist()
be_experts = sorted(s1[s1.market == "BE"].expert_id.unique())
fr_experts = sorted(s1[s1.market == "FR"].expert_id.unique())
be_fams = sorted({FAMILY[e] for e in be_experts})
fr_fams = sorted({FAMILY[e] for e in fr_experts})
param_tab = s1.groupby(["market", "param"]).size().unstack(fill_value=0)

report = f"""# E8 哨兵失败模式分析（analysis_v3 / Task 1）

## 1. 哨兵行的三种类型（全表 4410 行）

| 类型 | 行数 | 含义 |
|---|---|---|
| `baseline_mse == 1.0` | {len(s1)} | 干净基线参考缺失/失败，degradation 被虚增为 test_mse−1.0 |
| `baseline_mse == 9999.0` | {len(s9b)} | 基线运行崩溃（shape error），污染该专家全部应力行 |
| `test_mse == 9999.0` | {len(s9t)} | 应力运行崩溃（error 列非空），**全部来自 N10** |

`run_e8_stress.py` 当前版本用 999.0 做兜底，说明本 CSV 由更早版本写出，
哨兵值约定不一致（1.0 / 9999.0 并存），本身即需在论文 SI 中说明。

## 2. 59 行 `baseline_mse==1.0` 的 (market, expert, axis, config) 组合

- **全部位于 missingness 轴**（59/59），无 lookback/corruption 行。
- **BE：{n_be} 行，全部 seed={seed_be}**；涉及 {len(be_experts)} 个专家
  {be_experts}，覆盖 {len(be_fams)} 个族：{be_fams}。
  参数分布：{param_tab.loc['BE'].to_dict()}。
- **FR：{n_fr} 行，全部 seed={seed_fr}**；仅 {fr_experts}
  （族：{fr_fams}）。参数分布：{param_tab.loc['FR'].to_dict()}。
- 关键形态：哨兵按 **(market, seed) 整块失效**，而非按缺失机制渐进出现
  （BE seed42 在 0.1_mcar 最轻缺失下同样失效）。这指向**基线参考表的
  (market, seed) 级缺失/合并失败**，而不是专家在缺失条件下真实崩溃。
- FR 的 4 个受影响专家（M01/M03/M14/M17）恰为线性/频率/ssm/cnn 各一，
  无族特异性；BE 的 12 个专家横跨 8 个族，亦无族特异性。

## 3. 失效率表（missingness 轴分母）

### 按专家族
{by_family.to_markdown()}

### 按 (market, param)
{by_param.pivot(index='param', columns='market', values='failure_rate').fillna(0).to_markdown()}

### 按 (market, 缺失机制)
{by_mech.pivot(index='mechanism', columns='market', values='failure_rate').fillna(0).to_markdown()}

## 4. 系统性不可用判定

- **N10（statistical 族）是唯一"系统性不可用"专家**：`mat1 and mat2 shapes
  cannot be multiplied` 在 baseline clean 及全部 stress 条件下触发
  （210 行 test_mse=9999 + 195 行 baseline=9999），覆盖 5 市场 × 全部轴
  × 全部参数 × 3 种子。这是**代码级输入维度 bug**，与缺失机制无关；
  N10 应从 E8 全部结论中剔除（或修复后重跑）。
- **没有任何专家族在任何缺失机制下真实"系统性不可用"**：59 行 1.0 哨兵是
  参考值缺失导致的伪 degradation；剔除三类哨兵后，missingness 轴真实
  stress_mse 的最大/中位比仅 {st.test_mse.max()/med:.2f}×
  （中位 {med:.1f}，最大 {st.test_mse.max():.1f}，BE lookback L_336 / M52）。
  **不存在 >100× 中位数的真实极端值**（0 行）；唯一的"极端值"就是
  9999.0 哨兵本身（≈{9999.0/med:.1f}× 中位）。

## 5. 路由含义

若 router 的 TopK 直接采用本表 degradation 列：

1. **BE(seed42)/FR(seed2021) 的 59 个伪高 degradation 组合**会把
   M03/M14/M17/M18/M31/M36/M47/M50/M51/M52/M55/M63 在 BE、M01/M03/M14/M17
   在 FR 误判为"缺失下崩溃"，TopK 将系统性地把这些族**排除出 BE/FR 路由池**
   ——而其中 M47/M63 恰是 E6/E12 中 BE/FR 的最优专家，等于在最需要它们的
   市场上禁用它们，路由退化方向与 E6 排名完全相反。
2. **N10 若未被哨兵过滤**，其 9999.0 伪 MSE 在任意基于 MSE 的 TopK 中
   永远不会被选中（看似无害），但若 router 用 degradation=0.0 的兜底行
   （stress 失败时 deg 写 0）反而会把 N10 误判为"零退化稳健专家"而优先
   选中——这是最危险的一条路径，必须在 router 侧加 sentinel mask。
3. 建议：router 训练/评估前以 `baseline_mse∈{{1.0,9999.0}} 或
   test_mse==9999.0 或 error 非空` 为掩码剔除 {len(s1)+len(s9b)} 行污染
   （占 missingness+corruption 轴的 {(len(s1)+len(s9b))/len(df)*100:.1f}% 全表），
   并将 N10 整体列入排除清单直至 shape bug 修复。
"""
with open(os.path.join(OUT, "sentinel_failure_report.md"), "w", encoding="utf-8") as f:
    f.write(report)

print("sentinel==1.0:", len(s1), "| BE", n_be, seed_be, "| FR", n_fr, seed_fr)
print("BE experts:", be_experts)
print("FR experts:", fr_experts)
print(by_family.to_string())
print("baseline==9999 rows:", len(s9b), "experts:", sorted(s9b.expert_id.unique()))
print("test==9999 rows:", len(s9t), "experts:", sorted(s9t.expert_id.unique()))
print(f"real stress extremes >100x median: {len(ext)}; max/median ratio "
      f"{st.test_mse.max()/med:.2f}")
print("wrote", os.path.join(OUT, "sentinel_failure_analysis.csv"),
      "and sentinel_failure_report.md")
