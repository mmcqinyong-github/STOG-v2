"""P3 Task 2: E11 v2 — add true external domains (ETTm1, ECL, Solar) and recompute stratified LODO.

Usage:
  python run_e11_v2.py run [--domains ETTm1,ECL,Solar]   # resumable training
  python run_e11_v2.py analyze                            # stratified LODO + correlation + heatmap

Protocol mirrors run_e11_crossdomain.py: chronological 70/10/20, lookback 168, horizon 24,
z-score with train stats, target = last column. ECL/Solar are channel-subsampled (~24-26 channels,
last/target channel always kept) to keep flattened d_in tractable; this is documented in outputs.
Checkpoint: results/p3/e11v2_newdomains_runs.csv (append after every run; reruns skip finished combos).
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.utils.common import set_seed
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer

OUT_DIR = "./results/p3"
FIG_DIR = "./results/figures"
RUNS_CSV = os.path.join(OUT_DIR, "e11v2_newdomains_runs.csv")
SEEDS = [2021, 42, 3407]
# 19 experts identical to E6/E11 (registry also has M36/M51 which E6/E11 excluded)
EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52", "M55",
              "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
TRAIN_CFG = {"max_epochs": 5, "patience": 2, "batch_size": 256, "lr": 1e-4}
LT_BASE = "./dataset/Long-term multivariate dataset"

EPF_DOMAINS = ["NP", "PJM", "BE", "FR", "DE"]
NEW_DOMAINS = {
    "ETTm1": {"path": f"{LT_BASE}/ETT-small/ETTm1.csv", "date_col": "date", "subsample": None},
    "ECL": {"path": f"{LT_BASE}/electricity/electricity.csv", "date_col": "date", "subsample": 13},
    "Solar": {"path": f"{LT_BASE}/solar/solar_AL.txt", "date_col": None, "subsample": 6},
}


def load_domain_data(name):
    meta = NEW_DOMAINS[name]
    if meta["date_col"]:
        df = pd.read_csv(meta["path"])
        if meta["date_col"] in df.columns:
            df = df.drop(columns=[meta["date_col"]])
        data = df.values.astype(np.float32)
    else:
        data = np.loadtxt(meta["path"], delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0)
    n_ch = data.shape[1]
    if meta["subsample"] and n_ch > 30:
        keep = list(range(0, n_ch, meta["subsample"]))
        if (n_ch - 1) not in keep:
            keep.append(n_ch - 1)  # always keep target (last) channel
        data = data[:, keep]
    return data


def make_lt_windows(data, lookback=168, horizon=24):
    n = len(data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    splits = [data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]]

    def mk(sd):
        X, y = [], []
        for i in range(len(sd) - lookback - horizon + 1):
            X.append(sd[i:i + lookback].flatten())
            y.append(sd[i + lookback:i + lookback + horizon, -1])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    trX, trY = mk(splits[0])
    vaX, vaY = mk(splits[1])
    teX, teY = mk(splits[2])
    mean = trX.mean(dim=0)
    std = trX.std(dim=0) + 1e-8
    return {"train": (trX - mean) / std, "train_tgt": trY,
            "val": (vaX - mean) / std, "val_tgt": vaY,
            "test": (teX - mean) / std, "test_tgt": teY}


def load_done():
    if os.path.exists(RUNS_CSV):
        df = pd.read_csv(RUNS_CSV)
        return df, set(zip(df["domain"], df["seed"], df["expert_id"]))
    return pd.DataFrame(), set()


def run_domains(domain_names):
    os.makedirs(OUT_DIR, exist_ok=True)
    done_df, done = load_done()
    print(f"Resume state: {len(done_df)} rows already done")
    trainer = UnifiedTrainer(TRAIN_CFG)
    print(f"Device: {trainer.device}")

    for name in domain_names:
        t0 = time.time()
        data = load_domain_data(name)
        print(f"\n=== Domain {name}: shape={data.shape} ===")
        windows = make_lt_windows(data)
        d_in = windows["train"].shape[1]
        print(f"d_in={d_in} train={len(windows['train'])} test={len(windows['test'])}")

        class DM:
            pass
        dm = DM()
        dm.windows = windows

        for seed in SEEDS:
            set_seed(seed)
            for eid in EXPERT_IDS:
                if (name, seed, eid) in done:
                    continue
                try:
                    expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                    res = trainer.train_expert(expert, dm)
                    row = {"domain": name, "domain_type": "External", "expert_id": eid,
                           "seed": seed, "val_mse": res["val_mse"], "test_mse": res["test_mse"],
                           "test_mae": res.get("test_mae", 0.0), "error": ""}
                except Exception as ex:
                    row = {"domain": name, "domain_type": "External", "expert_id": eid,
                           "seed": seed, "val_mse": 9999.0, "test_mse": 9999.0,
                           "test_mae": 9999.0, "error": str(ex)[:200]}
                done_df = pd.concat([done_df, pd.DataFrame([row])], ignore_index=True)
                done.add((name, seed, eid))
                done_df.to_csv(RUNS_CSV, index=False)  # checkpoint after every run
                print(f"  {name} s{seed} {eid}: test_mse={row['test_mse']:.4f} {row['error']}")
        print(f"Domain {name} done in {time.time()-t0:.0f}s")
    print(f"\nAll requested domains complete. Rows: {len(done_df)}")


def lodo_table(df, train_domains, held_out, seeds, group_label):
    rows = []
    train_ranks = df[df["domain"].isin(train_domains)].groupby("expert_id")["rank"].mean()
    for seed in seeds:
        held = df[(df["domain"] == held_out) & (df["seed"] == seed)]
        if len(held) == 0:
            continue
        true_ranks = held.set_index("expert_id")["rank"]
        common = train_ranks.index.intersection(true_ranks.index)
        if len(common) < 3:
            continue
        rho, pval = stats.spearmanr(train_ranks[common], true_ranks[common])
        rows.append({"group": group_label, "held_out": held_out, "seed": seed,
                     "spearman_rho": rho, "p_value": pval, "n_experts": len(common),
                     "n_train_domains": len(train_domains)})
    return rows


def analyze():
    os.makedirs(OUT_DIR, exist_ok=True)
    old = pd.read_csv("./results/e11_crossdomain.csv")
    new = pd.read_csv(RUNS_CSV)
    df = pd.concat([old[["domain", "domain_type", "expert_id", "seed", "test_mse", "test_mae"]],
                    new[["domain", "domain_type", "expert_id", "seed", "test_mse", "test_mae"]]],
                   ignore_index=True)
    # drop only true failures (exact 9999.0 sentinel); large-but-real ECL MSEs are kept
    df = df[df["test_mse"].round(4) != 9999.0]
    df["rank"] = df.groupby(["domain", "seed"])["test_mse"].rank(method="min")

    domains = sorted(df["domain"].unique())
    external = [d for d in domains if d not in EPF_DOMAINS]
    print(f"Domains ({len(domains)}): {domains}")
    print(f"External ({len(external)}): {external}")

    rows = []
    # Group A: EPF-internal LODO (train = other EPF domains only)
    for h in EPF_DOMAINS:
        rows += lodo_table(df, [d for d in EPF_DOMAINS if d != h], h, SEEDS, "EPF_internal")
    # Group B: external LODO (train = other external domains only)
    for h in external:
        rows += lodo_table(df, [d for d in external if d != h], h, SEEDS, "External")
    # Reference: original mixed LODO over all domains
    for h in domains:
        rows += lodo_table(df, [d for d in domains if d != h], h, SEEDS, "All_mixed")
    # Cross-group transfer: external-trained -> EPF held-out, and vice versa
    for h in EPF_DOMAINS:
        rows += lodo_table(df, external, h, SEEDS, "Cross_ext2epf")
    for h in external:
        rows += lodo_table(df, EPF_DOMAINS, h, SEEDS, "Cross_epf2ext")

    lodo = pd.DataFrame(rows)
    lodo.to_csv(os.path.join(OUT_DIR, "e11v2_lodo_recomputed.csv"), index=False)
    summ = lodo.groupby("group").agg(mean_rho=("spearman_rho", "mean"),
                                     median_rho=("spearman_rho", "median"),
                                     min_rho=("spearman_rho", "min"),
                                     max_rho=("spearman_rho", "max"),
                                     n=("spearman_rho", "size"))
    print("\n=== Stratified LODO summary ===")
    print(summ.round(3).to_string())
    print("\n=== Per held-out domain (mean over seeds) ===")
    print(lodo.groupby(["group", "held_out"])["spearman_rho"].mean().round(3).to_string())

    # Full correlation matrix (all 11 domains)
    dm = df.groupby(["domain", "expert_id"])["test_mse"].mean().unstack()
    corr = dm.T.corr(method="spearman")
    corr = corr.reindex(index=EPF_DOMAINS + external, columns=EPF_DOMAINS + external)
    corr.to_csv(os.path.join(OUT_DIR, "e11v2_domain_correlation.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    # group separator
    ax.axhline(len(EPF_DOMAINS) - 0.5, color="k", lw=2)
    ax.axvline(len(EPF_DOMAINS) - 0.5, color="k", lw=2)
    ax.set_title("E11 v2: Cross-domain expert-rank Spearman correlation\n(5 EPF-internal | 6 external incl. new ETTm1/ECL/Solar)")
    fig.colorbar(im, label="Spearman rho")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "e11_v2_domain_correlation_heatmap.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {fig_path}")
    print(f"Saved {os.path.join(OUT_DIR, 'e11v2_lodo_recomputed.csv')}")
    print(f"Saved {os.path.join(OUT_DIR, 'e11v2_domain_correlation.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--domains", default="ETTm1,ECL,Solar")
    args = ap.parse_args()
    if args.mode == "run":
        run_domains(args.domains.split(","))
    else:
        analyze()
