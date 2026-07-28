"""V3 router: unified 21-method pool (19 deep experts + MSTL + LEAR) under
chronological split.

Per (market, seed) block (resume-capable, results/e6_v3/blocks/{market}_{seed}.npz):
  - MetaMorph/STOG-Router: probe HistGBM per pool member + lambda * static val
    prior (lambda grid on val), TopK=3 softmax fusion
  - Selection frequency: TopK membership counts on test windows
  - B8 FFORMA-lite (12 features -> 21-member error, HistGBM)
  - per-window SE/MAE arrays for router, B1-deep, B1-any, LEAR, MSTL, naive24
  - probe->rank Spearman (21 pool): probe-only / dual / FFORMA

Aggregation (runs every time, tolerates missing blocks):
  results/e6_v3/routing_v3_main.csv      method x market x seed test MSE
  results/e6_v3/routing_v3_summary.csv   per-market 5-seed mean +/- std + ALL
  results/e6_v3/router_selection_frequency.csv
  results/e6_v3/probe_rank_spearman_v3.csv
  results/e6_v3/tost_block_v3.csv        block-bootstrap TOST (block=168, B=10000)
  results/e6_v3/mase_cohensd.csv         MASE (naive-24h denom) + Cohen's d
  results/figures/e6_v3_router_selection_frequency.png
  results/figures/e6_v3_methods_comparison_21pool.png
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experts.zoo import get_all_cards

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407, 7, 12345]
DEEP_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
            "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
POOL = DEEP_IDS + ["MSTL", "LEAR"]
E = len(POOL)  # 21
PRED_DIR = "./results/preds_v3"
OUT_DIR = "./results/e6_v3"
BLK_DIR = "./results/e6_v3/blocks"
FIG_DIR = "./results/figures"

PROBE_IDX = [7, 10, 3, 11, 8]  # spec_decay, cond, kurt, regime_overlap, season_strength

CARDS = get_all_cards()
CARD_PRIOR = np.array([np.mean(list(CARDS[e].spectral_affinity.values()))
                       if e in CARDS else 0.5 for e in POOL])


def load_block(market, seed):
    meta = np.load(f"{PRED_DIR}/meta_{market}.npz")
    val_true, test_true = meta["val_true"], meta["test_true"]
    nv, H = val_true.shape
    nt = test_true.shape[0]
    val_pred = np.empty((E, nv, H), np.float32)
    test_pred = np.empty((E, nt, H), np.float32)
    train_err = []
    for i, eid in enumerate(POOL):
        if eid in ("MSTL", "LEAR"):
            d = np.load(f"{PRED_DIR}/{market}_{eid}.npz")
        else:
            d = np.load(f"{PRED_DIR}/{market}_{eid}_{seed}.npz")
        val_pred[i] = d["val_pred"]
        test_pred[i] = d["test_pred"]
        train_err.append(d["train_err"])
    train_err = np.stack(train_err, axis=1)
    naive = np.load(f"{PRED_DIR}/{market}_naive24.npz")
    return meta, val_pred, test_pred, val_true, test_true, train_err, naive


def fuse_topk(test_pred, err_hat, K, weighted=True):
    nt = err_hat.shape[0]
    topk = np.argsort(err_hat, axis=1)[:, :K]
    if weighted:
        s = -np.take_along_axis(err_hat, topk, axis=1)
        s = s - s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
    else:
        w = np.full((nt, K), 1.0 / K, dtype=np.float32)
    preds = np.transpose(test_pred, (1, 0, 2))[np.arange(nt)[:, None], topk]
    return (preds * w[:, :, None]).sum(axis=1), topk


def block_complete(market, seed):
    if not os.path.exists(f"{PRED_DIR}/meta_{market}.npz"):
        return False
    return all(os.path.exists(f"{PRED_DIR}/{market}_{e}_{seed}.npz") for e in DEEP_IDS)


def compute_block(market, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    meta, val_pred, test_pred, val_true, test_true, train_err, naive = load_block(market, seed)
    val_err = ((val_pred - val_true[None]) ** 2).mean(axis=2)
    test_err = ((test_pred - test_true[None]) ** 2).mean(axis=2)
    val_mse = val_err.mean(axis=1)
    nt = test_true.shape[0]

    # --- router scorer: probe -> per-member relative log err (HistGBM x21) ---
    Xtr_full = np.concatenate([meta["feat_train"][:, PROBE_IDX],
                               meta["feat_val"][:, PROBE_IDX]], axis=0)
    mu, sd = Xtr_full.mean(axis=0), Xtr_full.std(axis=0) + 1e-8
    Xtr_full = (Xtr_full - mu) / sd
    Ytr_full = np.log(np.concatenate([train_err, val_err.T], axis=0) + 1e-8)
    Ytr_full = Ytr_full - Ytr_full.mean(axis=1, keepdims=True)
    rng_sub = np.random.RandomState(0)
    sub_i = rng_sub.choice(Xtr_full.shape[0],
                           size=min(20000, Xtr_full.shape[0]), replace=False)
    sc_models = []
    for e in range(E):
        m = HistGradientBoostingRegressor(max_iter=120, max_depth=6,
                                          learning_rate=0.08, random_state=0)
        m.fit(Xtr_full[sub_i], Ytr_full[sub_i, e])
        sc_models.append(m)
    Xte = (meta["feat_test"][:, PROBE_IDX] - mu) / sd
    err_hat_dyn = np.stack([m.predict(Xte) for m in sc_models], axis=1)

    # --- dual score: dynamic + lambda * static val prior (lambda on val) ---
    Xva = (meta["feat_val"][:, PROBE_IDX] - mu) / sd
    err_hat_val = np.stack([m.predict(Xva) for m in sc_models], axis=1)
    z_static = (np.log(val_mse) - np.log(val_mse).mean()) / (np.log(val_mse).std() + 1e-8)
    best_lam, best_vmse = 0.0, np.inf
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        pv, _ = fuse_topk(val_pred, err_hat_val + lam * z_static[None, :], K=3)
        mv = float(((pv - val_true) ** 2).mean())
        if mv < best_vmse:
            best_vmse, best_lam = mv, lam
    err_hat = err_hat_dyn + best_lam * z_static[None, :]

    def mse_of(pred):
        return float(((pred - test_true) ** 2).mean())

    pred_full, topk = fuse_topk(test_pred, err_hat, K=3, weighted=True)
    res = {"R_full": mse_of(pred_full)}

    # --- baselines (21 pool) ---
    e_deep = int(np.argmin(val_mse[:len(DEEP_IDS)]))
    e_any = int(np.argmin(val_mse))
    res["B1_deep"] = float(test_err[e_deep].mean())
    res["B1_any"] = float(test_err[e_any].mean())
    res["B1_any_expert"] = POOL[e_any]
    res["B3_avg_ensemble"] = mse_of(test_pred.mean(axis=0))
    w4 = (1.0 / val_mse) / (1.0 / val_mse).sum()
    res["B4_val_weighted"] = mse_of(np.tensordot(w4, test_pred, axes=(0, 0)))
    top3v = np.argsort(val_mse)[:3]
    res["B5_static_top3"] = mse_of(test_pred[top3v].mean(axis=0))
    rng7 = np.random.RandomState(1234)
    res["B7_random_top3"] = float(np.mean([
        mse_of(test_pred[rng7.choice(E, size=3, replace=False)].mean(axis=0))
        for _ in range(20)]))
    res["B20_oracle"] = float(test_err.min(axis=0).mean())
    res["LEAR"] = float(test_err[E - 1].mean())
    res["MSTL"] = float(test_err[E - 2].mean())

    # --- B8 FFORMA-lite (12 feat -> 21 member err, HistGBM) ---
    X8 = np.concatenate([meta["feat_train"], meta["feat_val"]], axis=0)
    Y8 = np.log(np.concatenate([train_err, val_err.T], axis=0) + 1e-8)
    sub8 = np.random.RandomState(0).choice(X8.shape[0], size=min(8000, X8.shape[0]), replace=False)
    ff_models = []
    for e in range(E):
        m = HistGradientBoostingRegressor(max_iter=80, max_depth=6,
                                          learning_rate=0.1, random_state=0)
        m.fit(X8[sub8], Y8[sub8, e])
        ff_models.append(m)
    err_hat_ff = np.stack([m.predict(meta["feat_test"]) for m in ff_models], axis=1)
    w8 = np.exp(-err_hat_ff - (-err_hat_ff).max(axis=1, keepdims=True))
    w8 /= w8.sum(axis=1, keepdims=True)
    res["B8_fforma_lite"] = mse_of(np.einsum("ne,enh->nh", w8, test_pred))

    # --- probe->rank Spearman (21 pool) ---
    rng_sp = np.random.RandomState(7)
    sub = rng_sp.choice(nt, size=min(2000, nt), replace=False)
    rs_p, rs_d, rs_f = [], [], []
    for i in sub:
        rs_p.append(spearmanr(err_hat_dyn[i], test_err[:, i]).statistic)
        rs_d.append(spearmanr(err_hat[i], test_err[:, i]).statistic)
        rs_f.append(spearmanr(err_hat_ff[i], test_err[:, i]).statistic)
    spear = {"market": market, "seed": seed,
             "spearman_probe_only": float(np.nanmean(rs_p)),
             "spearman_router": float(np.nanmean(rs_d)),
             "spearman_fforma": float(np.nanmean(rs_f))}

    # --- per-window arrays for block-bootstrap stats ---
    se = lambda p: ((p - test_true) ** 2).mean(axis=1)
    mae = lambda p: np.abs(p - test_true).mean(axis=1)
    pw = {
        "se_router": se(pred_full), "mae_router": mae(pred_full),
        "se_b1deep": test_err[e_deep], "se_b1any": test_err[e_any],
        "se_lear": test_err[E - 1], "se_mstl": test_err[E - 2],
        "mae_lear": np.abs(test_pred[E - 1] - test_true).mean(axis=1),
        "mae_mstl": np.abs(test_pred[E - 2] - test_true).mean(axis=1),
        "mae_b1any": np.abs(test_pred[e_any] - test_true).mean(axis=1),
        "mae_naive": np.abs(naive["test_pred"] - test_true).mean(axis=1),
    }

    np.savez(f"{BLK_DIR}/{market}_{seed}.npz",
             topk=topk.astype(np.int16),
             test_err=test_err.astype(np.float32),
             **{k: v.astype(np.float32) for k, v in pw.items()})

    print(f"[{market}/{seed}] R_full={res['R_full']:.3f} B1deep={res['B1_deep']:.3f} "
          f"B1any={res['B1_any']:.3f}({res['B1_any_expert']}) LEAR={res['LEAR']:.3f} "
          f"MSTL={res['MSTL']:.3f} B8={res['B8_fforma_lite']:.3f} oracle={res['B20_oracle']:.3f} "
          f"lam={best_lam} spear={spear['spearman_router']:.3f} ({time.time()-t0:.0f}s)",
          flush=True)
    return res, spear


# ---------------- aggregation ----------------
def block_boot(bm, B=10000, seed=0):
    """bm: block means. Returns percentiles [2.5, 5, 95, 97.5] of resampled mean."""
    rng = np.random.RandomState(seed)
    nb = len(bm)
    idx = rng.randint(0, nb, size=(B, nb))
    means = bm[idx].mean(axis=1)
    return np.percentile(means, [2.5, 5, 95, 97.5])


def aggregate():
    main_csv = f"{OUT_DIR}/routing_v3_main.csv"
    blocks = {}
    for market in MARKETS:
        for seed in SEEDS:
            p = f"{BLK_DIR}/{market}_{seed}.npz"
            if os.path.exists(p):
                blocks[(market, seed)] = np.load(p)
    done = {(r.market, int(r.seed)) for r in
            pd.read_csv(main_csv).itertuples()} if os.path.exists(main_csv) else set()
    print(f"[aggregate] {len(blocks)}/25 blocks present, {len(done)} already in main csv")

    # ---- selection frequency + per-block stats ----
    sel_freq, tost_rows, mase_rows, summ_src = [], [], [], []
    naive_mae = {}
    for market in MARKETS:
        mblocks = [(s, blocks[(market, s)]) for s in SEEDS if (market, s) in blocks]
        if not mblocks:
            continue
        # selection frequency
        freqs = []
        for s, b in mblocks:
            tk = b["topk"]
            f = np.zeros(E)
            for e in range(E):
                f[e] = float((tk == e).mean())
            freqs.append(f)
        fm = np.mean(freqs, axis=0)
        fs = np.std(freqs, axis=0)
        for e, eid in enumerate(POOL):
            sel_freq.append({"market": market, "expert": eid,
                             "select_freq": fm[e], "std_across_seeds": fs[e],
                             "is_stat_baseline": eid in ("MSTL", "LEAR")})
        naive_mae[market] = float(np.mean([b["mae_naive"].mean() for _, b in mblocks]))

        # ---- block bootstrap TOST (pool block means across seeds) ----
        for comp, a_key, b_key in [("router_vs_B1any", "se_router", "se_b1any"),
                                   ("router_vs_LEAR", "se_router", "se_lear")]:
            bm, diffs_all, base_mse = [], [], []
            for s, b in mblocks:
                d = (b[a_key] - b[b_key]).astype(np.float64)
                nb = len(d) // 168
                bm.append(d[:nb * 168].reshape(nb, 168).mean(axis=1))
                diffs_all.append(d)
                base_mse.append(b[b_key].mean())
            bm = np.concatenate(bm)
            d_all = np.concatenate(diffs_all)
            q = block_boot(bm, seed=hash((market, comp)) % 2**31)
            md = float(d_all.mean())
            cohens = md / float(d_all.std(ddof=1))
            delta = 0.01 * float(np.mean(base_mse))
            equiv = bool((q[1] > -delta) and (q[2] < delta))
            tost_rows.append({"comparison": comp, "market": market,
                              "n_windows": int(d_all.shape[0]), "n_blocks": int(len(bm)),
                              "mean_diff": md, "ci95_lo": float(q[0]), "ci95_hi": float(q[3]),
                              "ci90_lo": float(q[1]), "ci90_hi": float(q[2]),
                              "margin": delta, "equivalent_1pct": equiv,
                              "cohens_d": float(cohens)})

        # ---- MASE (naive-24h denominator) ----
        for meth, key in [("R_full", "mae_router"), ("B1_any", "mae_b1any"),
                          ("LEAR", "mae_lear"), ("MSTL", "mae_mstl")]:
            maes = [float(b[key].mean()) for _, b in mblocks]
            mase_rows.append({"market": market, "method": meth,
                              "mae_mean": float(np.mean(maes)),
                              "mae_std_seeds": float(np.std(maes)),
                              "mase": float(np.mean(maes) / naive_mae[market])})
    pd.DataFrame(sel_freq).to_csv(f"{OUT_DIR}/router_selection_frequency.csv", index=False)

    # ---- summary (requires main csv) ----
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        rows = []
        for meth in df["method"].unique():
            sub = df[df["method"] == meth]
            rec = {"method": meth}
            pm_means, pm_vars = [], []
            for market in MARKETS:
                sm = sub[sub["market"] == market]["test_mse"]
                if len(sm):
                    rec[f"{market}_mean"] = float(sm.mean())
                    rec[f"{market}_std"] = float(sm.std())
                    pm_means.append(sm.mean()); pm_vars.append(sm.var())
            rec["ALL_mean"] = float(np.mean(pm_means))
            rec["ALL_seed_std"] = float(np.sqrt(np.mean(pm_vars)))
            rows.append(rec)
        summ = pd.DataFrame(rows).sort_values("ALL_mean")
        summ.to_csv(f"{OUT_DIR}/routing_v3_summary.csv", index=False)
        print("\n=== routing_v3_summary ===")
        print(summ.to_string(index=False))

    pd.DataFrame(tost_rows).to_csv(f"{OUT_DIR}/tost_block_v3.csv", index=False)
    pd.DataFrame(mase_rows).to_csv(f"{OUT_DIR}/mase_cohensd.csv", index=False)
    print("\n=== block-bootstrap TOST (block=168h, B=10000) ===")
    print(pd.DataFrame(tost_rows).to_string(index=False))

    # ---- figures ----
    make_figures(pd.DataFrame(sel_freq))


def make_figures(sel_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- selection frequency stacked bars ---
    if len(sel_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.tab20(np.linspace(0, 1, E))
        colors = {eid: cmap[i] for i, eid in enumerate(POOL)}
        colors["MSTL"] = "#d62728"
        colors["LEAR"] = "#ff7f0e"
        bottom = np.zeros(len(sel_df["market"].unique()))
        markets = list(sel_df["market"].unique())
        # plot deep experts first, stat baselines last (on top)
        for eid in DEEP_IDS + ["MSTL", "LEAR"]:
            vals = [float(sel_df[(sel_df.market == m) & (sel_df.expert == eid)]["select_freq"].iloc[0])
                    for m in markets]
            ax.bar(markets, vals, bottom=bottom, color=colors[eid], label=eid,
                   edgecolor="white", linewidth=0.3)
            bottom += np.array(vals)
        ax.set_ylabel("TopK=3 selection frequency (mean over 5 seeds)")
        ax.set_title("E6 v3 (chronological split): STOG-Router selection frequency, 21-method pool\n"
                     "MSTL (red) / LEAR (orange) — statistical baselines in the routing pool")
        ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/e6_v3_router_selection_frequency.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- methods comparison (21 pool) ---
    main_csv = f"{OUT_DIR}/routing_v3_main.csv"
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        methods = ["R_full", "B1_deep", "B1_any", "B3_avg_ensemble", "B4_val_weighted",
                   "B5_static_top3", "B7_random_top3", "B8_fforma_lite", "LEAR",
                   "MSTL", "B20_oracle"]
        fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
        for ax, market in zip(axes, MARKETS):
            sub = df[df.market == market]
            means = [sub[sub.method == m]["test_mse"].mean() for m in methods]
            stds = [sub[sub.method == m]["test_mse"].std() for m in methods]
            cols = ["#2ca02c" if m == "R_full" else "#d62728" if m == "MSTL"
                    else "#ff7f0e" if m == "LEAR" else "#1f77b4" if "oracle" in m
                    else "#9ecae1" for m in methods]
            ax.bar(range(len(methods)), means, yerr=stds, color=cols, capsize=2)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace("_", "\n") for m in methods],
                               rotation=90, fontsize=6)
            ax.set_title(market)
            ax.set_ylabel("test MSE" if market == "NP" else "")
        fig.suptitle("E6 v3 (chronological): router vs baselines, 21-method pool "
                     "(5-seed mean ± std)")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/e6_v3_methods_comparison_21pool.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print("figures saved")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=float, default=1e9)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BLK_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    main_csv = f"{OUT_DIR}/routing_v3_main.csv"
    spear_csv = f"{OUT_DIR}/probe_rank_spearman_v3.csv"
    done = set()
    if os.path.exists(main_csv):
        done = {(r.market, int(r.seed)) for r in pd.read_csv(main_csv).itertuples()}

    for seed in SEEDS:
        for market in MARKETS:
            if (market, seed) in done:
                continue
            if time.time() - t_start > args.time_budget:
                print("[budget] stop; rerun to resume", flush=True)
                aggregate()
                return
            if os.path.exists(f"{BLK_DIR}/{market}_{seed}.npz"):
                # block npz exists but main row missing -> recompute cheap? just redo
                os.remove(f"{BLK_DIR}/{market}_{seed}.npz")
            if not block_complete(market, seed):
                print(f"[{market}/{seed}] preds incomplete, skip for now", flush=True)
                continue
            res, spear = compute_block(market, seed)
            rows = [{"method": k, "market": market, "seed": seed, "test_mse": v}
                    for k, v in res.items() if not k.endswith("_expert")]
            rows.append({"method": "B1_any_expert", "market": market, "seed": seed,
                         "test_mse": np.nan, "expert": res["B1_any_expert"]})
            pd.DataFrame(rows).to_csv(main_csv, mode="a", index=False,
                                      header=not os.path.exists(main_csv))
            pd.DataFrame([spear]).to_csv(spear_csv, mode="a", index=False,
                                         header=not os.path.exists(spear_csv))

    aggregate()


if __name__ == "__main__":
    main()
