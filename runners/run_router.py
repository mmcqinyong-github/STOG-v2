"""MetaMorph-Lite router: probe features -> per-expert ridge scores -> TopK fusion.

Router trained per (market, seed) on train+val windows (no future info: probe
features use only the input window x). Evaluation on test windows using saved
expert predictions.

Variants / ablations:
  R_full            probe -> ridge -> TopK(3) softmax-weighted fusion
  R_K1 / R_K5       K sweep
  R_top3_mean       K=3 equal-weight fusion
  R_no_probe        static val-weighted (= B4, re-computed here for ablation table)
  R_card_only       fixed prior from genome-card mean spectral affinity (no data)

Metrics: test MSE per variant; probe->rank Spearman (router predicted expert
ranking vs realized test error ranking, mean over sampled windows); TOST
equivalence vs market champion (B1 expert), margin 1%.

Outputs (results/e6_routing/):
  routing_main.csv       all methods x market x seed test MSE (router variants
                         + key baselines B1/B4/B8/B20 merged for convenience)
  routing_summary.csv    method-level 5-market summary (mean/std/rank/improvement vs B1)
  routing_ablation.csv   router variant ablation table
  probe_rank_spearman.csv  per market x seed Spearman (router vs FFORMA-lite)
  tost_champion.csv      TOST results per market + pooled
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.experts.zoo import EXPERT_REGISTRY, get_all_cards

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
SEEDS = [2021, 42, 3407]
# E6 canonical 19 experts (same as run_e4_e6.py)
EXPERT_IDS = ["M01", "M03", "M14", "M17", "M18", "M31", "M47", "M50", "M52",
              "M55", "M63", "M89", "M117", "M220", "M233", "N01", "N07", "N08", "N10"]
E = len(EXPERT_IDS)
PRED_DIR = "./results/preds"
OUT_DIR = "./results/e6_routing"

# probe feature subset (indices into FEAT_NAMES in run_e6_preds.py)
# [mean,std,skew,kurt,acf1,acf24,spec_centroid,spec_decay,season_strength,
#  trend_slope,cond_number,regime_overlap]
PROBE_IDX = [7, 10, 3, 11, 8]  # spec_decay(alpha), cond(kappa), kurt(gamma),
                               # regime_overlap(delta), season_strength(s)

# card-only fixed prior: mean spectral affinity per expert
CARDS = get_all_cards()
CARD_PRIOR = np.array([np.mean(list(CARDS[e].spectral_affinity.values()))
                       if e in CARDS else 0.5 for e in EXPERT_IDS])


def load_block(market, seed):
    meta = np.load(f"{PRED_DIR}/meta_{market}_{seed}.npz")
    val_true, test_true = meta["val_true"], meta["test_true"]
    nv, H = val_true.shape
    nt = test_true.shape[0]
    val_pred = np.empty((E, nv, H), np.float32)
    test_pred = np.empty((E, nt, H), np.float32)
    train_err = []
    for i, eid in enumerate(EXPERT_IDS):
        d = np.load(f"{PRED_DIR}/{market}_{eid}_{seed}.npz")
        val_pred[i] = d["val_pred"]
        test_pred[i] = d["test_pred"]
        train_err.append(d["train_err"])
    train_err = np.stack(train_err, axis=1)
    return meta, val_pred, test_pred, val_true, test_true, train_err


def fuse_topk(test_pred, err_hat, K, weighted=True):
    """err_hat: (nt, E) predicted log-error. Returns fused (nt, H)."""
    nt = err_hat.shape[0]
    topk = np.argsort(err_hat, axis=1)[:, :K]                    # (nt, K)
    if weighted:
        s = -np.take_along_axis(err_hat, topk, axis=1)           # (nt, K)
        s = s - s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
    else:
        w = np.full((nt, K), 1.0 / K, dtype=np.float32)
    # gather predictions: test_pred is (E, nt, H) -> want (nt, K, H)
    preds = np.transpose(test_pred, (1, 0, 2))[np.arange(nt)[:, None], topk]  # (nt, K, H)
    return (preds * w[:, :, None]).sum(axis=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    router_rows, spear_rows, tost_rows = [], [], []
    base_df = pd.read_csv(f"{OUT_DIR}/baselines.csv")
    rng_spear = np.random.RandomState(7)

    # for TOST: per-market per-window diffs (router - champion), pooled over seeds
    tost_diffs = {m: [] for m in MARKETS}
    tost_champ_mse = {m: [] for m in MARKETS}

    for market in MARKETS:
        for seed in SEEDS:
            t0 = time.time()
            meta, val_pred, test_pred, val_true, test_true, train_err = load_block(market, seed)
            val_err = ((val_pred - val_true[None]) ** 2).mean(axis=2)     # (E, nv)
            test_err = ((test_pred - test_true[None]) ** 2).mean(axis=2)  # (E, nt)
            val_mse = val_err.mean(axis=1)
            e_star = int(np.argmin(val_mse))
            nt = test_true.shape[0]

            # --- train router scorer (small MLP on probe -> per-expert log err) ---
            Xtr_full = np.concatenate([meta["feat_train"][:, PROBE_IDX],
                                       meta["feat_val"][:, PROBE_IDX]], axis=0)
            mu, sd = Xtr_full.mean(axis=0), Xtr_full.std(axis=0) + 1e-8
            Xtr_full = (Xtr_full - mu) / sd
            Ytr_full = np.log(np.concatenate([train_err, val_err.T], axis=0) + 1e-8)
            # relative log-error target: subtract per-window mean across experts so
            # the scorer learns expert RANKING, not (unpredictable) window difficulty
            Ytr_full = Ytr_full - Ytr_full.mean(axis=1, keepdims=True)
            rng_sub = np.random.RandomState(0)
            sub_i = rng_sub.choice(Xtr_full.shape[0],
                                   size=min(20000, Xtr_full.shape[0]), replace=False)
            # per-expert HistGBM scorer (nonlinear, fast on CPU)
            from sklearn.ensemble import HistGradientBoostingRegressor
            sc_models = []
            for e in range(E):
                m = HistGradientBoostingRegressor(max_iter=120, max_depth=6,
                                                  learning_rate=0.08, random_state=0)
                m.fit(Xtr_full[sub_i], Ytr_full[sub_i, e])
                sc_models.append(m)
            Xte = (meta["feat_test"][:, PROBE_IDX] - mu) / sd
            err_hat = np.stack([m.predict(Xte) for m in sc_models], axis=1)
            # ridge variant for ablation
            ridge = Ridge(alpha=10.0)
            ridge.fit(Xtr_full[sub_i], Ytr_full[sub_i])
            err_hat_ridge = ridge.predict(Xte)

            # --- dual-score: dynamic probe score + lambda * static val prior ---
            # lambda selected on VALIDATION windows (grid), no test leakage
            Xva = (meta["feat_val"][:, PROBE_IDX] - mu) / sd
            err_hat_val = np.stack([m.predict(Xva) for m in sc_models], axis=1)
            z_static = (np.log(val_mse) - np.log(val_mse).mean()) / (np.log(val_mse).std() + 1e-8)
            best_lam, best_vmse = 0.0, np.inf
            for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
                pv = fuse_topk(val_pred, err_hat_val + lam * z_static[None, :],
                               K=3, weighted=True)
                mv = float(((pv - val_true) ** 2).mean())
                if mv < best_vmse:
                    best_vmse, best_lam = mv, lam
            err_hat_dyn = err_hat.copy()
            err_hat = err_hat + best_lam * z_static[None, :]

            def mse_of(pred):
                return float(((pred - test_true) ** 2).mean())

            res = {}
            res["R_full"] = mse_of(fuse_topk(test_pred, err_hat, K=3, weighted=True))
            res["R_dyn_only"] = mse_of(fuse_topk(test_pred, err_hat_dyn, K=3, weighted=True))
            res["R_K1"] = mse_of(fuse_topk(test_pred, err_hat, K=1, weighted=True))
            res["R_K5"] = mse_of(fuse_topk(test_pred, err_hat, K=5, weighted=True))
            res["R_top3_mean"] = mse_of(fuse_topk(test_pred, err_hat, K=3, weighted=False))
            res["R_ridge"] = mse_of(fuse_topk(test_pred, err_hat_ridge, K=3, weighted=True))
            # no-probe: static inverse-val-MSE weights (B4 equivalent)
            w4 = (1.0 / val_mse) / (1.0 / val_mse).sum()
            res["R_no_probe"] = mse_of(np.tensordot(w4, test_pred, axes=(0, 0)))
            # card-only: fixed prior TopK=3 equal weight
            top3_card = np.argsort(-CARD_PRIOR)[:3]
            res["R_card_only"] = mse_of(test_pred[top3_card].mean(axis=0))
            res["lambda_static"] = best_lam

            for k, v in res.items():
                if k == "lambda_static":
                    continue
                router_rows.append({"method": k, "market": market, "seed": seed,
                                    "test_mse": v})

            # --- probe->rank Spearman (router vs FFORMA-lite) ---
            sub = rng_spear.choice(nt, size=min(2000, nt), replace=False)
            rs_probe, rs_dual, rs_ff = [], [], []
            ff = np.load(f"{OUT_DIR}/fforma_errhat_{market}_{seed}.npz")["err_hat"]
            for i in sub:
                rs_probe.append(spearmanr(err_hat_dyn[i], test_err[:, i]).statistic)
                rs_dual.append(spearmanr(err_hat[i], test_err[:, i]).statistic)
                rs_ff.append(spearmanr(ff[i], test_err[:, i]).statistic)
            spear_rows.append({"market": market, "seed": seed,
                               "spearman_probe_only": float(np.nanmean(rs_probe)),
                               "spearman_router": float(np.nanmean(rs_dual)),
                               "spearman_fforma": float(np.nanmean(rs_ff))})

            # --- collect TOST diffs (router per-window se - champion per-window se) ---
            pred_full = fuse_topk(test_pred, err_hat, K=3, weighted=True)
            se_router = ((pred_full - test_true) ** 2).mean(axis=1)
            se_champ = test_err[e_star]
            tost_diffs[market].append(se_router - se_champ)
            tost_champ_mse[market].append(float(se_champ.mean()))

            print(f"[{market}/{seed}] R_full={res['R_full']:.3f} K1={res['R_K1']:.3f} "
                  f"K5={res['R_K5']:.3f} mean={res['R_top3_mean']:.3f} "
                  f"noprobe={res['R_no_probe']:.3f} card={res['R_card_only']:.3f} "
                  f"spearP={np.nanmean(rs_probe):.3f} spearB8={np.nanmean(rs_ff):.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # ---------- TOST vs champion (margin = 1% of champion mean SE) ----------
    for market in MARKETS:
        d = np.concatenate(tost_diffs[market])
        delta = 0.01 * float(np.mean(tost_champ_mse[market]))
        md, sd_d, n = float(d.mean()), float(d.std(ddof=1)), d.shape[0]
        ci = 1.645 * sd_d / np.sqrt(n)
        equiv = (md - ci > -delta) and (md + ci < delta)
        tost_rows.append({"market": market, "n": n, "mean_diff": md,
                          "ci90_half": ci, "margin": delta, "equivalent": bool(equiv)})
    d = np.concatenate([np.concatenate(v) for v in tost_diffs.values()])
    delta = 0.01 * float(np.mean([x for v in tost_champ_mse.values() for x in v]))
    md, sd_d, n = float(d.mean()), float(d.std(ddof=1)), d.shape[0]
    ci = 1.645 * sd_d / np.sqrt(n)
    tost_rows.append({"market": "ALL", "n": n, "mean_diff": md, "ci90_half": ci,
                      "margin": delta,
                      "equivalent": bool((md - ci > -delta) and (md + ci < delta))})

    # ---------- merge & save ----------
    router_df = pd.DataFrame(router_rows)
    router_df.to_csv(f"{OUT_DIR}/routing_ablation.csv", index=False)
    keep = base_df[base_df["method"].isin(
        ["B1_best_single", "B3_avg_ensemble", "B4_val_weighted", "B5_static_top3",
         "B7_random_top3", "B8_fforma_lite", "B20_oracle"])]
    main_df = pd.concat([router_df[router_df["method"] == "R_full"], keep],
                        ignore_index=True)
    main_df.to_csv(f"{OUT_DIR}/routing_main.csv", index=False)
    pd.DataFrame(spear_rows).to_csv(f"{OUT_DIR}/probe_rank_spearman.csv", index=False)
    tost_df = pd.DataFrame(tost_rows)
    tost_df.to_csv(f"{OUT_DIR}/tost_champion.csv", index=False)

    # ---------- summary ----------
    summ_rows = []
    piv = main_df.pivot_table(index=["market", "seed"], columns="method",
                              values="test_mse")
    ranks = piv.rank(axis=1).mean()
    b1 = main_df[main_df.method == "B1_best_single"].groupby("market")["test_mse"].mean()
    for meth in main_df["method"].unique():
        sub = main_df[main_df.method == meth]
        per_market = sub.groupby("market")["test_mse"].mean()
        summ_rows.append({
            "method": meth,
            "mean_test_mse": float(per_market.mean()),
            "std_across_markets": float(per_market.std()),
            "mean_rank": float(ranks[meth]),
            "improvement_vs_B1_pct": float((1 - per_market.mean() / b1.mean()) * 100),
        })
    summ_df = pd.DataFrame(summ_rows).sort_values("mean_test_mse")
    summ_df.to_csv(f"{OUT_DIR}/routing_summary.csv", index=False)
    print("\n=== routing_summary ===")
    print(summ_df.to_string(index=False))
    print("\n=== TOST vs champion (margin 1%) ===")
    print(tost_df.to_string(index=False))


if __name__ == "__main__":
    main()
