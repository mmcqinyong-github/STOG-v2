"""E5 manifold embedding test (Theorem 5a).

Uses ALL 19 experts' saved test point predictions (results/preds/) on all
5 EPF markets x 3 seeds.

Pipeline
  1) Expert distance matrix per (market,seed): d_ij = sqrt(2*(1-corr_ij))
     between flattened test prediction vectors.
  2) 2D embedding: PHATE if available, else sklearn MDS. Embeddings across
     the 15 (market,seed) samples are aligned by generalized orthogonal
     Procrustes (GPA) and pooled (285 points).
  3) Tests:
     (a) family clustering: silhouette of pooled coords by genome family,
         permutation p-value (label shuffles); per-sample silhouette too.
     (b) probe-coordinate alignment: per test window, best expert (min MSE);
         window inherits that expert's aligned coord. CCA between window
         environment probes (spec_decay=alpha_hat, cond_number=kappa_hat,
         kurt=gamma_hat, season_strength=s_hat; z-scored, kappa log) and
         inherited coords; canonical corrs + permutation p.
     (c) local smoothness: pairwise probe distance vs optimal-expert switch
         rate; Spearman corr over distance deciles + nearest-decile switch
         rate.

Outputs:
  results/e5/e5_manifold.csv
  results/e5/e5_embedding_coords.npz  (for figures)
Resume-capable: skips (market,seed) rows already in e5_manifold.csv.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd

from src.utils.common import ensure_dir
from src.experts.zoo import EXPERT_REGISTRY, get_all_cards

ensure_dir("./results/e5")

EXPERT_IDS = list(EXPERT_REGISTRY.keys())  # 19
CARDS = get_all_cards()
FAMILY = {mid: CARDS[mid].family for mid in EXPERT_IDS}
# Coarse 9-family grouping (linear/frequency/ssm/cnn/decomposition/attention/
# graph/hybrid/statistical) as specified for the family-clustering test.
COARSE_MAP = {
    "linear": "linear", "frequency": "frequency", "wavelet": "frequency",
    "periodic": "frequency", "ssm": "ssm", "cnn": "cnn",
    "decomposition": "decomposition", "basis_expansion": "decomposition",
    "attention": "attention", "graph": "graph", "hybrid": "hybrid",
    "statistical": "statistical",
}
COARSE = {mid: COARSE_MAP[FAMILY[mid]] for mid in EXPERT_IDS}
PROBE_IDX = {"alpha_hat": 7, "gamma_hat": 3, "kappa_hat": 10, "s_hat": 8}
RNG = np.random.RandomState(0)


def embed_2d(D):
    """D: (E,E) dissimilarity -> (E,2). PHATE preferred, MDS fallback."""
    try:
        import phate
        op = phate.PHATE(n_components=2, knn=min(5, D.shape[0] - 2),
                         decay=10, t="auto", random_state=0, verbose=False)
        return op.fit_transform(D)
    except Exception as ex:
        print(f"  phate failed ({type(ex).__name__}: {ex}); using MDS", flush=True)
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=0, n_init=4, max_iter=600)
        return mds.fit_transform(D)


def procrustes_to(X, Y):
    """Orthogonal Procrustes: rotate/reflect X onto Y (both centered)."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    M = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return Xc @ R * (np.linalg.norm(Yc) / (np.linalg.norm(Xc) + 1e-12))


def gpa_align(embeds, iters=10):
    """Generalized Procrustes alignment of a list of (E,2) embeddings."""
    Z = [e - e.mean(0) for e in embeds]
    Z = [z / (np.linalg.norm(z) + 1e-12) for z in Z]
    cons = np.mean(Z, axis=0)
    for _ in range(iters):
        Z = [procrustes_to(z, cons) for z in Z]
        cons = np.mean(Z, axis=0)
    return Z, cons


def silhouette_p(coords, labels, n_perm=999):
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    obs = silhouette_score(coords, labels)
    cnt = 0
    for _ in range(n_perm):
        cnt += silhouette_score(coords, RNG.permutation(labels)) >= obs
    return obs, (1 + cnt) / (n_perm + 1)


def cca_probe_coord(X, Y, n_perm=499):
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=2)
    Xc, Yc = cca.fit_transform(X, Y)
    r = [float(np.corrcoef(Xc[:, k], Yc[:, k])[0, 1]) for k in range(2)]
    cnt = 0
    for _ in range(n_perm):
        c2 = CCA(n_components=1)
        Xp, Yp = c2.fit_transform(X, RNG.permutation(Y))
        cnt += abs(np.corrcoef(Xp[:, 0], Yp[:, 0])[0, 1]) >= abs(r[0])
    return r, (1 + cnt) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NP,PJM,BE,FR,DE")
    ap.add_argument("--seeds", default="2021,42,3407")
    args = ap.parse_args()
    markets, seeds = args.markets.split(","), [int(s) for s in args.seeds.split(",")]

    out_csv = "./results/e5/e5_manifold.csv"
    done = set()
    if os.path.exists(out_csv):
        d0 = pd.read_csv(out_csv)
        done = set(zip(d0.market, d0.seed))

    rows, embeds, keys = [], [], []
    per_window = {}  # (market,seed) -> dict(probes, best_eid_idx)
    for market in markets:
        for seed in seeds:
            if (market, seed) in done and os.path.exists("./results/e5/e5_embedding_coords.npz"):
                continue
            t0 = time.time()
            meta = np.load(f"./results/preds/meta_{market}_{seed}.npz")
            yt = meta["test_true"].astype(np.float64)
            n = yt.shape[0]
            P = np.empty((len(EXPERT_IDS), n * 24))
            mse_mat = np.empty((n, len(EXPERT_IDS)))
            for i, eid in enumerate(EXPERT_IDS):
                p = np.load(f"./results/preds/{market}_{eid}_{seed}.npz")["test_pred"].astype(np.float64)
                P[i] = p.ravel()
                mse_mat[:, i] = ((p - yt) ** 2).mean(axis=1)
            # distance matrix
            C = np.corrcoef(P)
            C = np.clip(C, -1, 1)
            D = np.sqrt(np.maximum(2 * (1 - C), 0.0))
            np.fill_diagonal(D, 0.0)
            emb = embed_2d(D)
            embeds.append(emb); keys.append((market, seed))
            # environment probes
            F = meta["feat_test"].astype(np.float64)
            probes = np.stack([F[:, PROBE_IDX["alpha_hat"]],
                               np.log(np.clip(F[:, PROBE_IDX["kappa_hat"]], 1, None)),
                               F[:, PROBE_IDX["gamma_hat"]],
                               F[:, PROBE_IDX["s_hat"]]], axis=1)
            probes = (probes - probes.mean(0)) / (probes.std(0) + 1e-12)
            best = mse_mat.argmin(axis=1)
            per_window[(market, seed)] = {"probes": probes, "best": best}
            # per-sample silhouette (coarse 9-family labels)
            sil, _ = silhouette_p(emb, [COARSE[e] for e in EXPERT_IDS], n_perm=199)
            rows.append({"market": market, "seed": seed, "silhouette_own": sil})
            print(f"[{market}/{seed}] dist+embed+best {time.time()-t0:.1f}s sil={sil:.3f}", flush=True)

    # GPA alignment + pooled silhouette
    if embeds:
        Z, _ = gpa_align(embeds)
        all_coords, all_fam, all_fam_fine, all_mkt, all_eid = [], [], [], [], []
        for (market, seed), z in zip(keys, Z):
            all_coords.append(z)
            all_fam += [COARSE[e] for e in EXPERT_IDS]
            all_fam_fine += [FAMILY[e] for e in EXPERT_IDS]
            all_mkt += [market] * len(EXPERT_IDS)
            all_eid += EXPERT_IDS
            per_window[(market, seed)]["coord_map"] = z  # (E,2)
        all_coords = np.concatenate(all_coords)
        pooled_sil, pooled_p = silhouette_p(all_coords, all_fam, n_perm=999)
        fine_sil, fine_p = silhouette_p(all_coords, all_fam_fine, n_perm=999)
        print(f"POOLED silhouette(coarse)={pooled_sil:.4f} p={pooled_p:.4f} | "
              f"fine={fine_sil:.4f} p={fine_p:.4f}", flush=True)
        rows.append({"market": "POOLED", "seed": 0, "silhouette_own": pooled_sil,
                     "silhouette_p": pooled_p, "silhouette_fine": fine_sil,
                     "silhouette_fine_p": fine_p})
        np.savez("./results/e5/e5_embedding_coords.npz",
                 coords=all_coords, family=np.array(all_fam),
                 family_fine=np.array(all_fam_fine),
                 market=np.array(all_mkt), expert=np.array(all_eid))

        # (b) CCA and (c) smoothness per (market,seed)
        for r in rows:
            key = (r["market"], r["seed"])
            if key not in per_window or "coord_map" not in per_window[key]:
                continue
            pw = per_window[key]
            Y = pw["coord_map"][pw["best"]]  # (n,2)
            (r1, r2), p1 = cca_probe_coord(pw["probes"], Y)
            r["cca_r1"], r["cca_r2"], r["cca_p1"] = r1, r2, p1
            # smoothness on subsample
            sub = RNG.choice(pw["probes"].shape[0], size=min(1500, pw["probes"].shape[0]), replace=False)
            Xs = pw["probes"][sub]; bs = pw["best"][sub]
            Dp = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
            iu = np.triu_indices(len(sub), 1)
            dvec, sw = Dp[iu], (bs[iu[0]] != bs[iu[1]]).astype(float)
            qs = np.quantile(dvec, np.linspace(0, 1, 11))
            bin_id = np.clip(np.searchsorted(qs, dvec) - 1, 0, 9)
            rates = np.array([sw[bin_id == b].mean() for b in range(10)])
            dmeans = np.array([dvec[bin_id == b].mean() for b in range(10)])
            from scipy.stats import spearmanr
            rho, p_r = spearmanr(dmeans, rates)
            r["smooth_spearman"] = float(rho)
            r["smooth_spearman_p"] = float(p_r)
            r["smooth_nn_switch_rate"] = float(rates[0])
            r["smooth_far_switch_rate"] = float(rates[-1])
            # family-level smoothness: switch of the best expert's coarse family
            fam_arr = np.array([COARSE[e] for e in EXPERT_IDS])
            bf = fam_arr[bs]
            swf = (bf[iu[0]] != bf[iu[1]]).astype(float)
            rates_f = np.array([swf[bin_id == b].mean() for b in range(10)])
            rho_f, p_f = spearmanr(dmeans, rates_f)
            r["smooth_fam_spearman"] = float(rho_f)
            r["smooth_fam_nn_switch_rate"] = float(rates_f[0])
            r["smooth_fam_far_switch_rate"] = float(rates_f[-1])
            print(f"[{key}] cca_r1={r1:.3f}(p={p1:.3f}) smooth rho={rho:.3f} "
                  f"nn={rates[0]:.3f} far={rates[-1]:.3f} | fam nn={rates_f[0]:.3f} "
                  f"far={rates_f[-1]:.3f}", flush=True)

    df_new = pd.DataFrame(rows)
    if os.path.exists(out_csv) and done:
        df_new = pd.concat([pd.read_csv(out_csv), df_new], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=["market", "seed"], keep="last")
    df_new.to_csv(out_csv, index=False)
    print(df_new.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
