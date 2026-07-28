"""E9 v2: Incremental Learning — 6 strategies, numerically-safe Hedge, full metrics.

Fixes over run_e9_full.py:
  1. Log-space weight updates (logsumexp), per-month loss normalization,
     adaptive eta, weight floor 1e-4  -> no underflow / one-hot collapse.
  2. Gradient clipping in all training; winsorized monthly loss (rolling
     q95 cap) as DriftMonitor prototype. Both raw and winsorized reported
     (paper main = winsorized, raw -> appendix).
  3. Six arms: fixed / periodic_retrain / online_finetune / hedge /
     ctx_hedge / oracle. Retrain & finetune arms truly update the 5
     representative experts (M47/M63/M03/M17/M31) month by month.
  4. Metrics M32-M36 + theory curve sqrt(T ln N).
  5. H7 verdict computed honestly (mean over seeds + per-seed).

Resume-capable: per (market, seed) checkpoint npz in results/e9_v2/ckpt/.
A finished block writes results/e9_v2/rows_{market}_{seed}.csv and is
skipped on re-run. `finalize` mode rebuilds csvs/figures from row files.

Usage:
  python run_e9_v2.py                  # run missing blocks
  python run_e9_v2.py --markets NP     # subset
  python run_e9_v2.py --finalize-only  # only rebuild outputs
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, get_all_cards

# ===================== Config =====================
MARKETS = ["NP", "DE"]
SEEDS = [2021, 42, 3407]
EXPERT_IDS = ["M47", "M63", "M03", "M18", "M31", "M89", "M50", "M233", "M17", "M220"]
RETRAIN_POOL = ["M47", "M63", "M03", "M17", "M31"]   # arms (b)/(c) real-update pool
N_MONTHS = 12
WEIGHT_FLOOR = 1e-4
RETRAIN_TOPK = 3
RETRAIN_EVERY = 3

INIT_CFG = {"max_epochs": 8, "patience": 3, "batch_size": 256, "lr": 1e-4,
            "clip": 1.0}
RETR_CFG = {"max_epochs": 6, "patience": 2, "batch_size": 256, "lr": 1e-4,
            "clip": 1.0}
FT_CFG = {"epochs": 2, "batch_size": 256, "lr": 1e-5, "clip": 1.0}

OUT_DIR = "./results/e9_v2"
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
FIG_DIR = "./results/figures"
ensure_dir(OUT_DIR); ensure_dir(CKPT_DIR); ensure_dir(FIG_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Numerically-safe helpers =====================
def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)) + 1e-300)


def normalize_weights(logw, floor=WEIGHT_FLOOR):
    """logw -> normalized weights with a hard floor (no permanent lock-out)."""
    w = np.exp(logw - logsumexp(logw))
    w = np.maximum(w, floor)
    return w / w.sum()


def winsorize(losses_hist):
    """Cap each monthly loss vector at rolling q95 of all losses seen so far.

    losses_hist: list of (N,) arrays (raw monthly losses per expert), the
    last entry being the current month. Returns capped current-month vector
    and the cap value.
    """
    all_l = np.concatenate(losses_hist)
    cap = np.quantile(all_l, 0.95)
    cap = max(cap, 1e-8)
    return np.minimum(losses_hist[-1], cap), float(cap)


def norm_regret_form(losses):
    """Map losses to [0,1] regret-form: 0 = best expert this month."""
    lo = losses.min()
    hi = np.quantile(losses, 0.95)
    rng = max(hi - lo, 1e-8)
    return np.clip((losses - lo) / rng, 0.0, 1.0)


# ===================== Training helpers (with grad clipping) =====================
def train_model(expert, X, Y, cfg, val_xy=None, verbose=False):
    """Train from scratch on (X, Y) tensors. Returns train seconds."""
    expert = expert.to(DEVICE)
    X = X.to(DEVICE); Y = Y.to(DEVICE)
    loader = DataLoader(TensorDataset(X, Y), batch_size=cfg["batch_size"],
                        shuffle=True)
    opt = torch.optim.Adam(expert.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    best_val, bad = np.inf, 0
    best_state = None
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        expert.train()
        for xb, yb in loader:
            opt.zero_grad()
            p = expert(xb)
            if p.dim() == 1:
                p = p.unsqueeze(-1)
            loss = crit(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(expert.parameters(), cfg["clip"])
            opt.step()
        if val_xy is not None:
            xv, yv = val_xy
            expert.eval()
            with torch.no_grad():
                pv = expert(xv.to(DEVICE))
                if pv.dim() == 1:
                    pv = pv.unsqueeze(-1)
                vm = crit(pv, yv.to(DEVICE)).item()
            if vm < best_val:
                best_val, bad = vm, 0
                best_state = {k: v.detach().clone() for k, v in expert.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    if best_state is not None:
        expert.load_state_dict(best_state)
    return time.time() - t0


def finetune_model(expert, X, Y, cfg):
    """Small-step online fine-tune on one month of data. Returns seconds."""
    expert = expert.to(DEVICE)
    X = X.to(DEVICE); Y = Y.to(DEVICE)
    loader = DataLoader(TensorDataset(X, Y), batch_size=cfg["batch_size"],
                        shuffle=True)
    opt = torch.optim.Adam(expert.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    t0 = time.time()
    for _ in range(cfg["epochs"]):
        expert.train()
        for xb, yb in loader:
            opt.zero_grad()
            p = expert(xb)
            if p.dim() == 1:
                p = p.unsqueeze(-1)
            loss = crit(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(expert.parameters(), cfg["clip"])
            opt.step()
    return time.time() - t0


@torch.no_grad()
def predict_np(expert, X, bs=8192):
    expert.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        p = expert(X[i:i + bs].to(DEVICE))
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ===================== Contextual prior (spectral affinity) =====================
def spectral_prior(dm_feats_mean, expert_ids):
    """prior_i ∝ exp(affinity match between dataset probe stats and genome card).

    Uses spectral_affinity from the genome cards: match low_freq_decay against
    (1 - |spec_decay|/3-ish), spike_heavy_tail against kurtosis, strong_
    periodicity against season_strength. Soft, bounded scores.
    """
    cards = get_all_cards()
    kurt = float(np.clip(dm_feats_mean["kurt"], 0, 30)) / 30.0   # 0..1
    season = float(np.clip(dm_feats_mean["season_strength"], 0, 1))
    # spec_decay more negative -> stronger low-freq decay
    lowfreq = float(np.clip(-dm_feats_mean["spec_decay"] / 3.0, 0, 1))
    score = []
    for eid in expert_ids:
        aff = cards[eid].spectral_affinity
        s = (lowfreq * aff.get("low_freq_decay", 0.5)
             + kurt * aff.get("spike_heavy_tail", 0.5)
             + season * aff.get("strong_periodicity", 0.5)
             + (1 - lowfreq) * (1 - aff.get("low_freq_decay", 0.5)) * 0.5
             + (1 - kurt) * (1 - aff.get("spike_heavy_tail", 0.5)) * 0.5)
        score.append(s)
    score = np.array(score)
    score = score - score.max()
    p = np.exp(2.0 * score)          # temperature: mild discrimination
    return p / p.sum()


# ===================== Per-block runner =====================
def run_block(market, seed):
    rows_path = os.path.join(OUT_DIR, f"rows_{market}_{seed}.csv")
    if os.path.exists(rows_path):
        print(f"[skip] {market}/{seed} done", flush=True)
        return
    ckpt_path = os.path.join(CKPT_DIR, f"{market}_{seed}.npz")
    print(f"=== E9v2 block {market}/{seed} ===", flush=True)
    set_seed(seed)
    t_block = time.time()

    dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed,
                       data_dir="./dataset/epf")
    dm.make_windows()
    dm.normalize()
    d_in = dm.windows["train"].shape[1]

    # ----- static predictions for the 10 experts (reuse saved npz) -----
    test_true = np.load(f"./results/preds/meta_{market}_{seed}.npz")["test_true"].astype(np.float64)
    n_test = test_true.shape[0]
    bs_ = n_test // N_MONTHS
    batch_slices = [slice(i * bs_, min((i + 1) * bs_, n_test)) for i in range(N_MONTHS)]

    P = np.zeros((len(EXPERT_IDS), n_test, 24), dtype=np.float64)  # preds
    val_mse = {}
    val_true = np.load(f"./results/preds/meta_{market}_{seed}.npz")["val_true"].astype(np.float64)
    for i, eid in enumerate(EXPERT_IDS):
        d = np.load(f"./results/preds/{market}_{eid}_{seed}.npz")
        P[i] = d["test_pred"].astype(np.float64)
        val_mse[eid] = float(((d["val_pred"].astype(np.float64) - val_true) ** 2).mean())
    # per-expert monthly raw losses  (E10[i, m])
    E10 = np.zeros((len(EXPERT_IDS), N_MONTHS))
    for m, sl in enumerate(batch_slices):
        E10[:, m] = ((P[:, sl, :] - test_true[sl][None, :, :]) ** 2).mean(axis=(1, 2))

    # contextual prior: spectral affinity x validation performance
    meta = np.load(f"./results/preds/meta_{market}_{seed}.npz")
    feat_names = list(meta["feat_names"])
    fmean = {n: float(meta["feat_test"][:, feat_names.index(n)].mean())
             for n in ["kurt", "season_strength", "spec_decay"]}
    prior_spec = spectral_prior(fmean, EXPERT_IDS)
    vm = np.array([val_mse[e] for e in EXPERT_IDS])
    prior_val = np.exp(-(vm - vm.min()) / (np.median(vm) + 1e-8))
    prior_val /= prior_val.sum()
    prior_ctx = prior_spec * prior_val
    prior_ctx /= prior_ctx.sum()

    # ================= simulation arms (a)(d)(e)(f) =================
    N = len(EXPERT_IDS)
    eta = np.sqrt(8.0 * np.log(N) / N_MONTHS)
    sim = {s: {"loss_raw": [], "loss_win": [], "loss_norm": [], "cost": []}
           for s in ["fixed", "hedge", "ctx_hedge", "oracle"]}
    hist_hedge, hist_ctx = [], []
    logw_h = np.log(np.ones(N) / N)
    logw_c = np.log(prior_ctx + 1e-12)
    weight_traj = []   # ctx weights per month (before seeing the month)
    best_eid_idx = int(np.argmin(vm))

    for m in range(N_MONTHS):
        t0 = time.time()
        raw = E10[:, m].copy()
        # oracle
        sim["oracle"]["loss_raw"].append(raw.min())
        # fixed
        sim["fixed"]["loss_raw"].append(raw[best_eid_idx])
        # hedge
        w_h = normalize_weights(logw_h)
        w_c = normalize_weights(logw_c)
        weight_traj.append(w_c.copy())
        ens_h = np.tensordot(w_h, P[:, batch_slices[m], :], axes=1)
        ens_c = np.tensordot(w_c, P[:, batch_slices[m], :], axes=1)
        loss_h = float(((ens_h - test_true[batch_slices[m]]) ** 2).mean())
        loss_c = float(((ens_c - test_true[batch_slices[m]]) ** 2).mean())
        sim["hedge"]["loss_raw"].append(loss_h)
        sim["ctx_hedge"]["loss_raw"].append(loss_c)
        # winsorized monthly expert losses (rolling q95 cap)
        hist_hedge.append(raw.copy()); hist_ctx.append(raw.copy())
        win, cap = winsorize(hist_hedge)
        sim["oracle"]["loss_win"].append(win.min())
        sim["fixed"]["loss_win"].append(win[best_eid_idx])
        # strategy-level winsorize: cap strategy loss at same cap
        sim["hedge"]["loss_win"].append(min(loss_h, cap))
        sim["ctx_hedge"]["loss_win"].append(min(loss_c, cap))
        r = norm_regret_form(win)          # [0,1] regret-form losses
        sim["oracle"]["loss_norm"].append(0.0)
        sim["fixed"]["loss_norm"].append(r[best_eid_idx])
        sim["hedge"]["loss_norm"].append(float(np.dot(w_h, r)))
        sim["ctx_hedge"]["loss_norm"].append(float(np.dot(w_c, r)))
        # log-space updates with winsorized, normalized losses
        logw_h += -eta * r
        logw_c += -eta * r
        logw_h -= logsumexp(logw_h)
        logw_c -= logsumexp(logw_c)
        cost = time.time() - t0
        for s in ["fixed", "hedge", "ctx_hedge", "oracle"]:
            sim[s]["cost"].append(cost if s in ("hedge", "ctx_hedge") else 0.0)

    weight_traj = np.array(weight_traj)   # (N_MONTHS, N)

    # ================= arm (b): periodic retrain (real training) ============
    ck = {}
    if os.path.exists(ckpt_path):
        ck = dict(np.load(ckpt_path, allow_pickle=True))

    if "b_done" not in ck:
        print("  [b] periodic retrain: init training 5 experts...", flush=True)
        models_b, init_state, init_train_sec = {}, {}, 0.0
        for eid in RETRAIN_POOL:
            set_seed(seed + int(eid[1:]))
            exp = get_expert(eid, d_in, hidden=256, drop=0.1)
            sec = train_model(exp, dm.windows["train"], dm.windows["train_tgt"],
                              INIT_CFG, val_xy=(dm.windows["val"], dm.windows["val_tgt"]))
            init_train_sec += sec
            init_state[eid] = {k: v.cpu().clone() for k, v in exp.state_dict().items()}
            models_b[eid] = exp
        print(f"  [b] init train total {init_train_sec:.1f}s", flush=True)

        b_loss_raw = []; b_cost = []
        Xtr_cum = dm.windows["train"].clone()
        Ytr_cum = dm.windows["train_tgt"].clone()
        test_inp = dm.windows["test"]; test_tgt = dm.windows["test_tgt"]
        pool_idx = [EXPERT_IDS.index(e) for e in RETRAIN_POOL]
        cum_losses = np.zeros(len(RETRAIN_POOL))
        for m in range(N_MONTHS):
            sl = batch_slices[m]
            t0 = time.time()
            preds = np.stack([predict_np(models_b[e], test_inp[sl]) for e in RETRAIN_POOL])
            ens = preds.mean(axis=0)
            loss = float(((ens - test_tgt[sl].numpy()) ** 2).mean())
            pred_cost = time.time() - t0
            b_loss_raw.append(loss)
            month_losses = ((preds - test_tgt[sl].numpy()[None]) ** 2).mean(axis=(1, 2))
            cum_losses += month_losses
            upd_sec = 0.0
            if (m + 1) % RETRAIN_EVERY == 0 and m + 1 < N_MONTHS:
                topk = np.argsort(cum_losses)[:RETRAIN_TOPK]
                Xtr_cum = torch.cat([Xtr_cum, test_inp[sl]])
                Ytr_cum = torch.cat([Ytr_cum, test_tgt[sl]])
                for k in topk:
                    eid = RETRAIN_POOL[int(k)]
                    set_seed(seed + int(eid[1:]) + m)
                    exp = get_expert(eid, d_in, hidden=256, drop=0.1)
                    upd_sec += train_model(exp, Xtr_cum, Ytr_cum, RETR_CFG)
                    models_b[eid] = exp
                print(f"    [b] month {m}: retrained top-{RETRAIN_TOPK} "
                      f"({upd_sec:.1f}s)", flush=True)
            else:
                Xtr_cum = torch.cat([Xtr_cum, test_inp[sl]])
                Ytr_cum = torch.cat([Ytr_cum, test_tgt[sl]])
            b_cost.append(pred_cost + upd_sec)
        # forgetting: final models on months 0-1 vs initial models
        forget_b = []
        eval_sl = slice(0, batch_slices[1].stop)
        for eid in RETRAIN_POOL:
            p_fin = predict_np(models_b[eid], test_inp[eval_sl])
            exp0 = get_expert(eid, d_in, hidden=256, drop=0.1)
            exp0.load_state_dict(init_state[eid]); exp0.to(DEVICE)
            p_ini = predict_np(exp0, test_inp[eval_sl])
            y0 = test_tgt[eval_sl].numpy()
            m_fin = float(((p_fin - y0) ** 2).mean())
            m_ini = float(((p_ini - y0) ** 2).mean())
            forget_b.append((m_fin - m_ini) / max(m_ini, 1e-8))
        ck.update(b_done=np.array([1]), b_loss_raw=np.array(b_loss_raw),
                  b_cost=np.array(b_cost), b_forget=np.array(forget_b),
                  b_init_train_sec=np.array([init_train_sec]))
        np.savez(ckpt_path, **ck)
        del models_b
        torch.cuda.empty_cache()
    print(f"  [b] done ({time.time()-t_block:.0f}s into block)", flush=True)

    # ================= arm (c): online fine-tune (real training) ============
    if "c_done" not in ck:
        print("  [c] online fine-tune: init training 5 experts...", flush=True)
        models_c, init_state_c, init_sec_c = {}, {}, 0.0
        for eid in RETRAIN_POOL:
            set_seed(seed + int(eid[1:]))
            exp = get_expert(eid, d_in, hidden=256, drop=0.1)
            sec = train_model(exp, dm.windows["train"], dm.windows["train_tgt"],
                              INIT_CFG, val_xy=(dm.windows["val"], dm.windows["val_tgt"]))
            init_sec_c += sec
            init_state_c[eid] = {k: v.cpu().clone() for k, v in exp.state_dict().items()}
            models_c[eid] = exp
        c_loss_raw = []; c_cost = []
        test_inp = dm.windows["test"]; test_tgt = dm.windows["test_tgt"]
        for m in range(N_MONTHS):
            sl = batch_slices[m]
            t0 = time.time()
            preds = np.stack([predict_np(models_c[e], test_inp[sl]) for e in RETRAIN_POOL])
            ens = preds.mean(axis=0)
            loss = float(((ens - test_tgt[sl].numpy()) ** 2).mean())
            c_loss_raw.append(loss)
            ft_sec = 0.0
            if m + 1 < N_MONTHS:
                for eid in RETRAIN_POOL:
                    ft_sec += finetune_model(models_c[eid], test_inp[sl],
                                             test_tgt[sl], FT_CFG)
            c_cost.append((time.time() - t0))
        forget_c = []
        eval_sl = slice(0, batch_slices[1].stop)
        for eid in RETRAIN_POOL:
            p_fin = predict_np(models_c[eid], test_inp[eval_sl])
            exp0 = get_expert(eid, d_in, hidden=256, drop=0.1)
            exp0.load_state_dict(init_state_c[eid]); exp0.to(DEVICE)
            p_ini = predict_np(exp0, test_inp[eval_sl])
            y0 = test_tgt[eval_sl].numpy()
            m_fin = float(((p_fin - y0) ** 2).mean())
            m_ini = float(((p_ini - y0) ** 2).mean())
            forget_c.append((m_fin - m_ini) / max(m_ini, 1e-8))
        ck.update(c_done=np.array([1]), c_loss_raw=np.array(c_loss_raw),
                  c_cost=np.array(c_cost), c_forget=np.array(forget_c),
                  c_init_train_sec=np.array([init_sec_c]))
        np.savez(ckpt_path, **ck)
        del models_c
        torch.cuda.empty_cache()
    print(f"  [c] done ({time.time()-t_block:.0f}s into block)", flush=True)

    # ================= assemble rows =================
    b_loss = ck["b_loss_raw"]; c_loss = ck["c_loss_raw"]
    # winsorized versions for b/c: cap at the 10-expert rolling cap sequence
    caps = []
    hist = []
    for m in range(N_MONTHS):
        hist.append(E10[:, m].copy())
        _, cap = winsorize(hist)
        caps.append(cap)
    caps = np.array(caps)

    # per-pool (5-expert) oracle for reference
    pool_idx = [EXPERT_IDS.index(e) for e in RETRAIN_POOL]

    rows = []
    strat_loss_raw = {
        "fixed": sim["fixed"]["loss_raw"],
        "periodic_retrain": list(b_loss),
        "online_finetune": list(c_loss),
        "hedge": sim["hedge"]["loss_raw"],
        "ctx_hedge": sim["ctx_hedge"]["loss_raw"],
        "oracle": sim["oracle"]["loss_raw"],
    }
    strat_loss_win = {
        "fixed": sim["fixed"]["loss_win"],
        "periodic_retrain": [min(l, caps[m]) for m, l in enumerate(b_loss)],
        "online_finetune": [min(l, caps[m]) for m, l in enumerate(c_loss)],
        "hedge": sim["hedge"]["loss_win"],
        "ctx_hedge": sim["ctx_hedge"]["loss_win"],
        "oracle": sim["oracle"]["loss_win"],
    }
    strat_loss_norm = {
        "fixed": sim["fixed"]["loss_norm"],
        "periodic_retrain": None, "online_finetune": None,
        "hedge": sim["hedge"]["loss_norm"],
        "ctx_hedge": sim["ctx_hedge"]["loss_norm"],
        "oracle": sim["oracle"]["loss_norm"],
    }
    # normalized form for b/c too (same mapping)
    for s, losses in [("periodic_retrain", b_loss), ("online_finetune", c_loss)]:
        vals = []
        for m in range(N_MONTHS):
            win10, _ = winsorize(hist[: m + 1])
            lo = win10.min()
            hi = np.quantile(win10, 0.95)
            rng = max(hi - lo, 1e-8)
            vals.append(float(np.clip((min(losses[m], caps[m]) - lo) / rng, 0, 1)))
        strat_loss_norm[s] = vals

    strat_cost = {
        "fixed": [0.0] * N_MONTHS,
        "periodic_retrain": list(ck["b_cost"]),
        "online_finetune": list(ck["c_cost"]),
        "hedge": sim["hedge"]["cost"],
        "ctx_hedge": sim["ctx_hedge"]["cost"],
        "oracle": [0.0] * N_MONTHS,
    }

    for s in strat_loss_raw:
        cum_raw = cum_win = cum_norm = 0.0
        for m in range(N_MONTHS):
            or_raw = strat_loss_raw["oracle"][m]
            or_win = strat_loss_win["oracle"][m]
            reg_raw = strat_loss_raw[s][m] - or_raw
            reg_win = strat_loss_win[s][m] - or_win
            reg_norm = strat_loss_norm[s][m] - strat_loss_norm["oracle"][m]
            cum_raw += reg_raw; cum_win += reg_win; cum_norm += reg_norm
            rows.append({
                "market": market, "seed": seed, "strategy": s, "month": m,
                "loss_raw": strat_loss_raw[s][m], "loss_win": strat_loss_win[s][m],
                "loss_norm": strat_loss_norm[s][m],
                "regret_raw": reg_raw, "regret_win": reg_win, "regret_norm": reg_norm,
                "cum_regret_raw": cum_raw, "cum_regret_win": cum_win,
                "cum_regret_norm": cum_norm,
                "update_cost_sec": strat_cost[s][m],
                "win_cap": caps[m],
                "weight_top5": json.dumps(
                    dict(zip(EXPERT_IDS,
                             np.round(np.sort(weight_traj[m])[::-1][:5], 4).tolist()))
                    if s == "ctx_hedge" else ""),
            })
    pd.DataFrame(rows).to_csv(rows_path, index=False)

    # extra per-block artifacts for figures/metrics
    np.savez(os.path.join(CKPT_DIR, f"weights_{market}_{seed}.npz"),
             weight_traj=weight_traj, expert_ids=np.array(EXPERT_IDS))
    meta_block = {
        "market": market, "seed": seed,
        "b_forget": ck["b_forget"].tolist(), "c_forget": ck["c_forget"].tolist(),
        "b_init_train_sec": float(ck["b_init_train_sec"][0]),
        "c_init_train_sec": float(ck["c_init_train_sec"][0]),
        "val_mse": val_mse, "best_fixed": EXPERT_IDS[best_eid_idx],
        "prior_ctx": dict(zip(EXPERT_IDS, np.round(prior_ctx, 4).tolist())),
    }
    with open(os.path.join(CKPT_DIR, f"meta_{market}_{seed}.json"), "w") as f:
        json.dump(meta_block, f, indent=1)
    print(f"=== block {market}/{seed} finished in {time.time()-t_block:.0f}s ===",
          flush=True)


# ===================== Metrics / summary / figures =====================
def recovery_time(monthly_loss, k=2.0):
    """M34: shock = month with loss > k * 12-month median. Recovery = months
    until loss falls back below 1.25 x pre-shock mean (up to 3 months prior).
    Returns mean recovery over detected shocks (NaN if none)."""
    L = np.asarray(monthly_loss, dtype=float)
    med = np.median(L)
    recs = []
    for t in range(1, len(L)):
        if L[t] > k * med:
            pre = L[max(0, t - 3):t].mean()
            thr = 1.25 * pre
            r = len(L) - t  # default: never recovers within stream
            for u in range(t + 1, len(L)):
                if L[u] <= thr:
                    r = u - t
                    break
            recs.append(r)
    return float(np.mean(recs)) if recs else np.nan


def warmup_month(df_s, df_fixed):
    """First month from which strategy monthly regret <= fixed's, sustained
    to end of stream. Returns 0-indexed month (len(stream) if never)."""
    r_s = df_s.sort_values("month")["regret_win"].values
    r_f = df_fixed.sort_values("month")["regret_win"].values
    ok = r_s <= r_f + 1e-12
    for m in range(len(ok)):
        if ok[m:].all():
            return m
    return len(ok)


def finalize():
    row_files = [os.path.join(OUT_DIR, f"rows_{m}_{s}.csv")
                 for m in MARKETS for s in SEEDS]
    row_files = [f for f in row_files if os.path.exists(f)]
    if len(row_files) < len(MARKETS) * len(SEEDS):
        print(f"WARNING: only {len(row_files)}/6 blocks finished")
    df = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "e9v2_strategies.csv"), index=False)

    # ---- M32 rolling MSE (3-month) ----
    df = df.sort_values(["market", "seed", "strategy", "month"])
    df["roll3_mse_win"] = (df.groupby(["market", "seed", "strategy"])["loss_win"]
                             .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    df.to_csv(os.path.join(OUT_DIR, "e9v2_strategies.csv"), index=False)

    # ---- summary (strategy level, winsorized = main, raw = appendix) ----
    last = df[df["month"] == N_MONTHS - 1]
    summ = (last.groupby(["market", "strategy"])
            .agg(cum_regret_win_mean=("cum_regret_win", "mean"),
                 cum_regret_win_std=("cum_regret_win", "std"),
                 cum_regret_raw_mean=("cum_regret_raw", "mean"),
                 cum_regret_raw_std=("cum_regret_raw", "std"),
                 cum_regret_norm_mean=("cum_regret_norm", "mean"),
                 avg_regret_win=("regret_win", "mean"))
            .reset_index())
    # M35 update cost: mean over ALL months + amortized init training
    metas0 = {}
    for mkt in MARKETS:
        for sd in SEEDS:
            p = os.path.join(CKPT_DIR, f"meta_{mkt}_{sd}.json")
            if os.path.exists(p):
                metas0[(mkt, sd)] = json.load(open(p))
    cost_rows = []
    for (mkt, sd, s), g in df.groupby(["market", "seed", "strategy"]):
        c = g["update_cost_sec"].mean()
        meta = metas0.get((mkt, sd), {})
        if s == "periodic_retrain":
            c += meta.get("b_init_train_sec", 0) / N_MONTHS
        elif s == "online_finetune":
            c += meta.get("c_init_train_sec", 0) / N_MONTHS
        cost_rows.append({"market": mkt, "seed": sd, "strategy": s,
                          "cost": c})
    cost_df = pd.DataFrame(cost_rows)
    cost_agg = (cost_df.groupby(["market", "strategy"])["cost"]
                .agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "update_cost_sec_per_month",
                                 "std": "update_cost_std"}))
    summ = summ.merge(cost_agg, on=["market", "strategy"], how="left")
    summ.to_csv(os.path.join(OUT_DIR, "e9v2_summary.csv"), index=False)

    # ---- metrics csv: M34 recovery, M35 cost, M36 forgetting ----
    met_rows = []
    metas = {}
    for mkt in MARKETS:
        for sd in SEEDS:
            p = os.path.join(CKPT_DIR, f"meta_{mkt}_{sd}.json")
            if os.path.exists(p):
                metas[(mkt, sd)] = json.load(open(p))
    for (mkt, sd), g in df.groupby(["market", "seed"]):
        meta = metas.get((mkt, sd), {})
        for s, gs in g.groupby("strategy"):
            gs = gs.sort_values("month")
            rec = recovery_time(gs["loss_win"].values, k=2.0)
            rec15 = recovery_time(gs["loss_win"].values, k=1.5)
            cost = gs["update_cost_sec"].mean()
            if s == "periodic_retrain":
                fg = float(np.mean(meta.get("b_forget", [np.nan])))
                cost += meta.get("b_init_train_sec", 0) / N_MONTHS
            elif s == "online_finetune":
                fg = float(np.mean(meta.get("c_forget", [np.nan])))
                cost += meta.get("c_init_train_sec", 0) / N_MONTHS
            else:
                fg = 0.0
            met_rows.append({"market": mkt, "seed": sd, "strategy": s,
                             "M34_recovery_months_k2": rec,
                             "M34_recovery_months_k1p5": rec15,
                             "M35_update_cost_sec_per_month": cost,
                             "M36_forgetting_rel": fg})
    met = pd.DataFrame(met_rows)
    met.to_csv(os.path.join(OUT_DIR, "e9v2_metrics.csv"), index=False)

    # ---- H7 verdict ----
    h7 = {}
    for mkt in df["market"].unique():
        d = last[last["market"] == mkt]
        piv = d.pivot_table(index="seed", columns="strategy",
                            values="cum_regret_win")
        mean_by = piv.mean()
        order_ok_mean = (mean_by.get("ctx_hedge", np.inf)
                         < mean_by.get("hedge", np.inf)
                         < mean_by.get("fixed", np.inf))
        per_seed = ((piv["ctx_hedge"] < piv["hedge"])
                    & (piv["hedge"] < piv["fixed"]))
        # warmup on mean-over-seed monthly regret
        warm = {}
        for s in ["hedge", "ctx_hedge"]:
            ms = (df[(df["market"] == mkt) & (df["strategy"] == s)]
                  .groupby("month")["regret_win"].mean().reset_index())
            mf = (df[(df["market"] == mkt) & (df["strategy"] == "fixed")]
                  .groupby("month")["regret_win"].mean().reset_index())
            warm[s] = warmup_month(ms, mf)
        w_h, w_c = warm["hedge"], warm["ctx_hedge"]
        red = (w_h - w_c) / w_h if w_h > 0 else (1.0 if w_c == 0 else 0.0)
        h7[mkt] = {"order_mean_seeds": bool(order_ok_mean),
                   "order_per_seed": {int(k): bool(v) for k, v in per_seed.items()},
                   "warmup_hedge_month": int(w_h), "warmup_ctx_month": int(w_c),
                   "warmup_reduction": float(red),
                   "warmup_ok": bool(red >= 0.30)}
    with open(os.path.join(OUT_DIR, "e9v2_h7.json"), "w") as f:
        json.dump(h7, f, indent=1)

    make_figures(df)
    return df, summ, met, h7


def make_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        pass

    STRAT_LABEL = {"fixed": "(a) Fixed best-single",
                   "periodic_retrain": "(b) Periodic retrain",
                   "online_finetune": "(c) Online fine-tune",
                   "hedge": "(d) Post-hoc Hedge",
                   "ctx_hedge": "(e) Contextual Hedge (ours)",
                   "oracle": "(f) Oracle switching"}
    ORDER = ["fixed", "periodic_retrain", "online_finetune", "hedge",
             "ctx_hedge", "oracle"]
    colors = {"fixed": "#888888", "periodic_retrain": "#1f77b4",
              "online_finetune": "#2ca02c", "hedge": "#ff7f0e",
              "ctx_hedge": "#d62728", "oracle": "#9467bd"}

    # ---- Fig 1: cumulative regret (winsorized) + theory sqrt(T ln N) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, mkt in zip(axes, MARKETS):
        d = df[df["market"] == mkt]
        T = np.arange(1, N_MONTHS + 1)
        theory = np.sqrt(T * np.log(len(EXPERT_IDS)))
        for s in ORDER:
            g = (d[d["strategy"] == s].groupby("month")["cum_regret_win"]
                 .agg(["mean", "std"]).reset_index())
            ax.plot(g["month"] + 1, g["mean"], marker="o", ms=4,
                    color=colors[s], label=STRAT_LABEL[s])
            ax.fill_between(g["month"] + 1, g["mean"] - g["std"],
                            g["mean"] + g["std"], color=colors[s], alpha=0.12)
        # theory overlay on normalized-regret twin axis
        ax2 = ax.twinx()
        ax2.plot(T, theory, "k--", lw=1.5, alpha=0.7,
                 label=r"theory $\sqrt{T\ln N}$ (norm.)")
        g = (d[d["strategy"] == "ctx_hedge"].groupby("month")["cum_regret_norm"]
             .mean().reset_index())
        ax2.plot(g["month"] + 1, g["cum_regret_norm"], ":", color="#d62728",
                 lw=1.2, label="Ctx Hedge (norm. regret)")
        ax2.set_ylabel("normalized regret", fontsize=9)
        ax2.tick_params(labelsize=8)
        ax.set_title(f"{mkt} market", fontsize=12)
        ax.set_xlabel("month")
        ax.set_ylabel("cumulative regret (winsorized MSE)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    fig.suptitle("E9 v2: Cumulative regret of 6 strategies "
                 "(mean ± std over 3 seeds, winsorized)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e9_v2_cumulative_regret_6strategies.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: monthly loss heatmap (strategy x month), NP/DE panels ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, mkt in zip(axes, MARKETS):
        piv = (df[df["market"] == mkt]
               .groupby(["strategy", "month"])["loss_win"].mean()
               .unstack().reindex(ORDER))
        piv.index = [STRAT_LABEL[s] for s in piv.index]
        piv.columns = piv.columns + 1
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(N_MONTHS)); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center",
                        va="center", fontsize=6.5,
                        color="white" if piv.values[i, j] < piv.values.max() * 0.6
                        else "black")
        ax.set_title(f"{mkt}: monthly winsorized loss (mean over seeds)")
        ax.set_xlabel("month")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("E9 v2: Strategy × month loss heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e9_v2_monthly_loss_heatmap.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: ctx hedge weight evolution (top-5 experts) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, mkt in zip(axes, MARKETS):
        trajs = []
        for sd in SEEDS:
            p = os.path.join(CKPT_DIR, f"weights_{mkt}_{sd}.npz")
            if os.path.exists(p):
                w = np.load(p, allow_pickle=True)
                trajs.append((w["weight_traj"], list(w["expert_ids"])))
        if not trajs:
            continue
        eids = trajs[0][1]
        W = np.mean([t[0] for t in trajs], axis=0)  # (months, N)
        mean_w = W.mean(axis=0)
        top5 = np.argsort(mean_w)[::-1][:5]
        for k in top5:
            ax.plot(range(1, N_MONTHS + 1), W[:, k], marker="o", ms=4,
                    label=eids[k])
        others = np.delete(W, top5, axis=1).sum(axis=1)
        ax.plot(range(1, N_MONTHS + 1), others, "k--", lw=1.2, label="others (sum)")
        ax.axhline(1.0 / len(eids), color="gray", ls=":", lw=1,
                   label="uniform 1/N")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("month"); ax.set_ylabel("weight")
        ax.set_title(f"{mkt}: Ctx Hedge weights (mean over seeds)")
        ax.legend(fontsize=8)
    fig.suptitle("E9 v2: Contextual Hedge weight evolution — no one-hot collapse",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e9_v2_weight_evolution.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("figures saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()
    if args.finalize_only:
        finalize()
        return
    markets = args.markets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for mkt in markets:
        for sd in seeds:
            run_block(mkt, sd)
    row_files = [os.path.join(OUT_DIR, f"rows_{m}_{s}.csv")
                 for m in MARKETS for s in SEEDS]
    if all(os.path.exists(f) for f in row_files):
        finalize()


if __name__ == "__main__":
    main()
