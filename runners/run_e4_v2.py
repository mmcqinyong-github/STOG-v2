"""E4 v2: Regime-overlap x window-level gating benefit (Theorem 4, pre-registered H4).

Re-run of E4 with window-level dynamic gates instead of a scalar convex gate.
Five combination strategies compared per field (delta x seed):
  S1 static convex combination (val simplex grid)
  S2 learned scalar gate (single softmax weight vector, learned on val)
  S3 MoE top-2 window-level routing (gate net -> per-window top-2 softmax)
  S4 dual-score window-level routing (ours: static score + beta*(1-m_hat)*dynamic score)
  S5 BMA posterior weighting (p(z|expert best) Gaussian posterior per window)

delta_hat: estimated from window inputs via a 2-component GMM on segment log-std,
segment transition rate calibrated to delta through the known generator family.

Usage:
  python run_e4_v2.py --calibrate
  python run_e4_v2.py --fields 0.1,2021 0.1,42      # trains experts, caches preds
  python run_e4_v2.py --combine                      # gates, metrics, CSVs, figures
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import linregress, spearmanr

from src.utils.common import set_seed, ensure_dir
from src.data.synthetic import SynthConfig, SpatioTemporalFieldGenerator
from src.experts.zoo import get_expert
from src.training.trainer import UnifiedTrainer
from src.probes.input_probe import InputProbe

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "e4_v2")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
FIG_DIR = os.path.join(ROOT, "results", "figures")
ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR); ensure_dir(FIG_DIR)

EXPERT_IDS = ["M52", "M233", "M03", "M47"]  # DLinear, QuantMo, RLinear, Autoformer
DELTAS = [0.1, 0.3, 0.6, 0.9]
SEEDS = [2021, 42, 3407]
H = 24
V = 8
N_SEG = 6            # segments per input window for the regime GMM
PROBE = InputProbe()


# ----------------------------------------------------------------------------
# delta_hat estimation: 1-D 2-Gaussian EM on segment log-std
# ----------------------------------------------------------------------------
def gmm1d_fit(x, n_iter=100, seed=0):
    """2-component 1-D GMM. Returns dict with components ordered by variance."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [10, 90])
    mu = np.array([lo, hi])
    var = np.array([x.var(), x.var()]) + 1e-6
    pi = np.array([0.5, 0.5])
    for _ in range(n_iter):
        # E step
        logp = np.stack([
            np.log(pi[k] + 1e-12) - 0.5 * np.log(2 * np.pi * var[k]) - (x - mu[k]) ** 2 / (2 * var[k])
            for k in range(2)], axis=1)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp); p /= p.sum(axis=1, keepdims=True)
        # M step
        for k in range(2):
            w = p[:, k]; s = w.sum() + 1e-12
            pi[k] = s / len(x)
            mu[k] = (w * x).sum() / s
            var[k] = (w * (x - mu[k]) ** 2).sum() / s + 1e-6
        # order by mean ascending (component 1 = high energy = regime 1)
    order = np.argsort(mu)
    return {"mu": mu[order], "var": var[order], "pi": pi[order]}


def gmm1d_posterior(x, model):
    """Posterior P(component 1 | x) (component 1 = high-variance regime)."""
    x = np.asarray(x, dtype=np.float64)
    mu, var, pi = model["mu"], model["var"], model["pi"]
    logp = np.stack([
        np.log(pi[k] + 1e-12) - 0.5 * np.log(2 * np.pi * var[k]) - (x - mu[k]) ** 2 / (2 * var[k])
        for k in range(2)], axis=1)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp); p /= p.sum(axis=1, keepdims=True)
    return p[:, 1]


def window_regime_stats(inp_windows, gmm=None, fit=False):
    """inp_windows: (n, L*V) flattened windows. Returns per-window:
    p_seg (n, L) posterior of regime 1 per timestep,
    m_hat (n,) window mixedness 2*min(f,1-f),
    trans (n,) mean adjacent |dp| (step switching rate),
    e (n, L) raw log-energy feature.
    Step-level regime posterior: feature e_t = log ||X_t||^2 per timestep.
    Regime 1 has ~6x larger amplitude, so a 1-D 2-Gaussian GMM separates steps
    well; the step transition rate is then monotone in delta by construction
    (P(state change per step) = delta for the generator's Markov chain).
    """
    n = inp_windows.shape[0]
    X = inp_windows.reshape(n, H, V)                    # (n, L, V)
    e = np.log((X ** 2).sum(axis=2) + 1e-8)             # (n, L) step energy
    if fit:
        gmm = gmm1d_fit(e.reshape(-1))
    p_seg = gmm1d_posterior(e.reshape(-1), gmm).reshape(n, H)
    f_hat = p_seg.mean(axis=1)
    m_hat = 2.0 * np.minimum(f_hat, 1.0 - f_hat)        # 1.0 at 50/50 mix
    trans = np.abs(np.diff(p_seg, axis=1)).mean(axis=1)
    return p_seg, m_hat, trans, e, gmm


def build_calibration():
    """Calibrate field-level transition rate q_bar -> delta using the generator."""
    print("[calibrate] building delta calibration curve ...")
    grid = np.round(np.arange(0.02, 0.981, 0.04), 2)
    qs, mbars = [], []
    for d in grid:
        cfg = SynthConfig(T=3000, V=V, H=H, alpha=1.0, delta=float(d), seed=12345)
        data = SpatioTemporalFieldGenerator(cfg).generate()
        inp = data["train_inp"].numpy()
        _, m_hat, trans, _, _ = window_regime_stats(inp, fit=True)
        qs.append(float(trans.mean())); mbars.append(float(m_hat.mean()))
        print(f"  delta={d:.2f}  q_bar={qs[-1]:.4f}  m_bar={mbars[-1]:.4f}")
    qs, mbars = np.array(qs), np.array(mbars)
    # enforce monotone q for interpolation via isotonic-ish cumulative max
    q_mono = np.maximum.accumulate(qs)
    np.savez(os.path.join(OUT_DIR, "delta_calibration.npz"),
             delta_grid=grid, q_bar=qs, q_mono=q_mono, m_bar=mbars)
    print("[calibrate] saved delta_calibration.npz")


def estimate_delta(trans_mean):
    cal = np.load(os.path.join(OUT_DIR, "delta_calibration.npz"))
    q_mono, grid = cal["q_mono"], cal["delta_grid"]
    return float(np.interp(trans_mean, q_mono, grid, left=grid[0], right=grid[-1]))


# ----------------------------------------------------------------------------
# Field runner: train experts, cache per-window preds + regime stats
# ----------------------------------------------------------------------------
def true_window_regime_frac(regimes, n_train, n_val, n):
    """True regime-1 fraction over input+target span of each window."""
    fr = np.zeros(n)
    for i in range(n):
        fr[i] = regimes[i:i + 2 * H].mean()
    return fr[:n_train], fr[n_train:n_train + n_val], fr[n_train + n_val:]


def run_field(delta, seed):
    tag = f"d{delta}_s{seed}"
    cache_path = os.path.join(CACHE_DIR, f"field_{tag}.npz")
    if os.path.exists(cache_path):
        print(f"[field {tag}] cache exists, skip")
        return
    t0 = time.time()
    cfg = SynthConfig(T=5000, V=V, H=H, alpha=1.0, delta=delta, seed=seed)
    data = SpatioTemporalFieldGenerator(cfg).generate()
    n_train = data["train_inp"].shape[0]; n_val = data["val_inp"].shape[0]
    n = n_train + n_val + data["test_inp"].shape[0]

    # regime stats per window (GMM fit on train windows only)
    inp_all = torch.cat([data["train_inp"], data["val_inp"], data["test_inp"]]).numpy()
    tr_inp, va_inp, te_inp = inp_all[:n_train], inp_all[n_train:n_train+n_val], inp_all[n_train+n_val:]
    _, _, _, _, gmm = window_regime_stats(tr_inp, fit=True)
    p_tr, m_tr, q_tr, _, _ = window_regime_stats(tr_inp, gmm=gmm)
    p_va, m_va, q_va, _, _ = window_regime_stats(va_inp, gmm=gmm)
    p_te, m_te, q_te, _, _ = window_regime_stats(te_inp, gmm=gmm)
    delta_hat = estimate_delta(float(q_tr.mean()))
    fr_tr, fr_va, fr_te = true_window_regime_frac(data["regimes"], n_train, n_val, n)

    # probe features on target-channel series
    def probe_feats(inp):
        v0 = inp.reshape(inp.shape[0], H, V)[:, :, 0]
        return np.stack([PROBE(v0[i]) for i in range(inp.shape[0])])
    z_tr, z_va, z_te = probe_feats(tr_inp), probe_feats(va_inp), probe_feats(te_inp)

    class MockDM:
        def __init__(self, d): self.windows = d
    dm = MockDM({
        "train": data["train_inp"], "train_tgt": data["train_tgt"],
        "val": data["val_inp"], "val_tgt": data["val_tgt"],
        "test": data["test_inp"], "test_tgt": data["test_tgt"],
    })

    ptr_list, pv_list, pt_list, val_mses = [], [], [], []
    for eid in EXPERT_IDS:
        set_seed(seed)
        expert = get_expert(eid, data["train_inp"].shape[1], hidden=128)
        trainer = UnifiedTrainer({"max_epochs": 8, "patience": 2, "batch_size": 256, "lr": 1e-4})
        try:
            res = trainer.train_expert(expert, dm)
            expert.eval()
            dev = next(expert.parameters()).device
            with torch.no_grad():
                pr = expert(data["train_inp"].to(dev)).cpu()
                pv = expert(data["val_inp"].to(dev)).cpu()
                pt = expert(data["test_inp"].to(dev)).cpu()
            val_mses.append(res["val_mse"])
        except Exception as ex:
            print(f"  [field {tag}] expert {eid} failed: {ex}")
            pr = torch.zeros_like(data["train_tgt"])
            pv = torch.zeros_like(data["val_tgt"]); pt = torch.zeros_like(data["test_tgt"])
            val_mses.append(999.0)
        ptr_list.append(pr.numpy()); pv_list.append(pv.numpy()); pt_list.append(pt.numpy())
        print(f"  [field {tag}] {eid} done ({time.time()-t0:.0f}s)")

    np.savez(cache_path,
             delta=delta, seed=seed, delta_hat=delta_hat,
             q_bar=float(q_tr.mean()), m_bar=float(m_tr.mean()),
             pr=np.stack(ptr_list), pv=np.stack(pv_list), pt=np.stack(pt_list),  # (E, n, H)
             ytr=data["train_tgt"].numpy(),
             yv=data["val_tgt"].numpy(), yt=data["test_tgt"].numpy(),
             m_va=m_va, m_te=m_te, m_tr=m_tr,
             fr_va=fr_va, fr_te=fr_te, fr_tr=fr_tr,
             z_va=z_va, z_te=z_te, z_tr=z_tr,
             val_mses=np.array(val_mses))
    print(f"[field {tag}] saved ({time.time()-t0:.0f}s)  delta_hat={delta_hat:.3f}")


# ----------------------------------------------------------------------------
# Combination strategies
# ----------------------------------------------------------------------------
def mse_t(pred, y):
    return float(((pred - y) ** 2).mean())


def combine_w(P, w):
    """P: (E, n, H) torch, w: (n, E) torch -> (n, H)"""
    return (P.permute(1, 0, 2) * w.unsqueeze(-1)).sum(dim=1)


def strat_S1_static(pv, yv, pt, yt):
    """Simplex grid search on val."""
    E = pv.shape[0]
    grids = []
    step = 0.1
    def rec(rem, k, cur):
        if k == E - 1:
            grids.append(cur + [rem]); return
        v = 0.0
        while v <= rem + 1e-9:
            rec(round(rem - v, 10), k + 1, cur + [round(v, 10)])
            v += step
    rec(1.0, 0, [])
    W = torch.tensor(np.array(grids), dtype=torch.float32)
    best_mse, best_w = float("inf"), None
    yv_t = yv
    for i in range(W.shape[0]):
        w = W[i].unsqueeze(0).expand(pv.shape[1], E)
        m = mse_t(combine_w(pv, w), yv_t)
        if m < best_mse:
            best_mse, best_w = m, W[i]
    w_test = best_w.unsqueeze(0).expand(pt.shape[1], E)
    return best_w, combine_w(pt, w_test)


def strat_S2_scalar(pv, yv, pt, yt, steps=400):
    E = pv.shape[0]
    theta = torch.zeros(E, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    n = pv.shape[1]
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(theta, dim=0).unsqueeze(0).expand(n, E)
        loss = ((combine_w(pv, w) - yv) ** 2).mean()
        loss.backward(); opt.step()
    w = torch.softmax(theta.detach(), dim=0)
    return w, combine_w(pt, w.unsqueeze(0).expand(pt.shape[1], E))


class GateNet(nn.Module):
    def __init__(self, d_z, E, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, hidden), nn.ReLU(), nn.Linear(hidden, E))
    def forward(self, z):
        return self.net(z)


def train_gate(scores_fn, loss_fn, params, n_epochs=300, lr=1e-2):
    opt = torch.optim.Adam(params, lr=lr)
    best_val, best_state, patience = float("inf"), None, 0
    for ep in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward(); opt.step()
        vl = scores_fn()
        if vl < best_val - 1e-7:
            best_val, patience = vl, 0
            best_state = [p.detach().clone() for p in params]
        else:
            patience += 1
            if patience >= 30:
                break
    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.copy_(s)


def strat_S3_moe(ztr, ptr, ytr, zva, pv, yv, zte, pt, seed):
    """MoE top-2 window-level routing."""
    torch.manual_seed(seed)
    E = pv.shape[0]
    gate = GateNet(ztr.shape[1], E)

    def weights(z):
        s = gate(z)
        v, idx = torch.topk(s, 2, dim=-1)
        w2 = torch.softmax(v, dim=-1)
        w = torch.zeros(z.shape[0], E)
        w.scatter_(1, idx, w2)
        return w

    def train_loss():
        return ((combine_w(ptr, weights(ztr)) - ytr) ** 2).mean()
    def val_loss():
        with torch.no_grad():
            return float(((combine_w(pv, weights(zva)) - yv) ** 2).mean())
    train_gate(val_loss, train_loss, list(gate.parameters()))
    with torch.no_grad():
        w_te = weights(zte)
    return w_te, combine_w(pt, w_te)


def strat_S4_dual(ztr, mtr, ptr, ytr, zva, mva, pv, yv, zte, mte, pt, val_mses, seed):
    """Dual-score window-level routing (ours):
    score = static + beta * (1 - m_hat) * dynamic(probe).
    The dynamic net is trained to be PRECISION-WEIGHTED: cross-entropy against
    soft per-window precision targets q_i ∝ exp(-mse_i / tau) on train windows.
    beta is then calibrated on val combined MSE."""
    torch.manual_seed(seed)
    E = pv.shape[0]
    static = -np.log(np.clip(val_mses, 1e-8, None))
    static = (static - static.mean()) / (static.std() + 1e-8)
    static_t = torch.tensor(static, dtype=torch.float32)
    dyn = GateNet(ztr.shape[1], E)

    # soft precision targets on train windows
    per_mse_tr = ((ptr - ytr.unsqueeze(0)) ** 2).mean(dim=2)      # (E, n_tr)
    tau = float(per_mse_tr.median())
    q = torch.softmax(-per_mse_tr.T / tau, dim=-1)                # (n_tr, E)

    opt = torch.optim.Adam(dyn.parameters(), lr=1e-2)
    best_val, best_state, patience = float("inf"), None, 0
    beta0 = 1.0
    for ep in range(300):
        opt.zero_grad()
        s = dyn(ztr)
        loss = -(q * torch.log_softmax(s, dim=-1)).sum(-1).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            w = torch.softmax(static_t.unsqueeze(0) + beta0 * (1.0 - mva).unsqueeze(-1) * dyn(zva), dim=-1)
            vl = float(((combine_w(pv, w) - yv) ** 2).mean())
        if vl < best_val - 1e-7:
            best_val, patience = vl, 0
            best_state = [p.detach().clone() for p in dyn.parameters()]
        else:
            patience += 1
            if patience >= 30:
                break
    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(dyn.parameters(), best_state):
                p.copy_(s)

    # calibrate beta on val combined MSE (small grid)
    def weights(z, m, beta):
        s = static_t.unsqueeze(0) + beta * (1.0 - m).unsqueeze(-1) * dyn(z)
        return torch.softmax(s, dim=-1)
    best_beta, best_vm = 1.0, float("inf")
    with torch.no_grad():
        for b in np.linspace(0, 3, 31):
            vm = float(((combine_w(pv, weights(zva, mva, float(b))) - yv) ** 2).mean())
            if vm < best_vm:
                best_vm, best_beta = vm, float(b)
    with torch.no_grad():
        w_te = weights(zte, mte, best_beta)
        dyn_coef = (1.0 - mte).mean().item()
    return w_te, combine_w(pt, w_te), best_beta, dyn_coef


def strat_S5_bma(zva, pv, yv, zte, pt):
    """BMA posterior: p(z | expert i best) Gaussian (diag cov) x prior freq."""
    E = pv.shape[0]
    per_mse = ((pv - yv.unsqueeze(0)) ** 2).mean(dim=2)   # (E, n_val)
    best = per_mse.argmin(dim=0).numpy()
    Z = zva.numpy()
    logpost = np.zeros((zte.shape[0], E))
    for i in range(E):
        zi = Z[best == i]
        pi_i = max((best == i).mean(), 1e-3)
        if len(zi) < 5:
            mu = Z.mean(0); var = Z.var(0) + 1e-3
        else:
            mu = zi.mean(0); var = zi.var(0) + 1e-3
        lp = -0.5 * (((zte.numpy() - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(axis=1)
        logpost[:, i] = np.log(pi_i) + lp
    logpost -= logpost.max(axis=1, keepdims=True)
    w = np.exp(logpost); w /= w.sum(axis=1, keepdims=True)
    w_t = torch.tensor(w, dtype=torch.float32)
    return w_t, combine_w(pt, w_t)


# ----------------------------------------------------------------------------
# Combine: run strategies per field, metrics, CSVs, figures
# ----------------------------------------------------------------------------
def combine():
    fields = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("field_"))
    if not fields:
        print("no cached fields"); return
    rows, dh_rows, weight_store = [], [], []
    for fname in fields:
        d = np.load(os.path.join(CACHE_DIR, fname))
        delta, seed = float(d["delta"]), int(d["seed"])
        pr = torch.tensor(d["pr"])
        pv = torch.tensor(d["pv"]); pt = torch.tensor(d["pt"])
        ytr = torch.tensor(d["ytr"])
        yv = torch.tensor(d["yv"]); yt = torch.tensor(d["yt"])
        ztr = torch.tensor(d["z_tr"], dtype=torch.float32)
        zva = torch.tensor(d["z_va"], dtype=torch.float32)
        zte = torch.tensor(d["z_te"], dtype=torch.float32)
        mtr = torch.tensor(d["m_tr"], dtype=torch.float32)
        mva = torch.tensor(d["m_va"], dtype=torch.float32)
        mte = torch.tensor(d["m_te"], dtype=torch.float32)
        val_mses = d["val_mses"]
        # standardize probe features on train
        mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
        ztr, zva, zte = (ztr - mu) / sd, (zva - mu) / sd, (zte - mu) / sd

        E = pv.shape[0]
        w_eq = torch.full((pt.shape[1], E), 1.0 / E)
        mse_eq = mse_t(combine_w(pt, w_eq), yt)

        w1, pred1 = strat_S1_static(pv, yv, pt, yt)
        mse1 = mse_t(pred1, yt)
        w2, pred2 = strat_S2_scalar(pv, yv, pt, yt)
        mse2 = mse_t(pred2, yt)
        w3, pred3 = strat_S3_moe(ztr, pr, ytr, zva, pv, yv, zte, pt, seed)
        mse3 = mse_t(pred3, yt)
        w4, pred4, beta4, dyncoef4 = strat_S4_dual(
            ztr, mtr, pr, ytr, zva, mva, pv, yv, zte, mte, pt, val_mses, seed)
        mse4 = mse_t(pred4, yt)
        w5, pred5 = strat_S5_bma(zva, pv, yv, zte, pt)
        mse5 = mse_t(pred5, yt)

        # KL(S4 || S5) and symmetric KL, precision correlations
        eps = 1e-10
        w4n = w4.numpy() + eps; w4n /= w4n.sum(1, keepdims=True)
        w5n = w5.numpy() + eps; w5n /= w5n.sum(1, keepdims=True)
        kl45 = float((w4n * np.log(w4n / w5n)).sum(1).mean())
        kl54 = float((w5n * np.log(w5n / w4n)).sum(1).mean())
        prec = 1.0 / (((pt - yt.unsqueeze(0)) ** 2).mean(dim=2).numpy() + 1e-6)  # (E, n)
        rho4 = spearmanr(w4n.reshape(-1), prec.T.reshape(-1)).statistic
        rho5 = spearmanr(w5n.reshape(-1), prec.T.reshape(-1)).statistic

        # oracle (diagnostic upper bound): per-window true best expert on test
        per_te = ((pt - yt.unsqueeze(0)) ** 2).mean(dim=2)         # (E, n)
        mse0 = float(per_te.min(dim=0).values.mean())

        for sname, mse_s in [("S1_static", mse1), ("S2_scalar", mse2), ("S3_moe_top2", mse3),
                              ("S4_dual_score", mse4), ("S5_bma", mse5), ("S0_oracle", mse0)]:
            rows.append({"delta": delta, "seed": seed, "strategy": sname,
                         "one_minus_delta": 1 - delta, "test_mse": mse_s,
                         "mse_equal": mse_eq, "mse_S1": mse1,
                         "benefit_vs_static": mse1 - mse_s,
                         "benefit_vs_equal": mse_eq - mse_s,
                         "kl_S4_S5": kl45, "kl_S5_S4": kl54,
                         "rho_w_precision_S4": rho4, "rho_w_precision_S5": rho5,
                         "S4_beta": beta4, "S4_dyn_coef_mean": dyncoef4,
                         "m_hat_mean_test": float(mte.mean())})

        dh_rows.append({"delta": delta, "seed": seed, "delta_hat": float(d["delta_hat"]),
                        "q_bar": float(d["q_bar"]), "m_bar": float(d["m_bar"]),
                        "abs_err": abs(float(d["delta_hat"]) - delta)})
        weight_store.append({"delta": delta, "seed": seed,
                             "w4": w4n, "w5": w5n, "m_te": mte.numpy(),
                             "fr_te": d["fr_te"]})
        print(f"[combine] {fname}: S1={mse1:.4f} S2={mse2:.4f} S3={mse3:.4f} "
              f"S4={mse4:.4f} S5={mse5:.4f} eq={mse_eq:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "e4v2_strategies.csv"), index=False)
    dh = pd.DataFrame(dh_rows)
    dh.to_csv(os.path.join(OUT_DIR, "e4v2_delta_hat.csv"), index=False)

    # summary
    summ = []
    for sname, g in df.groupby("strategy"):
        lr_ = linregress(g["one_minus_delta"], g["benefit_vs_static"])
        lr2 = linregress(g["one_minus_delta"], g["benefit_vs_equal"])
        summ.append({"metric": "benefit_vs_static_R2", "strategy": sname, "value": lr_.rvalue ** 2})
        summ.append({"metric": "benefit_vs_static_slope", "strategy": sname, "value": lr_.slope})
        summ.append({"metric": "benefit_vs_static_p", "strategy": sname, "value": lr_.pvalue})
        summ.append({"metric": "benefit_vs_equal_R2", "strategy": sname, "value": lr2.rvalue ** 2})
        summ.append({"metric": "benefit_vs_static_mean", "strategy": sname,
                     "value": g["benefit_vs_static"].mean()})
    s4 = df[df["strategy"] == "S4_dual_score"]
    s5 = df[df["strategy"] == "S5_bma"]
    summ.append({"metric": "delta_hat_MAE", "strategy": "-", "value": dh["abs_err"].mean()})
    summ.append({"metric": "KL_S4_given_S5", "strategy": "S4_vs_S5",
                 "value": s5["kl_S4_S5"].mean()})
    summ.append({"metric": "symKL_S4_S5", "strategy": "S4_vs_S5",
                 "value": (s5["kl_S4_S5"] + s5["kl_S5_S4"]).mean() / 2})
    summ.append({"metric": "rho_weight_precision_S4", "strategy": "S4_dual_score",
                 "value": s5["rho_w_precision_S4"].mean()})
    summ.append({"metric": "rho_weight_precision_S5", "strategy": "S5_bma",
                 "value": s5["rho_w_precision_S5"].mean()})
    d9 = df[(df["delta"] == 0.9)]
    d9s4 = d9[d9["strategy"] == "S4_dual_score"]; d9s1 = d9[d9["strategy"] == "S1_static"]
    summ.append({"metric": "deg09_S4_minus_S1_mse", "strategy": "S4_dual_score",
                 "value": (d9s4["test_mse"].values - d9s1["test_mse"].values).mean()})
    summ.append({"metric": "deg09_dyn_coef_mean", "strategy": "S4_dual_score",
                 "value": d9s4["S4_dyn_coef_mean"].mean()})
    pd.DataFrame(summ).to_csv(os.path.join(OUT_DIR, "e4v2_summary.csv"), index=False)

    make_figures(df, dh, weight_store)
    print("\n==== SUMMARY ====")
    print(pd.DataFrame(summ).to_string(index=False))
    print(dh.to_string(index=False))


def make_figures(df, dh, weight_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: benefit vs 1-delta by strategy
    strats = ["S1_static", "S2_scalar", "S3_moe_top2", "S4_dual_score", "S5_bma"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=False)
    for ax, s in zip(axes, strats):
        g = df[df["strategy"] == s]
        col = "benefit_vs_static" if s != "S1_static" else "benefit_vs_equal"
        x, y = g["one_minus_delta"].values, g[col].values
        ax.scatter(x, y, c=g["delta"].values, cmap="viridis", s=45, edgecolors="k", lw=0.4)
        if np.std(y) > 1e-12:
            lr_ = linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, lr_.intercept + lr_.slope * xs, "r--", lw=1.5)
            ax.set_title(f"{s}\n$R^2$={lr_.rvalue**2:.3f}, p={lr_.pvalue:.3g}", fontsize=10)
        else:
            ax.set_title(f"{s}\n(zero variance)", fontsize=10)
        ax.set_xlabel(r"$1-\delta$"); ax.axhline(0, color="gray", lw=0.6)
    axes[0].set_ylabel("benefit (MSE reduction)")
    fig.suptitle("E4 v2: gating benefit vs regime stability (1-$\\delta$), by strategy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v2_benefit_vs_delta_by_strategy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: delta_hat scatter
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(dh["delta"], dh["delta_hat"], s=60, edgecolors="k", lw=0.5, c="steelblue")
    lim = [0, 1]
    ax.plot(lim, lim, "r--", label="identity")
    mae = dh["abs_err"].mean()
    ax.set_xlabel(r"true $\delta$"); ax.set_ylabel(r"estimated $\hat{\delta}$")
    ax.set_title(f"E4 v2: $\\hat{{\\delta}}$ estimation (MAE={mae:.3f})")
    ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v2_delta_hat_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: S4 vs BMA per-window weights
    E = weight_store[0]["w4"].shape[1]
    fig, axes = plt.subplots(1, E, figsize=(4.5 * E, 4.2))
    rng = np.random.RandomState(0)
    for i, ax in enumerate(axes):
        a4 = np.concatenate([ws["w4"][:, i] for ws in weight_store])
        a5 = np.concatenate([ws["w5"][:, i] for ws in weight_store])
        mm = np.concatenate([ws["m_te"] for ws in weight_store])
        idx = rng.choice(len(a4), size=min(4000, len(a4)), replace=False)
        sc = ax.scatter(a5[idx], a4[idx], c=mm[idx], cmap="coolwarm", s=6, alpha=0.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("BMA posterior weight"); ax.set_ylabel("S4 dual-score weight")
        ax.set_title(f"expert {EXPERT_IDS[i]}")
    fig.colorbar(sc, ax=axes[-1], label=r"window mixedness $\hat{m}$")
    fig.suptitle("E4 v2: per-window gate weights — S4 (precision-weighted) vs BMA posterior")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "e4_v2_gate_vs_bma_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figures saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--fields", nargs="*", default=[],
                    help="entries like 0.1,2021")
    args = ap.parse_args()
    if args.calibrate:
        build_calibration()
    for f in args.fields:
        d, s = f.split(",")
        run_field(float(d), int(s))
    if args.combine:
        combine()


if __name__ == "__main__":
    main()
