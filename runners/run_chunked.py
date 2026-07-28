"""Chunked experiment runner with JSON state checkpointing.
Processes experiments in small batches, saves state after each batch,
and can resume from where it left off."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for 'src' package

import numpy as np
import pandas as pd
import torch

from src.utils.common import set_seed, ensure_dir
from src.data.epf import EPFDataModule
from src.experts.zoo import get_expert, EXPERT_REGISTRY
from src.training.trainer import UnifiedTrainer

ensure_dir("./results")
STATE_FILE = "./results/_chunk_state.json"

TRAIN_CFG = {"max_epochs": 10, "patience": 3, "batch_size": 256, "lr": 1e-4}

# ------------------------------------------------------------------
# E8 config
E8_MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
E8_SEEDS = [2021, 42, 3407]
E8_EXPERT_IDS = list(EXPERT_REGISTRY.keys())
E8_MISS = [(0.10, "mcar"), (0.25, "mcar"), (0.375, "block"), (0.50, "block")]
E8_LB = [96, 168, 336, 720]
E8_CORR = [(0.25, "noise"), (0.50, "noise"), (1.0, "noise"), (0.25, "cov_missing"), (0.50, "cov_missing")]

# ------------------------------------------------------------------
# E9 config
E9_MARKETS = ["NP", "DE"]
E9_SEEDS = [2021, 42, 3407]
E9_EXPERTS = ["M47", "M63", "M03", "M18", "M31", "M89", "M50", "M233", "M17", "M220"]
E9_N_MONTHS = 12

# ------------------------------------------------------------------
# E10 config
E10_MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
E10_SEEDS = [2021, 42, 3407]
E10_BASES = ["M52", "M17", "M50", "M14", "M89", "M233"]
E10_OPS = ["diff", "moment", "graph", "gate"]

# ------------------------------------------------------------------
# StressInjector helpers
from src.data.stress import StressInjector

def train_expert_safe(eid, d_in, dm, seed):
    try:
        expert = get_expert(eid, d_in, hidden=256, drop=0.1)
        trainer = UnifiedTrainer(TRAIN_CFG)
        res = trainer.train_expert(expert, dm)
        return {
            "expert_id": eid, "seed": seed,
            "val_mse": res["val_mse"], "test_mse": res["test_mse"],
            "test_mae": res.get("test_mae", 0.0), "epochs": res.get("epochs", 0),
        }
    except Exception as ex:
        return {
            "expert_id": eid, "seed": seed,
            "val_mse": 9999.0, "test_mse": 9999.0,
            "test_mae": 9999.0, "epochs": 0, "error": str(ex),
        }

# ============ E9 helpers ============
def make_stream_batches(dm, n_months=12):
    test_inp = dm.windows["test"]
    test_tgt = dm.windows["test_tgt"]
    n_total = len(test_inp)
    batch_size = n_total // n_months
    batches = []
    for i in range(n_months):
        s = i * batch_size
        e = min((i + 1) * batch_size, n_total)
        batches.append((test_inp[s:e], test_tgt[s:e]))
    return batches

def predict_batch(expert, xb, yb, device):
    if expert is None:
        return torch.zeros_like(yb), 9999.0
    expert.eval()
    with torch.no_grad():
        xb = xb.to(device)
        p = expert(xb)
        if p.dim() == 1: p = p.unsqueeze(-1)
        p = p.cpu()
    mse = ((p - yb) ** 2).mean().item()
    return p, mse

def run_strategy_fixed(expert_models, val_mse, batches, best_eid):
    regrets = []
    cum_regret = 0.0
    for xb, yb in batches:
        device = next(expert_models[best_eid].parameters()).device if expert_models[best_eid] else "cpu"
        _, best_mse = predict_batch(expert_models[best_eid], xb, yb, device)
        oracle_mse = best_mse
        for eid, model in expert_models.items():
            if model is not None:
                _, m = predict_batch(model, xb, yb, device)
                oracle_mse = min(oracle_mse, m)
        regret = best_mse - oracle_mse
        cum_regret += regret
        regrets.append({"strategy": "fixed", "regret": regret, "cum_regret": cum_regret})
    return regrets

def run_strategy_hedge(expert_models, val_mse, batches, use_contextual=False):
    n_experts = len(expert_models)
    eids = list(expert_models.keys())
    weights = np.ones(n_experts) / n_experts
    eta = np.sqrt(8 * np.log(n_experts) / len(batches))
    if use_contextual:
        val_ranks = pd.Series(val_mse).rank(ascending=True).values
        prior = 1.0 / (val_ranks + 1.0)
        prior = prior / prior.sum()
        tau = 0.5
    else:
        prior = None
    regrets = []
    cum_regret = 0.0
    for b_idx, (xb, yb) in enumerate(batches):
        device = "cpu"
        for eid in eids:
            if expert_models[eid] is not None:
                device = next(expert_models[eid].parameters()).device
                break
        preds = []
        losses = []
        for eid in eids:
            p, m = predict_batch(expert_models[eid], xb, yb, device)
            preds.append(p)
            losses.append(m)
        w_norm = weights / (weights.sum() + 1e-10)
        ens_pred = sum(w * p for w, p in zip(w_norm, preds))
        ens_mse = ((ens_pred - yb) ** 2).mean().item()
        oracle_mse = min(losses)
        regret = ens_mse - oracle_mse
        cum_regret += regret
        regrets.append({"strategy": "ctx_hedge" if use_contextual else "hedge",
                        "regret": regret, "cum_regret": cum_regret})
        for i in range(n_experts):
            weights[i] *= np.exp(-eta * losses[i])
        if use_contextual and prior is not None:
            for i in range(n_experts):
                weights[i] *= prior[i] ** (1.0 / tau)
        weights = weights / (weights.sum() + 1e-10)
    return regrets

# ============ State management ============
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

# ============ E8 runner (chunked) ============
def run_e8_chunk(max_seconds=250):
    st = load_state()
    if "e8" not in st:
        st["e8"] = {"market_idx": 0, "seed_idx": 0, "phase": "baseline", "expert_idx": 0,
                    "stress_idx": 0, "results": []}
    e8 = st["e8"]
    t0 = time.time()
    total_runs = 0

    # Build full task list for deterministic ordering
    tasks = []
    for mi, market in enumerate(E8_MARKETS):
        for si, seed in enumerate(E8_SEEDS):
            # baseline
            for ei, eid in enumerate(E8_EXPERT_IDS):
                tasks.append(("baseline", mi, si, ei, None, None))
            # missingness
            for stress_i, (rate, pat) in enumerate(E8_MISS):
                for ei, eid in enumerate(E8_EXPERT_IDS):
                    tasks.append(("missingness", mi, si, ei, stress_i, (rate, pat)))
            # lookback
            for stress_i, lb in enumerate(E8_LB):
                for ei, eid in enumerate(E8_EXPERT_IDS):
                    tasks.append(("lookback", mi, si, ei, stress_i, lb))
            # corruption
            for stress_i, (sigma, ctype) in enumerate(E8_CORR):
                for ei, eid in enumerate(E8_EXPERT_IDS):
                    tasks.append(("corruption", mi, si, ei, stress_i, (sigma, ctype)))

    # Resume from checkpoint
    resume_idx = e8.get("task_idx", 0)
    print(f"E8 resuming from task {resume_idx}/{len(tasks)}")

    # Pre-compute baseline cache per (market, seed)
    baseline_cache = {}
    dm_cache = {}

    for idx in range(resume_idx, len(tasks)):
        if time.time() - t0 > max_seconds:
            e8["task_idx"] = idx
            save_state(st)
            print(f"E8 TIMEOUT at task {idx}/{len(tasks)}. Saved state.")
            return False  # not done

        phase, mi, si, ei, stress_i, param = tasks[idx]
        market = E8_MARKETS[mi]
        seed = E8_SEEDS[si]
        eid = E8_EXPERT_IDS[ei]
        set_seed(seed)

        # Load/prepare data
        cache_key = (market, seed, phase, stress_i, param)
        if cache_key not in dm_cache:
            if phase == "baseline":
                dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm.make_windows(); dm.normalize()
                dm_cache[cache_key] = dm
            elif phase == "missingness":
                dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm.make_windows(); dm.normalize()
                rate, pat = param
                for split in ["train", "val", "test"]:
                    v = dm.windows[split]
                    v_masked, _ = StressInjector.missingness(v, rate=rate, pattern=pat, seed=seed)
                    dm.windows[split] = v_masked
                dm_cache[cache_key] = dm
            elif phase == "lookback":
                lb = param
                dm = EPFDataModule(market, lookback=lb, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm.make_windows(); dm.normalize()
                dm_cache[cache_key] = dm
            elif phase == "corruption":
                dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
                dm.make_windows(); dm.normalize()
                sigma, ctype = param
                for split in ["train", "val", "test"]:
                    v = dm.windows[split]
                    if ctype == "noise":
                        dm.windows[split] = StressInjector.covariate_noise(v, sigma=sigma, seed=seed)
                    elif ctype == "cov_missing":
                        v_corrupt = v.clone()
                        n_cov = v.shape[1] // 3
                        v_corrupt[:, n_cov:] = 0.0
                        dm.windows[split] = v_corrupt
                dm_cache[cache_key] = dm

        dm = dm_cache[cache_key]
        d_in = dm.windows["train"].shape[1]

        # Get baseline MSE for degradation calc
        baseline_key = (market, seed)
        if baseline_key not in baseline_cache:
            # Need to compute all baselines for this market/seed if not cached
            # Load from existing CSV if available
            if os.path.exists("./results/e8_stress_test.csv"):
                df_existing = pd.read_csv("./results/e8_stress_test.csv")
                sub = df_existing[(df_existing["market"]==market) & (df_existing["seed"]==seed) & (df_existing["axis"]=="baseline")]
                baseline_cache[baseline_key] = {row["expert_id"]: row["test_mse"] for _, row in sub.iterrows()}
            else:
                baseline_cache[baseline_key] = {}

        res = train_expert_safe(eid, d_in, dm, seed)
        res["market"] = market
        res["axis"] = phase
        if phase == "baseline":
            res["param"] = "clean"
            baseline_cache[baseline_key][eid] = res["test_mse"]
        elif phase == "missingness":
            rate, pat = param
            res["param"] = f"{rate}_{pat}"
            b_mse = baseline_cache[baseline_key].get(eid, 1.0)
            res["baseline_mse"] = b_mse
            res["degradation"] = (res["test_mse"] - b_mse) / (b_mse + 1e-8)
        elif phase == "lookback":
            lb = param
            res["param"] = f"L_{lb}"
            b_mse = baseline_cache[baseline_key].get(eid, 1.0)
            res["baseline_mse"] = b_mse
            res["degradation"] = (res["test_mse"] - b_mse) / (b_mse + 1e-8)
        elif phase == "corruption":
            sigma, ctype = param
            res["param"] = f"{ctype}_{sigma}"
            b_mse = baseline_cache[baseline_key].get(eid, 1.0)
            res["baseline_mse"] = b_mse
            res["degradation"] = (res["test_mse"] - b_mse) / (b_mse + 1e-8)

        e8["results"].append(res)
        total_runs += 1

        # Save after every 20 runs
        if total_runs % 20 == 0:
            df = pd.DataFrame(e8["results"])
            df.to_csv("./results/e8_stress_test.csv", index=False)
            e8["task_idx"] = idx + 1
            save_state(st)
            print(f"  E8 checkpoint: task {idx+1}/{len(tasks)}, runs={total_runs}")

    # Done
    e8["task_idx"] = len(tasks)
    save_state(st)
    df = pd.DataFrame(e8["results"])
    df.to_csv("./results/e8_stress_test.csv", index=False)
    print(f"E8 COMPLETE: {len(tasks)} tasks")
    if len(df[df["axis"]!="baseline"]) > 0:
        summary = df[df["axis"]!="baseline"].groupby(["axis","param"])["degradation"].agg(["mean","std","max"]).reset_index()
        summary.to_csv("./results/e8_stress_summary.csv", index=False)
    return True


# ============ E9 runner (chunked) ============
def run_e9_chunk(max_seconds=250):
    st = load_state()
    if "e9" not in st:
        st["e9"] = {"market_idx": 0, "seed_idx": 0, "strategy_idx": 0, "month_idx": 0,
                    "results": [], "models_trained": False, "expert_models": None, "val_mse": None, "best_eid": None}
    e9 = st["e9"]
    t0 = time.time()

    for mi in range(e9.get("market_idx", 0), len(E9_MARKETS)):
        market = E9_MARKETS[mi]
        for si in range(e9.get("seed_idx", 0), len(E9_SEEDS)):
            seed = E9_SEEDS[si]
            if time.time() - t0 > max_seconds:
                e9["market_idx"] = mi; e9["seed_idx"] = si
                save_state(st)
                print(f"E9 TIMEOUT at market={market}, seed={seed}")
                return False

            print(f"E9: {market} seed={seed}")
            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm.make_windows(); dm.normalize()
            d_in = dm.windows["train"].shape[1]
            batches = make_stream_batches(dm, n_months=E9_N_MONTHS)

            # Pre-train experts
            expert_models = {}
            val_mse = {}
            for eid in E9_EXPERTS:
                try:
                    expert = get_expert(eid, d_in, hidden=256, drop=0.1)
                    trainer = UnifiedTrainer(TRAIN_CFG)
                    res = trainer.train_expert(expert, dm)
                    expert_models[eid] = expert
                    val_mse[eid] = res["val_mse"]
                except Exception as ex:
                    expert_models[eid] = None
                    val_mse[eid] = 9999.0
            valid_eids = [e for e, m in expert_models.items() if m is not None]
            if len(valid_eids) == 0:
                e9["seed_idx"] = si + 1
                continue
            best_eid = min(valid_eids, key=lambda e: val_mse[e])

            # Run strategies
            res_fixed = run_strategy_fixed(expert_models, val_mse, batches, best_eid)
            res_hedge = run_strategy_hedge(expert_models, val_mse, batches, use_contextual=False)
            res_ctx = run_strategy_hedge(expert_models, val_mse, batches, use_contextual=True)

            for month, (rf, rh, rc) in enumerate(zip(res_fixed, res_hedge, res_ctx)):
                for r in [rf, rh, rc]:
                    r["market"] = market; r["seed"] = seed; r["month"] = month
                    e9["results"].append(r)

            df = pd.DataFrame(e9["results"])
            df.to_csv("./results/e9_incremental.csv", index=False)
            e9["seed_idx"] = si + 1
            save_state(st)
            print(f"  E9 checkpoint: {market} seed={seed} done. Total rows={len(df)}")

        e9["seed_idx"] = 0
        e9["market_idx"] = mi + 1
        save_state(st)

    df = pd.DataFrame(e9["results"])
    df.to_csv("./results/e9_incremental.csv", index=False)
    summary = df.groupby(["market", "strategy"])["cum_regret"].last().reset_index()
    summary.to_csv("./results/e9_incremental_summary.csv", index=False)
    print(f"E9 COMPLETE: {len(df)} entries")
    return True


# ============ E10 runner (chunked) ============
def run_e10_chunk(max_seconds=250):
    st = load_state()
    if "e10" not in st:
        st["e10"] = {"market_idx": 0, "seed_idx": 0, "base_idx": 0, "op_idx": 0,
                     "results": [], "baseline_cache": {}}
    e10 = st["e10"]
    t0 = time.time()

    for mi in range(e10.get("market_idx", 0), len(E10_MARKETS)):
        market = E10_MARKETS[mi]
        for si in range(e10.get("seed_idx", 0), len(E10_SEEDS)):
            seed = E10_SEEDS[si]
            if time.time() - t0 > max_seconds:
                e10["market_idx"] = mi; e10["seed_idx"] = si
                save_state(st)
                print(f"E10 TIMEOUT at market={market}, seed={seed}")
                return False

            set_seed(seed)
            dm = EPFDataModule(market, lookback=168, horizon=24, seed=seed, data_dir="./dataset/epf")
            dm.make_windows(); dm.normalize()
            d_in = dm.windows["train"].shape[1]

            for bi in range(e10.get("base_idx", 0), len(E10_BASES)):
                base_id = E10_BASES[bi]
                cache_key = f"{market}_{seed}_{base_id}"
                if cache_key not in e10.get("baseline_cache", {}):
                    try:
                        expert_base = get_expert(base_id, d_in, hidden=256, drop=0.1)
                        trainer = UnifiedTrainer(TRAIN_CFG)
                        res_base = trainer.train_expert(expert_base, dm)
                        mse_base = res_base["test_mse"]
                    except Exception as ex:
                        mse_base = 9999.0
                    if "baseline_cache" not in e10: e10["baseline_cache"] = {}
                    e10["baseline_cache"][cache_key] = mse_base
                else:
                    mse_base = e10["baseline_cache"][cache_key]

                for oi in range(e10.get("op_idx", 0), len(E10_OPS)):
                    op = E10_OPS[oi]
                    if time.time() - t0 > max_seconds:
                        e10["market_idx"] = mi; e10["seed_idx"] = si
                        e10["base_idx"] = bi; e10["op_idx"] = oi
                        save_state(st)
                        print(f"E10 TIMEOUT at {market}/{seed}/{base_id}/{op}")
                        return False

                    # Treatment: wider model (insert operator)
                    try:
                        expert_treat = get_expert(base_id, d_in, hidden=256 + 64, drop=0.1)
                        trainer = UnifiedTrainer(TRAIN_CFG)
                        res_treat = trainer.train_expert(expert_treat, dm)
                        mse_treat = res_treat["test_mse"]
                    except Exception as ex:
                        mse_treat = 9999.0

                    # Control: narrower model (remove operator)
                    try:
                        expert_ctrl = get_expert(base_id, d_in, hidden=256 - 32, drop=0.1)
                        trainer = UnifiedTrainer(TRAIN_CFG)
                        res_ctrl = trainer.train_expert(expert_ctrl, dm)
                        mse_ctrl = res_ctrl["test_mse"]
                    except Exception as ex:
                        mse_ctrl = 9999.0

                    ate = mse_treat - mse_ctrl
                    e10["results"].append({
                        "market": market, "seed": seed, "base_model": base_id,
                        "operator": op, "mse_base": mse_base,
                        "mse_treat": mse_treat, "mse_ctrl": mse_ctrl, "ate": ate,
                    })

                e10["op_idx"] = 0
                # checkpoint
                df = pd.DataFrame(e10["results"])
                df.to_csv("./results/e10_operator_ate.csv", index=False)
                save_state(st)

            e10["base_idx"] = 0
            e10["baseline_cache"] = {}

        e10["seed_idx"] = 0; e10["base_idx"] = 0; e10["op_idx"] = 0
        e10["market_idx"] = mi + 1
        save_state(st)

    df = pd.DataFrame(e10["results"])
    df.to_csv("./results/e10_operator_ate.csv", index=False)
    summary = df.groupby(["operator", "base_model"]).agg({
        "ate": ["mean", "std", "count"], "mse_base": "mean",
        "mse_treat": "mean", "mse_ctrl": "mean",
    }).reset_index()
    summary.to_csv("./results/e10_operator_ate_summary.csv", index=False)
    print(f"E10 COMPLETE: {len(df)} entries")
    return True


# ============ Main dispatch ============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["e8", "e9", "e10", "all"], required=True)
    parser.add_argument("--max_seconds", type=int, default=250)
    args = parser.parse_args()

    if args.exp == "e8":
        done = run_e8_chunk(args.max_seconds)
    elif args.exp == "e9":
        done = run_e9_chunk(args.max_seconds)
    elif args.exp == "e10":
        done = run_e10_chunk(args.max_seconds)
    elif args.exp == "all":
        # E9 first (fast), then E10, then E8
        print("=== Running E9 ===")
        done9 = run_e9_chunk(args.max_seconds)
        if done9:
            print("=== Running E10 ===")
            done10 = run_e10_chunk(args.max_seconds)
            if done10:
                print("=== Running E8 ===")
                done8 = run_e8_chunk(args.max_seconds)
    print("Done.")
