"""Evaluation metrics."""
import numpy as np
import torch


def mse(yhat, y):
    return ((yhat - y) ** 2).mean().item()


def mae(yhat, y):
    return (yhat - y).abs().mean().item()


def rmse(yhat, y):
    return np.sqrt(mse(yhat, y))


def mape(yhat, y, eps=1e-8):
    return ((yhat - y).abs() / (y.abs() + eps)).mean().item()


def crps_from_quantiles(q, taus, y):
    """Compute CRPS from quantile predictions.
    q: (B, H, Q), taus: (Q,), y: (B, H)
    """
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        taus = np.array(taus)
    # Integral of pinball loss
    B, H, Q = q.shape
    crps = 0.0
    for i in range(Q):
        tau = taus[i]
        diff = q[:, :, i] - y
        crps += np.mean(np.where(diff >= 0, tau * diff, (tau - 1) * diff))
    return crps / Q


def pinball_loss(q, tau, y):
    diff = q - y
    return torch.mean(torch.where(diff >= 0, tau * diff, (tau - 1) * diff)).item()


def degradation(mse_stress, mse_clean):
    """Relative degradation rate."""
    return (mse_stress - mse_clean) / (mse_clean + 1e-10)


def average_regret(losses, oracle_losses):
    """Average regret vs oracle."""
    return np.mean(losses - oracle_losses)


def spearman_rank_correlation(ranks_a, ranks_b):
    from scipy.stats import spearmanr
    return spearmanr(ranks_a, ranks_b)


def topk_hit_rate(pred_ranks, true_ranks, k=3):
    """Fraction of times top-k predictions contain true top-k."""
    pred_topk = set(pred_ranks[:k])
    true_topk = set(true_ranks[:k])
    return len(pred_topk & true_topk) / k
