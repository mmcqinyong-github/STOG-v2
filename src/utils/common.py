"""Common utilities for STOG-MetaMorph experiments."""
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(name: str) -> dict:
    """Load a YAML config file from configs/."""
    cfg_path = Path(__file__).parent.parent.parent / "configs" / f"{name}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_results(df, path: str):
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def compute_mse(yhat, y):
    return ((yhat - y) ** 2).mean().item()


def compute_mae(yhat, y):
    return (yhat - y).abs().mean().item()


def compute_rmse(yhat, y):
    return np.sqrt(compute_mse(yhat, y))


def compute_mape(yhat, y, eps=1e-8):
    return ((yhat - y).abs() / (y.abs() + eps)).mean().item()


def rank_models(results_df: dict, metric="mse", ascending=True):
    """Rank models by metric."""
    items = list(results_df.items())
    items.sort(key=lambda x: x[1][metric], reverse=not ascending)
    return {k: i + 1 for i, (k, _) in enumerate(items)}
