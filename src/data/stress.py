"""Stress injection for robustness testing (E8)."""
import torch
from torch import Tensor
import numpy as np


class StressInjector:
    """Inject controlled stress into data windows."""

    @staticmethod
    def missingness(v: Tensor, rate: float, pattern: str = "mcar", seed: int = 0) -> tuple[Tensor, Tensor]:
        """Return (v_masked, mask)."""
        rng = np.random.RandomState(seed)
        mask = torch.ones_like(v)
        if pattern == "mcar":
            flat_mask = mask.view(-1)
            n_drop = int(len(flat_mask) * rate)
            drop_idx = rng.choice(len(flat_mask), n_drop, replace=False)
            flat_mask[drop_idx] = 0.0
        elif pattern == "block":
            # Block missingness
            B, D = v.shape
            block_len = max(1, int(D * rate * 2))
            for b in range(B):
                start = rng.randint(0, max(1, D - block_len))
                mask[b, start:start + block_len] = 0.0
        v_masked = v * mask
        return v_masked, mask

    @staticmethod
    def covariate_noise(v: Tensor, sigma: float, seed: int = 0) -> Tensor:
        """Add Gaussian noise."""
        rng = torch.Generator(device=v.device)
        rng.manual_seed(seed)
        noise = torch.randn(v.shape, generator=rng, device=v.device) * sigma
        return v + noise

    @staticmethod
    def lookback_truncate(v: Tensor, new_L: int) -> Tensor:
        """Truncate input to shorter lookback."""
        return v[:, -new_L:]
