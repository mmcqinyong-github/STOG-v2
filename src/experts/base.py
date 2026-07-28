"""Base Expert class and unified interface."""
import torch
import torch.nn as nn
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class OperatorGenomeCard:
    model_id: str
    name: str
    family: str
    temporal_basis: list
    spatial_basis: list
    robust_basis: list
    gate_basis: list
    spectral_affinity: dict
    spatial_affinity: dict
    regime_affinity: dict
    cost_tier: str = "medium"
    missing_robustness: float = 0.5
    source: str = "kept"


class BaseExpert(nn.Module):
    """Unified interface for all 50 experts."""
    genome_card: OperatorGenomeCard = None
    supports_quantile: bool = False

    def __init__(self, d_in: int, hidden: int = 256, drop: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.drop = drop
        # Standard projection
        self.proj = nn.Linear(d_in, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(drop)

    def encode(self, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input window to hidden representation."""
        if mask is not None:
            v = v * mask
        h = self.proj(v)
        h = self.norm(h)
        h = torch.relu(h)
        h = self.dropout(h)
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Point prediction head: (B, D) -> (B, H)."""
        raise NotImplementedError

    def quantile_head(self, h: torch.Tensor, taus: list[float]) -> torch.Tensor:
        """Quantile prediction head: (B, D) -> (B, H, Q)."""
        # Default: linear quantile head trained with pinball
        H = self.head(h).shape[1]
        Q = len(taus)
        q_proj = nn.Linear(h.shape[-1], H * Q, device=h.device)
        q = q_proj(h).view(h.shape[0], H, Q)
        return q

    def probe_hooks(self) -> dict[str, Callable]:
        """Return hooks for activation probe."""
        return {}

    def forward(self, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encode(v, mask)
        return self.head(h)


class QuantileLinearHead(nn.Module):
    """Shared quantile head."""
    def __init__(self, hidden: int, horizon: int, n_taus: int = 9):
        super().__init__()
        self.proj = nn.Linear(hidden, horizon * n_taus)
        self.horizon = horizon
        self.n_taus = n_taus

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.proj(h)
        return q.view(h.shape[0], self.horizon, self.n_taus)
