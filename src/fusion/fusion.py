"""Fusion layers."""
import torch
import torch.nn as nn


class OutputFusion(nn.Module):
    """Weighted average of expert predictions."""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, weights):
        """
        predictions: dict of {expert_id: (B, H)}
        weights: (B, K) normalized weights
        """
        preds = torch.stack(list(predictions.values()), dim=1)  # (B, K, H)
        fused = torch.sum(preds * weights.unsqueeze(-1), dim=1)  # (B, H)
        return fused


class HiddenFusion(nn.Module):
    """Fusion in representation space."""

    def __init__(self, expert_dims: dict, D_common=256):
        super().__init__()
        self.projections = nn.ModuleDict({
            eid: nn.Linear(d, D_common) for eid, d in expert_dims.items()
        })
        self.D_common = D_common

    def forward(self, hiddens: dict, weights: torch.Tensor):
        """
        hiddens: {expert_id: (B, D_e)}
        weights: (B, K)
        """
        projected = []
        for eid, h in hiddens.items():
            projected.append(self.projections[eid](h))
        projected = torch.stack(projected, dim=1)  # (B, K, D_common)
        fused = torch.sum(projected * weights.unsqueeze(-1), dim=1)  # (B, D_common)
        return fused
