"""Router implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualScoreRouter(nn.Module):
    """s_i = static spectral mismatch + delta_hat * dynamic regime term."""

    def __init__(self, probe_dim=90, card_dim=20, hidden=128, n_experts=50):
        super().__init__()
        self.n_experts = n_experts
        self.static_proj = nn.Sequential(
            nn.Linear(card_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(probe_dim + card_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def static_score(self, probe_features: torch.Tensor, cards_emb: torch.Tensor) -> torch.Tensor:
        # probe_features: (B, probe_dim), cards_emb: (N, card_dim)
        # Simplified: use cards_emb only for static score
        return self.static_proj(cards_emb).squeeze(-1)  # (N,)

    def dynamic_score(self, probe_features: torch.Tensor, cards_emb: torch.Tensor) -> torch.Tensor:
        # probe_features: (B, probe_dim), cards_emb: (N, card_dim)
        B = probe_features.shape[0]
        N = cards_emb.shape[0]
        probe_exp = probe_features.unsqueeze(1).expand(B, N, -1)  # (B, N, probe_dim)
        cards_exp = cards_emb.unsqueeze(0).expand(B, N, -1)  # (B, N, card_dim)
        combined = torch.cat([probe_exp, cards_exp], dim=-1)  # (B, N, probe_dim+card_dim)
        return self.dynamic_mlp(combined).squeeze(-1)  # (B, N)

    def forward(self, probe_features, cards_emb, delta_hat=0.5):
        static = self.static_score(probe_features, cards_emb)  # (N,)
        dynamic = self.dynamic_score(probe_features, cards_emb)  # (B, N)
        # Expand static to match batch
        static = static.unsqueeze(0).expand_as(dynamic)
        scores = static + delta_hat * dynamic
        return scores


class SparseTopKRouter(nn.Module):
    """Top-K selection with temperature."""

    def __init__(self, n_experts=50, K=3, tau=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.K = K
        self.tau = tau

    def select(self, scores: torch.Tensor):
        """scores: (B, N) -> topk_indices (B, K), weights (B, K)"""
        topk_vals, topk_idx = torch.topk(scores, self.K, dim=-1)
        weights = F.softmax(topk_vals / self.tau, dim=-1)
        return topk_idx, weights
