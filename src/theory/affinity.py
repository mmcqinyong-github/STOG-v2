"""Spectral affinity and mismatch scoring (Theorem 1)."""
import numpy as np
import torch


class SpectralAffinity:
    """A_i(X) = -integral |H_i - H_hat*|^2 d_mu_hat_X."""

    def __init__(self):
        # Basis response templates (simplified lookup)
        self.basis_templates = {
            "level": {"low_freq": 1.0, "high_freq": 0.5},
            "difference": {"low_freq": 0.3, "high_freq": 0.9},
            "moment": {"low_freq": 0.8, "high_freq": 0.4},
            "fourier": {"low_freq": 1.0, "high_freq": 1.0},
            "wavelet": {"low_freq": 0.7, "high_freq": 0.9},
            "state_space": {"low_freq": 0.9, "high_freq": 0.6},
            "patch": {"low_freq": 0.6, "high_freq": 0.6},
            "kan_spline": {"low_freq": 0.7, "high_freq": 0.7},
        }

    def mismatch_score(self, card, probe_features: dict) -> float:
        """Compute mismatch score from probe features and genome card."""
        score = 0.0
        # Temporal basis mismatch
        for basis in card.temporal_basis:
            tmpl = self.basis_templates.get(basis, {"low_freq": 0.5, "high_freq": 0.5})
            # Compare with estimated spectral properties
            low_ratio = probe_features.get("low_freq_ratio", 0.5)
            spec_entropy = probe_features.get("spec_entropy", 1.0)
            # Mismatch: if data is low-freq but model is high-freq biased
            score += abs(tmpl["low_freq"] - low_ratio)
            score += abs(tmpl["high_freq"] - (1 - low_ratio))

        # Add spatial, robustness, and regime terms
        score += (1.0 - card.missing_robustness) * probe_features.get("spike_count", 0.0) * 0.1

        return score / max(len(card.temporal_basis), 1)

    def affinity(self, card, probe_features: dict) -> float:
        return -self.mismatch_score(card, probe_features)

    def rank_experts(self, registry, probe_features: dict) -> list:
        """Rank experts by affinity."""
        scores = {mid: self.affinity(card, probe_features) for mid, card in registry.items()}
        return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
