"""Synthetic spatio-temporal field generator for controlled theory validation."""
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable


@dataclass
class SynthConfig:
    T: int = 20000
    V: int = 8
    H: int = 24
    alpha: float = 1.0
    spatial_type: str = "lowrank"
    kappa_st: float = 0.0
    spike_rate: float = 0.0
    spike_amp: float = 0.0
    delta: float = 0.3
    missing: float = 0.0
    missing_type: str = "mcar"
    seed: int = 0
    # --- E2-v2 additions (backward compatible: off by default) ---
    # When True, alpha actually shapes the data spectrum (1/f^alpha colored
    # component) instead of only parameterizing the unused H_star. Higher
    # alpha -> stronger low-frequency dominance -> higher cond(Sigma_x) and,
    # after differencing whitens the series, lower cond(Sigma_dx).
    alpha_filter: bool = False
    alpha_strength: float = 1.0  # relative scale of the colored component
    # When True (requires alpha_filter), replace the regime-basis mixture by
    # the pure colored component (+ small white noise) for a clean alpha
    # gradient in E2-style condition-number studies.
    alpha_pure: bool = False
    # --- E4-v4 addition (backward compatible: off by default) ---
    # When True, rescale the regime-1 coefficient matrix so that both regimes
    # emit comparable output amplitude (Theorem-4 amplitude-balance scope).
    # Legacy default False keeps the ~6x regime-1 amplitude dominance.
    amplitude_balance: bool = False


class SpatioTemporalFieldGenerator:
    """X(t,v) = sum_kl a_kl phi_k(t) psi_l(v) + epsilon."""

    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

    def temporal_basis(self, K_t: int = 16) -> np.ndarray:
        """Fourier + trend + AR decay basis."""
        T = self.cfg.T
        t = np.arange(T) / T
        phi = []
        # Trend
        phi.append(t)
        phi.append(t ** 2)
        # Fourier
        for k in range(1, K_t - 1):
            phi.append(np.sin(2 * np.pi * k * t))
            phi.append(np.cos(2 * np.pi * k * t))
        phi = np.stack(phi[:K_t], axis=1)  # (T, K_t)
        return phi

    def spatial_basis(self, K_s: int = 4) -> np.ndarray:
        """Generate spatial basis psi_l(v) according to spatial_type."""
        V = self.cfg.V
        stype = self.cfg.spatial_type
        if stype == "aligned":
            psi = np.eye(V)[:K_s]
            if K_s > V:
                psi = np.vstack([psi, np.zeros((K_s - V, V))])
        elif stype == "lowrank":
            u = self.rng.randn(V, K_s)
            psi = u.T  # (K_s, V)
        elif stype == "misaligned":
            psi = self.rng.randn(K_s, V)
            # Make them nearly orthogonal but slightly misaligned
            q, _ = np.linalg.qr(psi.T)
            psi = q.T + self.rng.randn(K_s, V) * 0.1
        else:  # white
            psi = self.rng.randn(K_s, V)
        return psi

    def true_operator(self) -> Callable:
        """Return the true transfer function H*(omega, lambda)."""
        cfg = self.cfg

        def H_star(omega, lambda_):
            # Simple low-pass in time + spatial projection
            alpha = cfg.alpha
            H_t = 1.0 / (1.0 + np.abs(omega) ** alpha)
            # Spatial factor
            if cfg.spatial_type == "lowrank":
                H_s = np.exp(-lambda_)
            else:
                H_s = 1.0
            return H_t * H_s

        return H_star

    def sample_regimes(self) -> np.ndarray:
        """Two-regime switching sequence."""
        T = self.cfg.T
        delta = self.cfg.delta
        # Higher delta -> more overlap / faster switching
        switch_prob = delta
        regimes = np.zeros(T, dtype=int)
        for t in range(1, T):
            if self.rng.rand() < switch_prob:
                regimes[t] = 1 - regimes[t - 1]
            else:
                regimes[t] = regimes[t - 1]
        return regimes

    def colored_series(self, T: int, alpha: float, rng: np.random.RandomState) -> np.ndarray:
        """1/f^alpha colored noise via spectral synthesis (unit variance)."""
        white = rng.randn(T)
        W = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(T)
        amp = 1.0 / (np.power(freqs, alpha) + 1e-6)
        amp[0] = amp[1] if len(amp) > 1 else 1.0  # kill DC divergence
        colored = np.fft.irfft(W * amp, n=T)
        colored = (colored - colored.mean()) / (colored.std() + 1e-10)
        return colored

    def generate(self) -> dict:
        """Generate field and return data dict."""
        cfg = self.cfg
        K_t, K_s = 16, 4
        phi = self.temporal_basis(K_t)  # (T, K_t)
        psi = self.spatial_basis(K_s)  # (K_s, V)

        # Coefficients
        a = self.rng.randn(K_t, K_s) * (1.0 / np.sqrt(K_t * K_s))

        # Regimes with actual regime-dependent spectral characteristics
        regimes = self.sample_regimes()
        # Two different coefficient sets for two regimes
        a0 = a
        a1 = a + self.rng.randn(K_t, K_s) * 0.8  # regime 1 has different spectral structure

        # E4-v4: optional amplitude balance. Rescale a1 so its output amplitude
        # matches regime 0 (no extra RNG draws -> stream unchanged when off).
        if cfg.amplitude_balance:
            a1 = a1 * (np.linalg.norm(a0) / (np.linalg.norm(a1) + 1e-12))
        amp_ratio_measured = None

        # Build field regime-aware
        X = np.zeros((cfg.T, cfg.V))
        if cfg.alpha_filter and cfg.alpha_pure:
            # Pure 1/f^alpha field (E2-v2 controlled design)
            for v in range(cfg.V):
                X[:, v] = cfg.alpha_strength * self.colored_series(
                    cfg.T, cfg.alpha, self.rng)
            X += self.rng.randn(cfg.T, cfg.V) * 0.1
            eps = self.rng.randn(cfg.T, cfg.V)  # keep rng stream length stable
        else:
            for t in range(cfg.T):
                a_r = a0 if regimes[t] == 0 else a1
                X[t] = (phi[t:t+1] @ a_r @ psi).flatten()

            # E4-v4: measured per-regime output amplitude ratio (deterministic
            # part, before noise) for verification of the balance condition.
            if cfg.amplitude_balance:
                s0 = X[regimes == 0].std()
                s1 = X[regimes == 1].std()
                amp_ratio_measured = float(s1 / (s0 + 1e-12))
                print(f"[synthetic] amplitude_balance=True: measured "
                      f"amplitude ratio regime1/regime0 = {amp_ratio_measured:.3f}")

            # Add noise
            eps = self.rng.randn(cfg.T, cfg.V)
            if cfg.alpha > 0:
                pass  # simplified: white noise for now
            X += eps * 0.1

        # E2-v2: opt-in alpha-shaped spectral component. Adds a per-variable
        # 1/f^alpha colored series so that alpha stratifies the true temporal
        # covariance spectrum (off by default -> legacy behavior unchanged).
        if cfg.alpha_filter and not cfg.alpha_pure:
            base_scale = float(X.std()) + 1e-10
            for v in range(cfg.V):
                X[:, v] += cfg.alpha_strength * base_scale * self.colored_series(
                    cfg.T, cfg.alpha, self.rng)

        # Spike contamination
        if cfg.spike_rate > 0:
            n_spikes = int(cfg.T * cfg.V * cfg.spike_rate)
            spike_idx = self.rng.choice(cfg.T * cfg.V, n_spikes, replace=False)
            spike_vals = self.rng.choice([-1, 1], n_spikes) * cfg.spike_amp * X.std()
            X.flat[spike_idx] += spike_vals

        # Create windows for target variable v=0
        L, H = self.cfg.H, self.cfg.H  # Use H as lookback for synthetic
        inp_list, tgt_list = [], []
        for i in range(cfg.T - L - H):
            inp_list.append(X[i:i + L].flatten())
            tgt_list.append(X[i + L:i + L + H, 0])

        X_tensor = torch.tensor(X, dtype=torch.float32)
        inp_tensor = torch.stack([torch.tensor(v, dtype=torch.float32) for v in inp_list])
        tgt_tensor = torch.stack([torch.tensor(v, dtype=torch.float32) for v in tgt_list])

        # Train/val/test split
        n = len(inp_list)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)

        return {
            "X": X_tensor,
            "train_inp": inp_tensor[:n_train],
            "train_tgt": tgt_tensor[:n_train],
            "val_inp": inp_tensor[n_train:n_train + n_val],
            "val_tgt": tgt_tensor[n_train:n_train + n_val],
            "test_inp": inp_tensor[n_train + n_val:],
            "test_tgt": tgt_tensor[n_train + n_val:],
            "H_star": self.true_operator(),
            "regimes": regimes,
            "amplitude_ratio_r1_over_r0": amp_ratio_measured,
            "mu_X_true": {"alpha": cfg.alpha, "spatial_type": cfg.spatial_type,
                          "delta": cfg.delta, "spike_rate": cfg.spike_rate,
                          "alpha_filter": cfg.alpha_filter},
        }
