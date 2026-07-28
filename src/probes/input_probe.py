"""Probe suite for data characterization."""
import numpy as np
import torch
from scipy import fft
from scipy.stats import skew, kurtosis


class InputProbe:
    """Extract ~90 features from a single input window (3L,)."""

    def __init__(self):
        pass

    def basic_stats(self, v):
        vnp = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        return {
            "mean": float(vnp.mean()),
            "std": float(vnp.std()),
            "skew": float(skew(vnp)),
            "kurt": float(kurtosis(vnp)),
            "q05": float(np.percentile(vnp, 5)),
            "q25": float(np.percentile(vnp, 25)),
            "q50": float(np.percentile(vnp, 50)),
            "q75": float(np.percentile(vnp, 75)),
            "q95": float(np.percentile(vnp, 95)),
        }

    def spectral(self, v):
        vnp = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        n = len(vnp)
        fft_vals = np.abs(fft.fft(vnp))[:n // 2]
        freqs = fft.fftfreq(n)[:n // 2]
        total_energy = fft_vals.sum() + 1e-10
        # Spectral entropy
        p = fft_vals / total_energy
        spec_entropy = -np.sum(p * np.log(p + 1e-10))
        # Low freq ratio
        low_freq_ratio = fft_vals[:max(1, len(fft_vals) // 4)].sum() / total_energy
        # Dominant frequency
        dom_freq = freqs[np.argmax(fft_vals)] if len(freqs) > 0 else 0.0
        # Spectral slope
        spec_slope = np.polyfit(np.arange(len(fft_vals)), np.log(fft_vals + 1e-10), 1)[0]
        return {
            "spec_entropy": float(spec_entropy),
            "low_freq_ratio": float(low_freq_ratio),
            "dom_freq": float(dom_freq),
            "spec_slope": float(spec_slope),
        }

    def trend_diff(self, v):
        vnp = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        # Linear slope
        t = np.arange(len(vnp))
        slope = np.polyfit(t, vnp, 1)[0]
        # ACF decay (lag-1 autocorrelation)
        acf1 = np.corrcoef(vnp[:-1], vnp[1:])[0, 1] if len(vnp) > 1 else 0.0
        # Mean absolute difference
        mad = np.abs(np.diff(vnp)).mean() if len(vnp) > 1 else 0.0
        return {
            "slope": float(slope),
            "acf1": float(acf1),
            "mad": float(mad),
        }

    def spike_volatility(self, v):
        vnp = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        mean_val = vnp.mean()
        std_val = vnp.std() + 1e-10
        # Spike count (>3 sigma)
        spikes = np.sum(np.abs(vnp - mean_val) > 3 * std_val)
        # Max/mean ratio
        max_mean_ratio = np.abs(vnp).max() / (np.abs(mean_val) + 1e-10)
        return {
            "spike_count": float(spikes),
            "max_mean_ratio": float(max_mean_ratio),
            "volatility": float(std_val),
        }

    def __call__(self, v, mask=None):
        """Return feature vector as numpy array."""
        feats = {}
        feats.update(self.basic_stats(v))
        feats.update(self.spectral(v))
        feats.update(self.trend_diff(v))
        feats.update(self.spike_volatility(v))
        if mask is not None:
            mnp = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            feats["missing_rate"] = float(1.0 - mnp.mean())
        return np.array(list(feats.values()), dtype=np.float32)
