"""Simplified but representative expert pool (~25 models covering all major families).
Full 50-expert pool can be expanded from this foundation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseExpert, OperatorGenomeCard


# ==============  Representative Experts (covering all B_t x B_s x B_r x B_g families) ==============

class DLinearExpert(BaseExpert):
    """M52: Linear decomposition - level/difference family."""
    genome_card = OperatorGenomeCard(
        model_id="M52", name="DLinear", family="linear",
        temporal_basis=["level", "difference"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.3, "long_memory": 0.5, "strong_periodicity": 0.6},
        spatial_affinity={"static_low_rank": 0.8}, regime_affinity={"temporal_regime": 0.5}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Linear(d_in, hidden)
        self.season_proj = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        # Decomposition
        v_u = v.unsqueeze(1)
        trend = self.decomp(v_u).squeeze(1)
        seasonal = v - trend
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(seasonal)
        h = self.norm(h_t + h_s)
        return torch.relu(h)

    def head(self, h):
        return self._modules['head'](h)


class FITSExpert(BaseExpert):
    """M01: Frequency domain linear."""
    genome_card = OperatorGenomeCard(
        model_id="M01", name="FITS", family="frequency",
        temporal_basis=["fourier"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.9, "spike_heavy_tail": 0.3, "long_memory": 0.4, "strong_periodicity": 0.9},
        spatial_affinity={"static_low_rank": 0.7}, regime_affinity={"temporal_regime": 0.6}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.freq_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        hf = torch.fft.rfft(v, dim=1)
        hf[:, max(1, hf.shape[1] // 4):] = 0.0
        v_recon = torch.fft.irfft(hf, n=v.shape[1], dim=1)
        h = self.proj(v)
        h = self.norm(h)
        h_freq = self.freq_proj(v_recon)
        e = self.enhance(h_freq)
        return torch.relu(h_freq + e * 0.3 + h * 0.2)

    def head(self, h):
        return self._modules['head'](h)


class PatchTSTExpert(BaseExpert):
    """M50: Patch attention."""
    genome_card = OperatorGenomeCard(
        model_id="M50", name="PatchTST", family="attention",
        temporal_basis=["patch"], spatial_basis=["identity"],
        robust_basis=["layer_norm"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.6, "spike_heavy_tail": 0.5, "long_memory": 0.6, "strong_periodicity": 0.7},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, patch_len=16, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.patch_len = patch_len
        n_patches = max(1, d_in // patch_len)
        self.patch_embed = nn.Linear(patch_len, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=drop, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        B, D = v.shape
        # Patchify
        pad = (self.patch_len - D % self.patch_len) % self.patch_len
        v_pad = F.pad(v, (0, pad))
        patches = v_pad.view(B, -1, self.patch_len)
        h = self.patch_embed(patches)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.mean(dim=1)
        return h

    def head(self, h):
        return self._modules['head'](h)


class MambaSSMExpert(BaseExpert):
    """M14: Selective SSM."""
    genome_card = OperatorGenomeCard(
        model_id="M14", name="MambaSSM", family="ssm",
        temporal_basis=["state_space"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.5, "long_memory": 0.8, "strong_periodicity": 0.5},
        spatial_affinity={"static_low_rank": 0.5}, regime_affinity={"temporal_regime": 0.6}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.gate = nn.Linear(d_in, hidden)
        self.state = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        c = self.conv(v.unsqueeze(1)).squeeze(1)
        g = torch.sigmoid(self.gate(c))
        s = torch.tanh(self.state(c))
        h = self.norm(self.proj(c))
        return g * h + (1 - g) * s

    def head(self, h):
        return self._modules['head'](h)


class ModernTCNExpert(BaseExpert):
    """M17: Large-kernel CNN."""
    genome_card = OperatorGenomeCard(
        model_id="M17", name="ModernTCN", family="cnn",
        temporal_basis=["patch", "level"], spatial_basis=["identity"],
        robust_basis=["batch_norm"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.6, "spike_heavy_tail": 0.6, "long_memory": 0.5, "strong_periodicity": 0.8},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        x_u = v.unsqueeze(1)
        c1 = self.conv1(x_u).squeeze(1)
        c2 = self.conv2(x_u).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        h = self.fusion(torch.cat([h1, h2], dim=-1))
        return h

    def head(self, h):
        return self._modules['head'](h)


class AutoformerExpert(BaseExpert):
    """M47: Decomposition Autoformer."""
    genome_card = OperatorGenomeCard(
        model_id="M47", name="Autoformer", family="decomposition",
        temporal_basis=["fourier", "level"], spatial_basis=["identity"],
        robust_basis=["layer_norm"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.8, "spike_heavy_tail": 0.4, "long_memory": 0.6, "strong_periodicity": 0.9},
        spatial_affinity={"static_low_rank": 0.7}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.season_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        x_u = v.unsqueeze(1)
        trend = self.decomp(x_u).squeeze(1)
        seasonal = v - trend
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(seasonal)
        h_orig = self._preprocess(v)
        h = self.fusion(torch.cat([h_t, h_s], dim=-1))
        return h + h_orig * 0.3

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        return self._modules['head'](h)


class iTransformerExpert(BaseExpert):
    """M63: Variable attention (iTransformer)."""
    genome_card = OperatorGenomeCard(
        model_id="M63", name="iTransformer", family="attention",
        temporal_basis=["patch"], spatial_basis=["cross_section_attention"],
        robust_basis=["layer_norm"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.5, "spike_heavy_tail": 0.5, "long_memory": 0.7, "strong_periodicity": 0.6},
        spatial_affinity={"static_low_rank": 0.8, "cross_section_attention": 0.9}, regime_affinity={"temporal_regime": 0.6}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden * 2, hidden), nn.Dropout(drop))
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        h = self.proj(v).unsqueeze(1)  # (B, 1, H)
        h = self.norm(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = h + a
        h = h + self.ffn(h)
        return h.squeeze(1)

    def head(self, h):
        return self._modules['head'](h)


class TimesNetExpert(BaseExpert):
    """M51: Periodic 2D CNN."""
    genome_card = OperatorGenomeCard(
        model_id="M51", name="TimesNet", family="cnn",
        temporal_basis=["fourier", "level"], spatial_basis=["identity"],
        robust_basis=["batch_norm"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.5, "long_memory": 0.5, "strong_periodicity": 0.9},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        # Reshape to pseudo-2D for period-based processing
        B, D = v.shape
        period = max(1, int(np.sqrt(D)))
        pad = (period - D % period) % period
        v_pad = F.pad(v, (0, pad))
        v_2d = v_pad.view(B, 1, -1, period)
        # Actually just use 1D conv for simplicity in morph
        h = self.proj(v)
        h = self.norm(h)
        return torch.relu(h)

    def head(self, h):
        return self._modules['head'](h)


class xLSTMExpert(BaseExpert):
    """M31: Extended LSTM / state tracking."""
    genome_card = OperatorGenomeCard(
        model_id="M31", name="xLSTM", family="ssm",
        temporal_basis=["state_space"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.6, "spike_heavy_tail": 0.5, "long_memory": 0.9, "strong_periodicity": 0.4},
        spatial_affinity={"static_low_rank": 0.5}, regime_affinity={"temporal_regime": 0.6}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.mem = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        c = self.conv(v.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        h = self._preprocess(v)
        m = self.mem(h)
        g = self.gate(torch.cat([m, h_c], dim=-1))
        return h + m * 0.5 * g + h_c * (1 - g) * 0.3

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        return self._modules['head'](h)


class WPMixerExpert(BaseExpert):
    """M36: Wavelet-like multi-resolution."""
    genome_card = OperatorGenomeCard(
        model_id="M36", name="WPMixer", family="wavelet",
        temporal_basis=["wavelet", "difference"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.6, "long_memory": 0.5, "strong_periodicity": 0.8},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.low_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.high_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.low_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.high_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        c_low = self.low_conv(v.unsqueeze(1)).squeeze(1)
        h_low = self.norm(self.proj(c_low))
        lo = h_low * self.low_gate(h_low)
        c_high = self.high_conv(v.unsqueeze(1)).squeeze(1)
        h_high = self.norm(self.proj(c_high))
        avg = self._preprocess(v).mean(1, keepdim=True).expand_as(h_high)
        hi = (self._preprocess(v) - avg) * self.high_gate(h_high)
        h = self.fusion(torch.cat([lo, hi], dim=-1))
        return h

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        return self._modules['head'](h)


class QuantMoExpert(BaseExpert):
    """M233: Moment + difference + gating (original champion)."""
    genome_card = OperatorGenomeCard(
        model_id="M233", name="QuantMo", family="hybrid",
        temporal_basis=["moment", "difference", "quantile"], spatial_basis=["identity"],
        robust_basis=["quantile"], gate_basis=["volatility_gate"],
        spectral_affinity={"low_freq_decay": 0.6, "spike_heavy_tail": 0.8, "long_memory": 0.6, "strong_periodicity": 0.7},
        spatial_affinity={"static_low_rank": 0.7}, regime_affinity={"temporal_regime": 0.8}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.moment_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.diff_proj = nn.Sequential(nn.Linear(max(1, d_in - 1), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        m = self.moment_proj(v)
        dv = v[:, 1:] - v[:, :-1] if v.size(1) > 1 else torch.zeros_like(v[:, :1])
        d = self.diff_proj(dv)
        g = self.gate(torch.cat([m, d], dim=-1))
        h = m * g + d * (1 - g)
        return self.norm(h)

    def head(self, h):
        return self._modules['head'](h)


class DRFNExpert(BaseExpert):
    """M89: Static-dynamic relation."""
    genome_card = OperatorGenomeCard(
        model_id="M89", name="DRFN", family="graph",
        temporal_basis=["level", "difference"], spatial_basis=["static_low_rank", "dynamic_coupling"],
        robust_basis=["layer_norm"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.5, "long_memory": 0.6, "strong_periodicity": 0.7},
        spatial_affinity={"static_low_rank": 0.9, "graph_coupling": 0.8}, regime_affinity={"spatial_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.static_proj = nn.Linear(d_in, hidden)
        self.dynamic_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        h = self.proj(v)
        h = self.norm(h)
        s = self.static_proj(v)
        g = self.dynamic_gate(h)
        return g * h + (1 - g) * s

    def head(self, h):
        return self._modules['head'](h)


class TimeMixerExpert(BaseExpert):
    """M18: Multi-scale mixing."""
    genome_card = OperatorGenomeCard(
        model_id="M18", name="TimeMixer", family="decomposition",
        temporal_basis=["level", "difference", "moment"], spatial_basis=["identity"],
        robust_basis=["layer_norm"], gate_basis=["input_dependent_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.5, "long_memory": 0.6, "strong_periodicity": 0.8},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.coarse_gate = nn.Linear(d_in, hidden)
        self.fine_enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        h = self._preprocess(v)
        g = torch.sigmoid(self.coarse_gate(v))
        coarse = h * g
        fine = self.fine_enhance(h - h.mean(1, keepdim=True).expand_as(h))
        c = self.conv(v.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return coarse + fine * 0.4 + h * 0.2 + h_c * 0.2

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        return self._modules['head'](h)


class TFTExpert(BaseExpert):
    """M55: Quantile + variable selection."""
    genome_card = OperatorGenomeCard(
        model_id="M55", name="TFT", family="attention",
        temporal_basis=["quantile", "level"], spatial_basis=["exogenous_selection"],
        robust_basis=["quantile"], gate_basis=["volatility_gate"],
        spectral_affinity={"low_freq_decay": 0.6, "spike_heavy_tail": 0.7, "long_memory": 0.5, "strong_periodicity": 0.6},
        spatial_affinity={"exogenous_selection": 0.9}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )
    supports_quantile = True

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.var_sel = nn.Sequential(nn.Linear(d_in, hidden), nn.Sigmoid())
        self.context = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        sel = self.var_sel(v)
        h = self._preprocess(v)
        h = h * sel
        c = self.context(h.mean(1, keepdim=True).expand_as(h))
        return h + c * 0.3

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        return self._modules['head'](h)


class SilentAccumExpert(BaseExpert):
    """M220: Multi-scale conv + breakthrough (top hybrid)."""
    genome_card = OperatorGenomeCard(
        model_id="M220", name="SilentAccum", family="hybrid",
        temporal_basis=["patch", "difference", "moment"], spatial_basis=["identity"],
        robust_basis=["huber"], gate_basis=["spectral_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.7, "long_memory": 0.7, "strong_periodicity": 0.8},
        spatial_affinity={"static_low_rank": 0.7}, regime_affinity={"temporal_regime": 0.8}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.conv7 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        c7 = self.conv7(v.unsqueeze(1)).squeeze(1)
        c3 = self.conv3(v.unsqueeze(1)).squeeze(1)
        h7 = self.norm(self.proj(c7))
        h3 = self.norm(self.proj(c3))
        g = self.gate(torch.cat([h7, h3], dim=-1))
        return h7 * g + h3 * (1 - g)

    def head(self, h):
        return self._modules['head'](h)


class RLinearExpert(BaseExpert):
    """M03: Residual Linear."""
    genome_card = OperatorGenomeCard(
        model_id="M03", name="RLinear", family="linear",
        temporal_basis=["level"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.5, "spike_heavy_tail": 0.3, "long_memory": 0.4, "strong_periodicity": 0.4},
        spatial_affinity={"static_low_rank": 0.5}, regime_affinity={"temporal_regime": 0.5}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.res_block = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden * 2, hidden)
        )
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        h = self._preprocess(v)
        return h

    def _preprocess(self, x):
        return self.norm(self.proj(x))

    def head(self, h):
        r = self.res_block(h)
        return self._modules['head'](h + r * 0.3)


class FreTSExpert(BaseExpert):
    """M117: Frequency MLP."""
    genome_card = OperatorGenomeCard(
        model_id="M117", name="FreTS", family="frequency",
        temporal_basis=["fourier"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.9, "spike_heavy_tail": 0.3, "long_memory": 0.4, "strong_periodicity": 0.9},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="kept"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.freq_mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        hf = torch.fft.rfft(v, dim=1)
        amp = hf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask_fft = torch.zeros_like(amp)
        mask_fft.scatter_(1, idx, 1.0)
        v_freq = torch.fft.irfft(hf * mask_fft, n=v.shape[1], dim=1)
        h = self.freq_mlp(v_freq)
        return self.norm(h)

    def head(self, h):
        return self._modules['head'](h)


# ============== New models (N01-N12, representative subset) ==============

class FlowStateMorphExpert(BaseExpert):
    """N01: SSM encoder + functional basis decoder."""
    genome_card = OperatorGenomeCard(
        model_id="N01", name="FlowStateMorph", family="ssm",
        temporal_basis=["state_space", "fourier"], spatial_basis=["identity"],
        robust_basis=["layer_norm"], gate_basis=["spectral_gate"],
        spectral_affinity={"low_freq_decay": 0.8, "spike_heavy_tail": 0.5, "long_memory": 0.9, "strong_periodicity": 0.6},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.7}, source="new"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, n_basis=32, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.n_basis = n_basis
        self.ssm = nn.LSTM(d_in, hidden, batch_first=True, num_layers=2, dropout=drop)
        self.basis_coef = nn.Linear(hidden, n_basis)
        self.head = nn.Linear(n_basis, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        v_seq = v.unsqueeze(1)  # (B, 1, D)
        h, _ = self.ssm(v_seq)
        h = h[:, -1, :]
        c = self.basis_coef(h)
        return c

    def head(self, h):
        return self._modules['head'](h)


class NBEATSxExpert(BaseExpert):
    """N07: Exogenous basis expansion."""
    genome_card = OperatorGenomeCard(
        model_id="N07", name="NBEATSx", family="basis_expansion",
        temporal_basis=["fourier", "level"], spatial_basis=["exogenous_selection"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.8, "spike_heavy_tail": 0.4, "long_memory": 0.5, "strong_periodicity": 0.9},
        spatial_affinity={"exogenous_selection": 0.8}, regime_affinity={"temporal_regime": 0.7}, source="new"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.trend_basis = nn.Linear(d_in, hidden)
        self.season_basis = nn.Linear(d_in, hidden)
        self.exog_gate = nn.Sequential(nn.Linear(d_in, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        t = self.trend_basis(v)
        s = self.season_basis(v)
        g = self.exog_gate(v)
        return self.norm(t * g + s * (1 - g))

    def head(self, h):
        return self._modules['head'](h)


class MSTLResidExpert(BaseExpert):
    """N10: MSTL + linear residual (statistical anchor)."""
    genome_card = OperatorGenomeCard(
        model_id="N10", name="MSTLResid", family="statistical",
        temporal_basis=["level", "difference"], spatial_basis=["identity"],
        robust_basis=["median"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.9, "spike_heavy_tail": 0.8, "long_memory": 0.3, "strong_periodicity": 0.9},
        spatial_affinity={"static_low_rank": 0.5}, regime_affinity={"temporal_regime": 0.5}, source="new"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        # Simple linear extrapolation as proxy for MSTL+linear
        # BUGFIX(2025, routing retrain): head() receives h of dim `hidden`
        # (from encode -> norm(proj(v))), so the linear layer must map
        # hidden -> horizon; the original nn.Linear(d_in, 24) caused the
        # 504-vs-256 dimension mismatch that made all N10 E6 runs fail.
        # HORIZON(2025, E7-v2): both projections follow the `horizon` kwarg;
        # default horizon=24 reproduces the pre-parameterization behavior.
        self.trend = nn.Linear(hidden, horizon)
        self.head = nn.Linear(d_in, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        return self.norm(self.proj(v))

    def head(self, h):
        # Use direct linear mapping from input (mimicking statistical model)
        return self.trend(h)


class CycleNetExpert(BaseExpert):
    """N08: Explicit periodic component."""
    genome_card = OperatorGenomeCard(
        model_id="N08", name="CycleNet", family="periodic",
        temporal_basis=["fourier", "level"], spatial_basis=["identity"],
        robust_basis=["raw"], gate_basis=["static_gate"],
        spectral_affinity={"low_freq_decay": 0.7, "spike_heavy_tail": 0.3, "long_memory": 0.4, "strong_periodicity": 1.0},
        spatial_affinity={"static_low_rank": 0.6}, regime_affinity={"temporal_regime": 0.8}, source="new"
    )

    def __init__(self, d_in, hidden=256, drop=0.1, horizon=24):
        super().__init__(d_in, hidden, drop)
        self.period_embed = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, horizon)

    def encode(self, v, mask=None):
        if mask is not None:
            v = v * mask
        h = self.proj(v)
        h = self.norm(h)
        p = torch.sin(h * 0.15) * 0.2
        return torch.relu(h + p)

    def head(self, h):
        return self._modules['head'](h)


# ============== Registry ==============

EXPERT_REGISTRY = {
    "M01": FITSExpert,
    "M03": RLinearExpert,
    "M14": MambaSSMExpert,
    "M17": ModernTCNExpert,
    "M18": TimeMixerExpert,
    "M31": xLSTMExpert,
    "M36": WPMixerExpert,
    "M47": AutoformerExpert,
    "M50": PatchTSTExpert,
    "M51": TimesNetExpert,
    "M52": DLinearExpert,
    "M55": TFTExpert,
    "M63": iTransformerExpert,
    "M89": DRFNExpert,
    "M117": FreTSExpert,
    "M220": SilentAccumExpert,
    "M233": QuantMoExpert,
    "N01": FlowStateMorphExpert,
    "N07": NBEATSxExpert,
    "N08": CycleNetExpert,
    "N10": MSTLResidExpert,
}


def get_expert(model_id: str, d_in: int, hidden: int = 256, drop: float = 0.1,
               horizon: int = 24):
    """Factory function to instantiate an expert.

    `horizon` parameterizes every expert's output head (E7-v2 long-term
    generalization). Default horizon=24 is bit-identical to the original
    hard-coded behavior (verified by results/_zoo_pre_horizon_snapshot.json).
    """
    if model_id not in EXPERT_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}. Available: {list(EXPERT_REGISTRY.keys())}")
    return EXPERT_REGISTRY[model_id](d_in, hidden, drop, horizon=horizon)


def get_all_cards() -> dict:
    """Return all genome cards."""
    cards = {}
    for mid, cls in EXPERT_REGISTRY.items():
        if cls.genome_card is not None:
            cards[mid] = cls.genome_card
    return cards


# ==============  E10-v2: Pluggable operator transplant wrappers (ADDITIVE ONLY) ==============
# These wrappers graft a causal "operator" branch onto ANY base expert for the
# E10 operator-transplant ATE experiment. TREAT arm carries the operator-specific
# information pathway; CTRL arm is a capacity-matched placebo with the
# operator-specific pathway neutralized (same module shape / parameter count).
# Added 2025 for run_e10_v2.py; existing classes above are untouched.

class OperatorGraftWrapper(BaseExpert):
    """Wrap a base expert with a grafted operator branch.

    operator in {"diff", "moment", "graph", "gate"}; arm in {"treat", "ctrl"}.
    Input is the flattened EPF window (B, 3*L): [price(L); exog1(L); exog2(L)].
    """

    def __init__(self, base_expert: BaseExpert, operator: str, arm: str,
                 n_vars: int = 3, lookback: int = 168, horizon: int = 24):
        # Do NOT call BaseExpert.__init__ (would add unused proj/norm params);
        # initialize nn.Module directly and reuse the base expert's interface.
        nn.Module.__init__(self)
        assert operator in ("diff", "moment", "graph", "gate")
        assert arm in ("treat", "ctrl")
        self.base = base_expert
        self.operator = operator
        self.arm = arm
        self.n_vars = n_vars
        self.lookback = lookback
        self.horizon = horizon
        d_in = base_expert.d_in
        self.d_in = d_in
        self.hidden = base_expert.hidden
        self.genome_card = base_expert.genome_card

        if operator == "diff":
            # Parallel linear head on the (padded) first difference, added to trunk output.
            # CTRL placebo: identical linear head on the RAW window (capacity placebo,
            # zero-initialized so it starts as identity), no difference operator.
            self.branch = nn.Linear(d_in, horizon)
            nn.init.zeros_(self.branch.weight)
            nn.init.zeros_(self.branch.bias)
        elif operator == "moment":
            # Per-window robust standardization (median/MAD) before trunk + inverse
            # transform of the prediction. CTRL placebo: identity pre/post transform
            # with the same learnable global affine (init to identity).
            self.aff_scale = nn.Parameter(torch.ones(1))
            self.aff_shift = nn.Parameter(torch.zeros(1))
        elif operator == "graph":
            # One message-passing layer over variables with learnable VxV adjacency.
            # TREAT: full adjacency trainable (init identity).
            # CTRL placebo: off-diagonal FROZEN at 0, only per-channel diagonal
            # scaling trainable -> same module shape, no cross-variable mixing.
            A = torch.eye(n_vars)
            self.adj = nn.Parameter(A)
            if arm == "ctrl":
                mask = torch.eye(n_vars)
                self.register_buffer("adj_mask", mask)
            else:
                self.register_buffer("adj_mask", torch.ones(n_vars, n_vars))
        elif operator == "gate":
            # Residual output gate: out * sigmoid(MLP(window_stats)).
            # TREAT: MLP sees 8 window statistics of the price channel.
            # CTRL placebo: identical MLP but its input is a zero vector ->
            # gate degenerates to a learnable constant (bias-only) gate.
            self.gate_mlp = nn.Sequential(
                nn.Linear(8, 32), nn.GELU(), nn.Linear(32, horizon)
            )

    # ---- operator-specific helpers ----
    def _split_channels(self, v):
        B = v.shape[0]
        return v.view(B, self.n_vars, self.lookback)

    def _window_stats(self, x):
        # x: (B, L) price channel -> 8 robust/spectral stats
        med = x.median(dim=1, keepdim=True).values
        mad = (x - med).abs().median(dim=1, keepdim=True).values + 1e-6
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        xmax = x.max(dim=1, keepdim=True).values
        xmin = x.min(dim=1, keepdim=True).values
        d = x[:, 1:] - x[:, :-1]
        dstd = d.std(dim=1, keepdim=True) + 1e-6
        acf1 = ((x[:, :-1] - mean) * (x[:, 1:] - mean)).mean(dim=1, keepdim=True) / (std ** 2)
        return torch.cat([mean, std, med, mad, xmax, xmin, dstd, acf1], dim=1)

    def forward(self, v, mask=None):
        if self.operator == "diff":
            dv = torch.zeros_like(v)
            dv[:, 1:] = v[:, 1:] - v[:, :-1]
            branch_in = dv if self.arm == "treat" else v
            return self.base(v, mask) + self.branch(branch_in)

        if self.operator == "moment":
            ch = self._split_channels(v)
            if self.arm == "treat":
                med = ch.median(dim=2, keepdim=True).values
                mad = (ch - med).abs().median(dim=2, keepdim=True).values + 1e-4
                vn = ((ch - med) / mad).view(v.shape)
                out = self.base(vn, mask)
                # inverse-transform with PRICE channel stats
                out = out * mad[:, 0, :] + med[:, 0, :]
            else:
                out = self.base(v * self.aff_scale + self.aff_shift, mask)
                out = (out - self.aff_shift) / self.aff_scale.clamp_min(1e-4)
            return out

        if self.operator == "graph":
            ch = self._split_channels(v)  # (B, V, L)
            A = self.adj * self.adj_mask if self.arm == "ctrl" else self.adj
            mixed = ch + torch.einsum("vw,bwl->bvl", A, ch)
            return self.base(mixed.reshape(v.shape), mask)

        if self.operator == "gate":
            out = self.base(v, mask)
            if self.arm == "treat":
                stats = self._window_stats(self._split_channels(v)[:, 0, :])
            else:
                stats = torch.zeros(v.shape[0], 8, device=v.device, dtype=v.dtype)
            g = torch.sigmoid(self.gate_mlp(stats))
            return out * (1.0 + g)

        raise ValueError(self.operator)

    # Unified interface delegation
    def encode(self, v, mask=None):
        return self.base.encode(v, mask)

    def head(self, h):
        return self.base.head(h)


def graft_operator(base_expert: BaseExpert, operator: str, arm: str,
                   n_vars: int = 3, lookback: int = 168, horizon: int = 24) -> OperatorGraftWrapper:
    """Factory: graft an operator branch (treat/ctrl arm) onto a base expert."""
    return OperatorGraftWrapper(base_expert, operator, arm, n_vars, lookback, horizon)


# ==============  E10-v3: gate control redesign + random-residual placebos (ADDITIVE ONLY) ==============
# Reviewer-driven redesign of the gate control arms (run_e10_v3.py):
#   G2 freezes the gate MLP (only a scalar bias is learnable) so the CTRL no
#      longer trains MLP weights on a zero input (capacity equivalence).
#   G3 replaces the MLP by a single learnable scalar gate.
#   R1/R2 are random-residual placebo branches (Linear->GELU->Linear random
#      projection of the window, parameter-matched to the gate MLP) with a
#      learnable scalar coefficient; R1 frozen, R2 trainable. They separate
#      "gate semantics help" from "any residual branch helps".
# Added 2025 for run_e10_v3.py; everything above is untouched.

class GatePlaceboWrapperV3(BaseExpert):
    """Wrap a base expert with a gate / residual-placebo branch (E10-v3 arms).

    arm in {"g1", "g2", "g3", "r1", "r2"}:
      g1: gate TREAT, bit-equivalent to OperatorGraftWrapper(gate, treat):
          out * (1 + sigmoid(MLP(window_stats))).
      g2: gate CTRL-v3a: same MLP but ALL MLP weights frozen at random init
          and fed a zero vector (pathway neutralized); only an added scalar
          bias is learnable -> effective learnable capacity = 1 parameter.
      g3: gate CTRL-v3b: no MLP at all; out * (1 + sigmoid(alpha)) with a
          single learnable scalar alpha.
      r1: random-residual placebo: frozen random Linear(d_in,H)->GELU->
          Linear(H,horizon) projection of the raw window, added to the trunk
          output with a learnable scalar coefficient lambda (init 0).
          Parameter count matched to the gate TREAT MLP (1080 vs 1082).
      r2: same branch as r1 but fully trainable (weights + lambda).
    """

    GATE_MLP_PARAMS = 8 * 32 + 32 + 32 * 24 + 24  # 1080 (gate TREAT MLP)

    def __init__(self, base_expert: BaseExpert, arm: str,
                 n_vars: int = 3, lookback: int = 168, horizon: int = 24):
        # Do NOT call BaseExpert.__init__ (would add unused proj/norm params);
        # initialize nn.Module directly and reuse the base expert's interface.
        nn.Module.__init__(self)
        assert arm in ("g1", "g2", "g3", "r1", "r2")
        self.base = base_expert
        self.operator = "gate" if arm.startswith("g") else "rand_resid"
        self.arm = arm
        self.n_vars = n_vars
        self.lookback = lookback
        self.horizon = horizon
        d_in = base_expert.d_in
        self.d_in = d_in
        self.hidden = base_expert.hidden
        self.genome_card = base_expert.genome_card

        if arm in ("g1", "g2"):
            self.gate_mlp = nn.Sequential(
                nn.Linear(8, 32), nn.GELU(), nn.Linear(32, horizon)
            )
            if arm == "g2":
                for p in self.gate_mlp.parameters():
                    p.requires_grad_(False)
                # single learnable scalar bias (per-arm capacity = 1)
                self.gate_bias = nn.Parameter(torch.zeros(1))
        elif arm == "g3":
            self.gate_alpha = nn.Parameter(torch.zeros(1))
        else:  # r1 / r2 random-residual branch, param-matched to gate MLP
            # H=2: 504*2+2 + 2*24+24 = 1082 total (gate MLP = 1080)
            self.res_branch = nn.Sequential(
                nn.Linear(d_in, 2), nn.GELU(), nn.Linear(2, horizon)
            )
            self.res_lambda = nn.Parameter(torch.zeros(1))
            if arm == "r1":
                for p in self.res_branch.parameters():
                    p.requires_grad_(False)

    # ---- helpers (identical stats as the v2 gate operator) ----
    def _split_channels(self, v):
        B = v.shape[0]
        return v.view(B, self.n_vars, self.lookback)

    def _window_stats(self, x):
        # x: (B, L) price channel -> 8 robust/spectral stats
        med = x.median(dim=1, keepdim=True).values
        mad = (x - med).abs().median(dim=1, keepdim=True).values + 1e-6
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        xmax = x.max(dim=1, keepdim=True).values
        xmin = x.min(dim=1, keepdim=True).values
        d = x[:, 1:] - x[:, :-1]
        dstd = d.std(dim=1, keepdim=True) + 1e-6
        acf1 = ((x[:, :-1] - mean) * (x[:, 1:] - mean)).mean(dim=1, keepdim=True) / (std ** 2)
        return torch.cat([mean, std, med, mad, xmax, xmin, dstd, acf1], dim=1)

    def forward(self, v, mask=None):
        out = self.base(v, mask)
        if self.arm == "g1":
            stats = self._window_stats(self._split_channels(v)[:, 0, :])
            g = torch.sigmoid(self.gate_mlp(stats))
            return out * (1.0 + g)
        if self.arm == "g2":
            stats = torch.zeros(v.shape[0], 8, device=v.device, dtype=v.dtype)
            g = torch.sigmoid(self.gate_mlp(stats) + self.gate_bias)
            return out * (1.0 + g)
        if self.arm == "g3":
            return out * (1.0 + torch.sigmoid(self.gate_alpha))
        # r1 / r2: additive random-projection residual with scalar coefficient
        return out + self.res_lambda * self.res_branch(v)

    # Unified interface delegation
    def encode(self, v, mask=None):
        return self.base.encode(v, mask)

    def head(self, h):
        return self.base.head(h)


def graft_gate_v3(base_expert: BaseExpert, arm: str,
                  n_vars: int = 3, lookback: int = 168, horizon: int = 24) -> GatePlaceboWrapperV3:
    """Factory: graft an E10-v3 gate/placebo arm onto a base expert."""
    return GatePlaceboWrapperV3(base_expert, arm, n_vars, lookback, horizon)
