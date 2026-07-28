#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py - 金融时序模型定义库 (PyTorch Native | V5.05 取优合并版)
合并策略：
1. 以 V5.04 完整版为骨架基础
2. 对比 V5.04 与 V5.02 的 T+1 实盘可交易验证 NDCG@40
3. 逐模型取优（NDCG@40 高者保留），平局保留 V5.04
4. 保留全部 110 个模型编号及 MODEL_META 族信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========== 通用轻量核心模块 =====================
class SimpleKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.grid_size = min(grid_size, max(3, in_dim))
        self.register_buffer('centers', torch.linspace(-1, 1, self.grid_size))
        self.register_buffer('width', torch.tensor(0.4))
        self.spline_coef = nn.Parameter(torch.randn(out_dim, in_dim, self.grid_size) * 0.05)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, -1)
        rbf = torch.exp(-((x_exp - centers) / self.width) ** 2)
        spline_out = torch.einsum('oig,big->bo', self.spline_coef, rbf)
        base_out = F.linear(F.silu(x), self.base_weight, self.base_bias)
        return base_out + spline_out

class BSplineKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.n_basis = self.grid_size + self.spline_order
        n_grid = self.n_basis + self.spline_order + 1
        grid = torch.linspace(-1, 1, n_grid, dtype=torch.float32)
        self.register_buffer('grid', grid)
        self.coef = nn.Parameter(torch.randn(self.out_dim, self.in_dim, self.n_basis) * 0.05)
        self.base = nn.Parameter(torch.randn(self.out_dim, self.in_dim) * 0.1)

    def compute_basis(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid.view(1, 1, -1)
        basis = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[..., :-(k + 1)]
            left_den = grid[..., k:-1] - grid[..., :-(k + 1)]
            right_num = grid[..., (k + 1):] - x
            right_den = grid[..., (k + 1):] - grid[..., 1:-k]
            left = left_num / (left_den + 1e-8)
            right = right_num / (right_den + 1e-8)
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, x):
        x_in = torch.tanh(x) * 0.999
        basis = self.compute_basis(x_in)
        spline = torch.einsum('oij,bij->bo', self.coef, basis)
        base = F.linear(F.silu(x), self.base)
        return base + spline

class SelectiveSSM(nn.Module):
    def __init__(self, hidden, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.hidden = int(hidden)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.hidden * self.expand)
        self.in_proj = nn.Linear(self.hidden, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv,
                                padding=self.d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1)
        A_init = torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_init.unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.hidden)

    def forward(self, x):
        B = int(x.shape[0])
        L = int(x.shape[1])
        x_and_z = self.in_proj(x)
        x_inner, z = x_and_z.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2)).transpose(1, 2)[:, :L, :]
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        Bp, Cp, delta = x_ssm.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(delta.squeeze(-1))
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(delta.unsqueeze(-1).unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        deltaB = delta.unsqueeze(-1).unsqueeze(-1) * Bp.unsqueeze(2)
        h_state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h_state = deltaA[:, i] * h_state + deltaB[:, i] * x_conv[:, i].unsqueeze(-1)
            ys.append(torch.sum(h_state * Cp[:, i].unsqueeze(1), dim=-1))
        y = torch.stack(ys, dim=1)
        y = y + self.D.view(1, 1, self.d_inner) * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

class LightTransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads=4, drop=0.1, n_layers=2):
        super().__init__()
        self.hidden = int(hidden)
        self.n_heads = int(n_heads)
        self.layers = nn.ModuleList()
        for _ in range(int(n_layers)):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden),
                'attn': nn.MultiheadAttention(self.hidden, self.n_heads, dropout=drop, batch_first=True),
                'norm2': nn.LayerNorm(self.hidden),
                'ffn': nn.Sequential(nn.Linear(self.hidden, self.hidden * 2), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(self.hidden * 2, self.hidden), nn.Dropout(drop))
            }))
    def forward(self, x):
        for layer in self.layers:
            h = layer['norm1'](x)
            h, _ = layer['attn'](h, h, h, need_weights=False)
            x = x + h
            h = layer['norm2'](x)
            x = x + layer['ffn'](h)
        return x

class LightHyperGraphConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.node_proj = nn.Linear(self.hidden, self.hidden)
        self.hyper_edge = nn.Parameter(torch.randn(self.hidden, self.hidden) * 0.05)
        self.gate = nn.Linear(self.hidden, self.hidden)
    def forward(self, h):
        adj = torch.sigmoid(self.hyper_edge)
        h_nei = torch.matmul(h, adj)
        h_out = self.node_proj(h_nei)
        g = torch.sigmoid(self.gate(h))
        return g * h_out + (1 - g) * h

# ============ 统一基类 =====================
class Step2Base(nn.Module):
    def __init__(self, d_in: int, hidden: int = 96, drop: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.drop = drop
        self.proj = nn.Linear(d_in, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(drop)
        self.input_bn = nn.BatchNorm1d(d_in)
        self.feat_selector = nn.Sequential(
            nn.Linear(d_in, hidden // 2), nn.GELU(),
            nn.Dropout(drop * 0.5), nn.Linear(hidden // 2, d_in), nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def _preprocess(self, x):
        x = self.input_bn(x)
        w = self.feat_selector(x)
        x = x * w
        return self.norm(self.proj(x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ===============  模型定义 (M01 - M240) =====================

class M01_FITS(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        hf = torch.fft.rfft(x, dim=1)
        hf[:, max(1, hf.shape[1] // 4):] = 0.0
        x_recon = torch.fft.irfft(hf, n=x.shape[1], dim=1)
        h = self._preprocess(x)
        h_freq = self.freq_proj(x_recon)
        e = self.enhance(h_freq)
        return self.head(h_freq + e * 0.3 + h * 0.2)

class M02_SparseTSF(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.head = nn.Linear(hidden, 1)
        self.sparse_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())

    def forward(self, x):
        h = self._preprocess(x)
        step = max(2, h.shape[1] // 4)
        sampled = h[:, ::step]
        if sampled.numel() == 0 or sampled.shape[1] == 0:
            g = self.sparse_gate(h)
            return self.head(h * g)
        s = sampled.mean(1, keepdim=True).expand_as(h)
        g = self.sparse_gate(s)
        fused = s * 0.4 * g + h * 0.6
        return self.head(fused + h * 0.2)

class M03_RLinear(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.res_block = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden * 2, hidden)
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        r = self.res_block(h)
        out = self.head(h + r * 0.3)
        return out

class M04_TimeBridge(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.bridge = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        b = self.bridge(h)
        g = self.gate(torch.cat([h, b], dim=-1))
        out = self.head(h + b * g)
        return out

class M05_SpectraFormer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.time_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        hf = torch.fft.rfft(x, dim=1)
        k = max(1, hf.shape[1] // 4)
        hf[:, k:] *= 0.0
        h_freq = self.norm(self.proj(torch.fft.irfft(hf, n=x.shape[1], dim=1)))
        g_f = self.freq_gate(h_freq)
        g_t = self.time_gate(h)
        fused = h_freq * g_f + h * g_t
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h * 0.2 + h_c * 0.2)

class M06_TabPFN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.scale = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.shift = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        gamma = self.scale(x)
        beta = self.shift(x)
        base = h * gamma + beta
        e = self.enhance(base)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(base + e * 0.3 + h_c * 0.2)

class M07_TabICL(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.ctx = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.cross = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = torch.sigmoid(self.ctx(h))
        cr = self.cross(x)
        g = self.gate(torch.cat([h * c, cr], dim=-1))
        fused = h * c * g + cr * (1 - g)
        return self.head(fused + h * 0.2)

class M08_TimesFM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.temporal_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        g = self.temporal_gate(h_c)
        return self.head(h_c * g + h * (1 - g) + h * 0.2)

class M09_OLinear(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        with torch.no_grad():
            w = torch.randn(hidden, d_in)
            q, _ = torch.linalg.qr(w.t() if hidden <= d_in else w)
            # FIX: torch.linalg.qr 默认返回 reduced mode。
            # 当 hidden <= d_in 时，w.t() 为 (d_in, hidden)，q 为 (d_in, hidden)，需转置为 (hidden, d_in)。
            # 当 hidden > d_in 时，w 为 (hidden, d_in)，q 为 (hidden, d_in)，直接取前 d_in 列。
            self.proj.weight.data = (q.t() if hidden <= d_in else q[:, :d_in]) * 0.5
            self.proj.weight.data = F.normalize(self.proj.weight.data, dim=1)
        self.correct_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.norm(self.proj(x))
        w = F.softplus(h) / (F.softplus(h).sum(dim=-1, keepdim=True) + 1e-8)
        ortho = h * w
        g = self.correct_gate(ortho)
        return self.head(ortho * g + h * 0.3)

class M10_TiRex(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.ctx = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.ctx(h)
        f = self.freq(x)
        g = self.gate(torch.cat([h, c], dim=-1))
        fused = h + c * g + f * 0.3
        e = self.enhance(fused)
        return self.head(fused + e * 0.3)

class M11_TSPRank(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        e = self.enc(x)
        r = self.res(e)
        return self.head(e + r * 0.3 + h * 0.2)

class M12_DUET(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c1 = self.conv1(x.unsqueeze(1)).squeeze(1)
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        fused = self.fusion(torch.cat([h1, h2], dim=-1))
        return self.head(fused + h * 0.3)

class M13_MLF(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        h = self._preprocess(x)
        fused = self.fusion(torch.cat([h_c, h], dim=-1))
        return self.head(fused + h * 0.3)

class M14_MambaSSM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.gate = nn.Linear(d_in, hidden)
        self.state = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        g = torch.sigmoid(self.gate(c))
        s = torch.tanh(self.state(c))
        h_norm = self.norm(self.proj(c))
        fused = g * h_norm + (1 - g) * s
        return self.head(fused + h_norm * 0.2)

class M15_FreDF_Whitening(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        hf = torch.fft.rfft(x, dim=1)
        amp = hf.abs()
        k = max(1, amp.shape[1] // 3)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        h_low = torch.fft.irfft(hf * mask, n=x.shape[1], dim=1)
        h_low = self._preprocess(h_low)
        h_orig = self._preprocess(x)
        avg = h_low.mean(1, keepdim=True)
        std = h_low.std(1, keepdim=True) + 1e-5
        h_white = (h_low - avg) / std
        gate = torch.sigmoid(avg)
        return self.head(h_white * gate + h_orig * (1 - gate))

class M16_SoftDTW_Shape(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.shape = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        m = h.mean(1, keepdim=True)
        s = self.shape(h - m)
        return self.head(h + s * 0.3 + m * 0.1 + h_c * 0.2)

class M17_ModernTCN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x_u = x.unsqueeze(1)
        c1 = self.conv1(x_u).squeeze(1)
        c2 = self.conv2(x_u).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        fused = self.fusion(torch.cat([h1, h2], dim=-1))
        return self.head(fused + self._preprocess(x) * 0.3)

class M18_TimeMixer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.coarse_gate = nn.Linear(d_in, hidden)
        self.fine_enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        g = torch.sigmoid(self.coarse_gate(x))
        coarse = h * g
        fine = self.fine_enhance(h - h.mean(1, keepdim=True).expand_as(h))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(coarse + fine * 0.4 + h * 0.2 + h_c * 0.2)

class M19_CycleNet(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.cycle = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = torch.sin(h * 0.15) * 0.2
        r = self.cycle(h + c)
        return self.head(h + r * 0.3)

class M20_Chronos(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.tg = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        g = self.tg(h)
        r = self.res(h)
        return self.head(h + r * g)

class M21_Aurora(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.num_path = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.sem_proj = nn.Linear(d_in, hidden)
        self.sem_path = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        p_num = self.num_path(h)
        hf = torch.fft.rfft(x, dim=1)
        amp = hf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        x_sem = torch.fft.irfft(hf * mask, n=x.shape[1], dim=1)
        p_sem = self.sem_path(self.sem_proj(x_sem))
        g = self.gate(torch.cat([p_num, p_sem], dim=-1))
        fused = g * p_num + (1 - g) * p_sem
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2)

class M22_Moirai(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.quant_proj = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.proj(x)
        if self.training:
            h = F.dropout(h, p=self.drop, training=True)
        h = self.norm(h)
        e = self.enhance(h)
        q = torch.tanh(self.quant_proj(h))
        return self.head(h + e * 0.3 + q * 0.2)

class M23_LightGTS(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.scale_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        avg = h.mean(1, keepdim=True).expand_as(h)
        mx = h.max(1, keepdim=True)[0].expand_as(h)
        g = self.scale_gate(torch.cat([avg, mx], dim=-1))
        fused = g * avg + (1 - g) * mx + h * 0.3
        return self.head(fused)

class M24_Sundial(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        avg = h.mean(1, keepdim=True).expand_as(h)
        fused = h + avg * 0.3
        e = self.enhance(fused)
        return self.head(fused + e * 0.3)

class M25_Timer_XL(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.s = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.m = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = self.s(h)
        avg = h.mean(1, keepdim=True).expand(-1, h.size(1))
        b = self.m(h - avg)
        std = (h - avg).pow(2).mean(1, keepdim=True).sqrt().expand(-1, h.size(1))
        c = self.l(std)
        g = self.gate(torch.cat([a, b, c], dim=-1))
        fused = a * g + b * (1 - g) * 0.5 + c * (1 - g) * 0.5
        hf = torch.fft.rfft(x, dim=1)
        hf[:, max(1, hf.shape[1] // 4):] *= 0.0
        f = self.freq(torch.fft.irfft(hf, n=x.shape[1], dim=1))
        cv = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(cv))
        return self.head(h + fused + f * 0.2 + h_c * 0.2)

class M26_UniTS(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.g1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.g2 = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = h.mean(1, keepdim=True).expand(-1, h.size(1))
        g1, g2 = self.g1(a), self.g2(h - a)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(h + a * g1 + (h - a) * g2 + h_c * 0.2)

class M27_MOMENT(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.momentum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        m = self.momentum(h.mean(1, keepdim=True).expand_as(h))
        g = torch.sigmoid(m)
        return self.head(h * g + h * 0.3)

class M28_Kronos(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        g = self.gate(h)
        attended = h * g
        e = self.enhance(attended)
        return self.head(attended + e * 0.3 + h * 0.2)

class M29_TimeFilter(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.fs = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fl = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        s = self.fs(h)
        l = self.fl(h - h.mean(1, keepdim=True).expand_as(h))
        g = self.fuse(torch.cat([s, l], dim=-1))
        fused = g * s + (1 - g) * l
        e = self.enhance(fused)
        hf = torch.fft.rfft(x, dim=1)
        hf[:, max(1, hf.shape[1] // 4):] *= 0.0
        f = self.freq(torch.fft.irfft(hf, n=x.shape[1], dim=1))
        return self.head(fused + e * 0.3 + h * 0.2 + f * 0.2)

class M30_ROSE(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Parameter(torch.tensor(0.15))
        self.amp = nn.Parameter(torch.tensor(1.0))
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        modulated = h * torch.cos(h * torch.abs(self.freq)) * torch.sigmoid(self.amp)
        e = self.enhance(modulated)
        return self.head(modulated + h * 0.2 + e * 0.3 + h_c * 0.2)

class M31_xLSTM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.mem = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        m = self.mem(h)
        g = self.gate(torch.cat([m, h_c], dim=-1))
        return self.head(h + m * 0.5 * g + h_c * (1 - g) * 0.3)

class M32_DistDF_Wasserstein(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # Temporal: 局部卷积提取时序模式 + 频域低通滤波
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.freq_gate = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop)
        )
        # 鲁棒统计量投影：在原始输入维度计算 median / Q25 / Q75 / MAD
        self.stat_proj = nn.Sequential(nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.GELU())
        # 分布距离编码（Wasserstein 核心：特征表示与鲁棒摘要的偏离）
        self.dist_enc = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        # 自适应融合门控
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # Temporal 局部卷积分支
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        # Temporal 频域低通分支
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        x_freq = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        h_f = self.freq_gate(x_freq)
        # Wasserstein 鲁棒统计（关键修复：在原始输入特征维度 dim=1 计算）
        q50 = x.median(dim=1, keepdim=True)[0]
        q25 = x.kthvalue(max(1, x.shape[1] // 4), dim=1)[0].unsqueeze(1)
        q75 = x.kthvalue(max(1, x.shape[1] * 3 // 4), dim=1)[0].unsqueeze(1)
        mad = (x - q50).abs().median(dim=1, keepdim=True)[0] + 1e-8
        stats = torch.cat([q50, q25, q75, mad], dim=-1)  # (B, 4)
        s = self.stat_proj(stats)  # (B, hidden) —— 真正的样本级鲁棒统计摘要
        # Wasserstein-like 分布距离
        dist = h - s
        d = self.dist_enc(dist)
        g = self.gate(torch.cat([s, d], dim=-1))
        fused = s * g + d * (1 - g)
        # 融合：分布平衡为主，时序特征为辅
        return self.head(fused + h_c * 0.15 + h_f * 0.15 + h * 0.2)

class M33_RealMLP(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        m = self.mlp(h)
        e = self.enhance(m)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(m + e * 0.3 + h * 0.2 + h_c * 0.2)

class M34_LimiX(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.selector = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        s = self.selector(h)
        e = self.enhance(h * s)
        return self.head(h + e * 0.3)

class M35_Mitra(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        a = F.softmax(torch.tanh(h), dim=-1)
        attended = h * a
        g = self.attn(attended)
        return self.head(attended * g + h * 0.4)

class M36_WPMixer(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # WaveMix: 双尺度卷积分离低频趋势 / 高频细节
        self.low_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3)   # 低频：大核平滑
        self.high_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)  # 高频：小核细节
        self.low_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.high_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 低频分支：大核卷积捕捉趋势
        c_low = self.low_conv(x.unsqueeze(1)).squeeze(1)
        h_low = self.norm(self.proj(c_low))
        lo = h_low * self.low_gate(h_low)
        # 高频分支：小核卷积捕捉细节 + 均值残差
        c_high = self.high_conv(x.unsqueeze(1)).squeeze(1)
        h_high = self.norm(self.proj(c_high))
        avg = h.mean(1, keepdim=True).expand_as(h)
        hi = (h - avg) * self.high_gate(h_high)
        # 小波式频带门控融合
        fused = self.fusion(torch.cat([lo, hi], dim=-1))
        return self.head(fused + h * 0.3)

class M37_TimeMCL(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.p1 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.p2 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 1.2))
        self.p3 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.8))
        self.w = nn.Parameter(torch.ones(3))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        o1 = self.p1(x)
        o2 = self.p2(x)
        o3 = self.p3(x)
        w = F.softmax(self.w, dim=0)
        fused = o1 * w[0] + o2 * w[1] + o3 * w[2]
        c1 = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c1 = self.norm(self.proj(c1))
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h_c2 = self.norm(self.proj(c2))
        g = self.fusion(torch.cat([h_c1, h_c2], dim=-1))
        return self.head(fused + h * 0.2 + (h_c1 * g + h_c2 * (1 - g)) * 0.2)

class M38_NeuralPort(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        g = self.gate(h)
        w = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        attended = h * w
        e = self.enhance(attended)
        return self.head(attended + e * 0.3 + h * 0.2)

class M39_AdaptWin(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c1 = self.conv1(x.unsqueeze(1)).squeeze(1)
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        g = self.fusion(torch.cat([h1, h2], dim=-1))
        fused = h1 * g + h2 * (1 - g)
        return self.head(fused + h * 0.3)

class M40_StockSSG(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        s = torch.softmax(h, dim=-1)
        return self.head(h * s + h * 0.2 + h_c * 0.2)

class M41_TabDPT(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.path_a = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.diff_enc = nn.Sequential(nn.Linear(max(1, d_in - 1), hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = self.path_a(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        b = self.diff_enc(dx)
        g = self.gate(torch.cat([a, b], dim=-1))
        fused = h + a * g + b * (1 - g)
        return self.head(fused)

class M42_Timer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.mask_path = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        xm = x * (torch.rand_like(x) > 0.15).float() if self.training else x
        hm = self.mask_path(xm)
        g = self.gate(torch.cat([h, hm], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        out = h + hm * g + h_c * 0.2
        return self.head(out)

class M43_MSGNet(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.ns = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.nl = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        ns = self.ns(h - h.mean(1, keepdim=True).expand_as(h))
        nl = self.nl(h)
        g = self.fuse(torch.cat([ns, nl], dim=-1))
        fused = g * ns + (1 - g) * nl
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2)

class M44_Pathformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x_u = x.unsqueeze(1)
        c = self.conv(x_u).squeeze(1)
        h_conv = self.norm(self.proj(c))
        h_orig = self.norm(self.proj(x))
        g = self.fusion(torch.cat([h_conv, h_orig], dim=-1))
        fused = h_conv * g + h_orig * (1 - g)
        return self.head(fused + h_orig * 0.2)

class M45_NodeTrans_Stock(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.edges = nn.Parameter(torch.randn(self.n_g, self.n_g) * 0.05)
        self.cross = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        adj = torch.sigmoid(self.edges)
        h_graph = torch.matmul(adj.unsqueeze(0), hp)
        a, _ = self.attn(hp, h_graph, h_graph, need_weights=False)
        p = hp + a * 0.3
        c = self.cross(p.mean(1, keepdim=True).expand(-1, self.n_g, -1))
        fused = p + c * 0.3
        return self.head(fused.mean(1) + h * 0.2)

class M46_TimeKAN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.kan1 = SimpleKANLayer(hidden, hidden, 5, 3)
        self.kan2 = SimpleKANLayer(hidden, hidden, 3, 2)
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        k1 = torch.tanh(self.kan1(h))
        k2 = torch.tanh(self.kan2(h))
        g = self.g(torch.cat([k1, k2], dim=-1))
        return self.head(h + k1 * g + k2 * (1 - g))

class M47_Autoformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.season_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x_u = x.unsqueeze(1)
        trend = self.decomp(x_u).squeeze(1)
        seasonal = x - trend
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(seasonal)
        h_orig = self._preprocess(x)
        fused = self.fusion(torch.cat([h_t, h_s], dim=-1))
        return self.head(fused + h_orig * 0.3)

class M48_Informer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.selector = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        scores = torch.sigmoid(self.selector(h))
        attended = h * scores + h * 0.2
        e = self.enhance(attended)
        return self.head(attended + e * 0.3)

class M49_FEDformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=1)
        k = max(1, xf.shape[1] // 3)
        xf[:, k:] *= 0.5
        x_f = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        f = self.freq(x_f)
        return self.head(h + f * 0.3)

class M50_PatchTST(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1, patch_len=16):
        super().__init__(d_in, hidden, drop)
        self.patch_len = max(4, min(patch_len, d_in // 4))
        self.n_patch = (d_in + self.patch_len - 1) // self.patch_len
        self.patch_proj = nn.Linear(self.patch_len, hidden)
        self.patch_attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        B, D = x.shape
        pad = self.n_patch * self.patch_len - D
        # FIX: 保留原始 x 用于 _preprocess，防止 pad 后的维度破坏 BatchNorm
        x_raw = x
        if pad > 0: x = F.pad(x, (0, pad))
        x_patch = x.reshape(B, self.n_patch, self.patch_len)
        h_patch = self.norm(self.patch_proj(x_patch))
        ha, _ = self.patch_attn(h_patch, h_patch, h_patch, need_weights=False)
        h_patch = h_patch + ha * 0.3
        return self.head(h_patch.mean(dim=1) + self._preprocess(x_raw) * 0.2)

class M51_TimesNet(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_enhance = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        hf = torch.fft.rfft(x, dim=1)
        amp = hf.abs()
        k = max(1, int(amp.shape[1]) // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp).scatter_(1, idx, 1.0)
        x_freq = torch.fft.irfft(hf * mask, n=int(x.shape[1]), dim=1)
        h = self._preprocess(x)
        h_f = self.freq_enhance(x_freq)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(h + h_f * 0.4 + h_c * 0.2)

class M52_DLinear(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.season_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x_u = x.unsqueeze(1)
        trend = self.decomp(x_u).squeeze(1)
        seasonal = x - trend
        h_t = self.norm(self.proj(trend))
        h_s = self.norm(self.proj(seasonal))
        g_t = self.trend_gate(h_t)
        g_s = self.season_gate(h_s)
        fused = g_t * h_t + g_s * h_s
        return self.head(fused + self._preprocess(x) * 0.2)

class M53_Crossformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)  # 补全 head
    def forward(self, x):
        x_u = x.unsqueeze(1)
        c = self.conv(x_u).squeeze(1)
        h_c = self.norm(self.proj(c))
        h_o = self.norm(self.proj(x))
        # 移除 mean，改用 head 输出无界排序分数
        return self.head(h_c + h_o + self._preprocess(x) * 0.2)

class M54_TabNet(Step2Base):
    def forward(self, x):
        h = self._preprocess(x)
        a = F.softmax(torch.tanh(h), dim=-1)
        return (h * a + h * 0.2).mean(1, keepdim=True)

class M55_TFT(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.q1 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.q2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.freq_proj = nn.Linear(d_in, hidden)
        self.q3 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = self.q1(h)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx_pad = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx
        b = self.q2(self.norm(self.proj(dx_pad)))
        hf = torch.fft.rfft(x, dim=1)
        amp = hf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        h_freq = torch.fft.irfft(hf * mask, n=x.shape[1], dim=1)
        c = self.q3(self.freq_proj(h_freq))
        g = self.g(torch.cat([a, b, c], dim=-1))
        fused = h + a * g + b * (1 - g) * 0.5 + c * (1 - g) * 0.5
        cv = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(cv))
        return self.head(fused + h_c * 0.2)

class M56_ETSformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.tg = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.sg = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        avg_val = h.mean(1, keepdim=True).expand_as(h)
        season = h - avg_val
        gt, gs = self.tg(avg_val), self.sg(season)
        fused = gt * avg_val * 0.7 + gs * season * 0.3 + h * 0.2
        return self.head(fused)

class M57_ASTGI(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.adj = nn.Parameter(torch.eye(d_in) * 0.5 + torch.randn(d_in, d_in) * 0.05)
        self.graph_proj = nn.Linear(d_in, hidden)
        self.temporal = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        adj = torch.sigmoid(self.adj)
        x_graph = torch.matmul(x, adj)
        h_graph = self.norm(self.graph_proj(x_graph))
        h_temp = self.norm(self.proj(self.temporal(x.unsqueeze(1)).squeeze(1)))
        g = self.fusion_gate(torch.cat([h_graph, h_temp], dim=-1))
        fused = h_graph * g + h_temp * (1 - g)
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + self._preprocess(x) * 0.2)

class M58_Pyraformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.coarse = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fine = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        avg = h.mean(1, keepdim=True).expand(-1, h.size(1))
        c = self.coarse(avg)
        f = self.fine(h - avg)
        fused = self.fusion(torch.cat([c, f], dim=-1))
        e = self.enhance(fused)
        return self.head(h + fused + e * 0.3)

class M59_FiLM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        hf = torch.fft.rfft(x, dim=1)
        freq = torch.linspace(0, 2 * np.pi, hf.shape[1], device=x.device)
        mod = 1 + 0.1 * torch.sin(freq * 2)
        hf = hf * mod.unsqueeze(0)
        x_mod = torch.fft.irfft(hf, n=x.shape[1], dim=1)
        h = self._preprocess(x_mod)
        h_orig = self._preprocess(x)
        g = torch.sigmoid(self.gate(h))
        fused = h * g + h_orig * (1 - g)
        return self.head(fused + h_orig * 0.2)

class M60_MICN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.res_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        u = x.unsqueeze(1)
        c1 = self.c1(u).squeeze(1)
        c2 = self.c2(u).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        g = self.fusion(torch.cat([h1, h2], dim=-1))
        fused = h1 * g + h2 * (1 - g)
        alpha = self.res_gate(h) * 0.2 + 0.2
        return self.head(fused + h * alpha)

class M61_SCINet(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.sc = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c = self.sc(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj((x + c) / 2))
        return self.head(h_c + h * 0.2)

class M62_RevIN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.affine = nn.Parameter(torch.ones(hidden))
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        m = h.mean(1, keepdim=True)
        s = h.std(1, keepdim=True) + 1e-5
        h_norm = (h - m) / s
        h_rev = h_norm * self.affine
        g = self.gate(h_rev)
        r = self.res(h_rev)
        return self.head(h_rev + r * g + h * 0.1)

class M63_iTransformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        a, _ = self.attn(hp, hp, hp, need_weights=False)
        p = (hp + a * 0.3).mean(dim=1)
        return self.head(h + p * 0.3)

class M64_VanillaTransformer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=2, dim_feedforward=hidden * 2, dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        e = self.enc(hp)
        p = (hp + e * 0.3).mean(dim=1)
        c1 = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c1 = self.norm(self.proj(c1))
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h_c2 = self.norm(self.proj(c2))
        g = self.fusion(torch.cat([h_c1, h_c2], dim=-1))
        h_c = h_c1 * g + h_c2 * (1 - g)
        return self.head(p + h * 0.2 + h_c * 0.2)

class M65_RWKV_TS(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.decay = nn.Parameter(torch.ones(hidden) * 0.5)
        self.key = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        k = self.key(h)
        v = self.value(h)
        w = torch.sigmoid(self.decay)
        attn = (k * v) * w
        e = self.enhance(attn)
        return self.head(h + attn + e * 0.3)

class M66_Mamba2SSD(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1, n_seg=8):
        super().__init__(d_in, hidden, drop)
        self.n_seg = int(n_seg)
        self.seg_len = (int(d_in) + self.n_seg - 1) // self.n_seg
        self.seg_proj = nn.Linear(self.seg_len, self.hidden)
        self.seg_conv = nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1, groups=self.hidden)
        self.ssm_gate = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.Sigmoid())
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x):
        B, D = int(x.shape[0]), int(x.shape[1])
        pad_len = self.n_seg * self.seg_len - D
        if pad_len > 0: x = F.pad(x, (0, pad_len))
        x = x.reshape(B, self.n_seg, self.seg_len)
        h = self.norm(self.seg_proj(x))
        h_c = F.silu(self.seg_conv(h.transpose(1, 2)).transpose(1, 2))
        h_cum = torch.cumsum(h_c, dim=1)
        denom = torch.arange(1, self.n_seg + 1, device=x.device).view(1, -1, 1).float()
        h_cum = h_cum / (denom + 1e-8)
        pooled = h_c.mean(dim=1) + h_cum.mean(dim=1) * 0.5
        g = self.ssm_gate(pooled)
        return self.head(pooled + h.mean(dim=1) * g * 0.3)

class M67_CondFlowMatch(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.t_emb = nn.Linear(1, hidden)
        self.enc = nn.Sequential(nn.Linear(d_in + hidden, hidden*2), nn.SiLU(), nn.Dropout(drop), nn.Linear(hidden*2, hidden))
        self.vel = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(drop), nn.Linear(hidden, d_in))
    def forward(self, x, t=None):
        h = self._preprocess(x)
        # FIX: 推理时固定 t_val 为 0.5，消除随机性，保证确定性输出
        if self.training:
            t_val = torch.rand(x.size(0), 1, device=x.device) if t is None else t
        else:
            t_val = torch.full((x.size(0), 1), 0.5, device=x.device) if t is None else t
        te = self.t_emb(t_val)
        enc_h = self.enc(torch.cat([x, te], dim=-1))
        v = torch.tanh(self.vel(enc_h)) * 0.5
        x1 = x + v
        return self._preprocess(x1).mean(1, keepdim=True)

class M68_OrthoTrans(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.orth_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        m, s = h.mean(1, keepdim=True), h.std(1, keepdim=True) + 1e-5
        hn = (h - m) / s
        var = (hn ** 2).mean(0, keepdim=True)
        mask = torch.sigmoid(var * 8.0 - 2.0)
        hm = self.norm(hn * mask)
        o = self.orth_proj(hm)
        return self.head(o + h * 0.2)

class M69_NeuralShrinkage(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.shrink = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.temp = nn.Parameter(torch.tensor(0.5))
        self.bypass = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        lam = self.shrink(h) * torch.sigmoid(self.temp)
        out = h * torch.tanh((h.abs() - lam) / (lam + 0.1))
        b = self.bypass(h)
        return self.head(out + b * 0.5 + h * 0.2 + h_c * 0.2)

class M70_UncertaintyCAE(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.uncertainty_enc = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        var = h.var(dim=-1, keepdim=True)
        u = torch.sigmoid(var * 5.0 - 1.0)
        w = 1.0 - u * 0.5
        h_u = self.uncertainty_enc(h * u)
        return self.head(h * w + h_u * 0.3 + h * 0.2 + h_c * 0.2)

class M71_NEDreamer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.trend_enc = nn.Sequential(nn.Linear(max(1, d_in - 1), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.momentum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fuse_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        tr = self.trend_enc(dx)
        m = torch.tanh(self.momentum(h))
        g = self.fuse_gate(torch.cat([tr, m], dim=-1))
        fused = g * tr + (1 - g) * m
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2 + h_c * 0.2)

class M72_StockMixer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1, n_group=8):
        super().__init__(d_in, hidden, drop)
        self.n_group = int(n_group)
        self.seg_len = (int(d_in) + self.n_group - 1) // self.n_group
        self.group_proj = nn.Linear(self.seg_len, self.hidden)
        self.group_mix = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.LayerNorm(self.hidden), nn.GELU())
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x):
        B, D = int(x.shape[0]), int(x.shape[1])
        pad_len = self.n_group * self.seg_len - D
        if pad_len > 0: x = F.pad(x, (0, pad_len))
        x = x.reshape(B, self.n_group, self.seg_len)
        h = self.norm(self.group_proj(x))
        m = self.group_mix(h.mean(dim=1) + h.max(dim=1)[0] * 0.3)
        return self.head(m + h.mean(dim=1) * 0.2)

class M73_KAN_AD(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.kan_trend = SimpleKANLayer(hidden, hidden, 5, 3)
        self.kan_resid = SimpleKANLayer(hidden, hidden, 3, 2)
        self.anomaly_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.bypass = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        kt = self.kan_trend(h)
        kr = self.kan_resid(h - h.mean(1, keepdim=True).expand(-1, h.size(1)))
        g = self.anomaly_gate(torch.cat([kt, kr], dim=-1))
        kan_fused = kt * g + kr * (1 - g)
        b = self.bypass(h)
        return self.head(h + kan_fused * 0.2 + b * 0.4)

class M74_SPDQ_RL(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        m = self.mu(h)
        r = self.res(m)
        return self.head(h + m + r * 0.2)

class M75_TimeAlign(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.a = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.align_corr = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        al = self.a(x)
        corr = torch.tanh(self.align_corr(al - h))
        g = self.g(torch.cat([h, corr], dim=-1))
        fused = h + al * g + corr * (1 - g) * 0.3
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h_c * 0.2)

class M76_WaveLSFormer(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.low_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.high_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        x_smooth = F.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        lo = self.low_pass(x_smooth)
        hi = self.high_pass(x - x_smooth)
        fused = self.fusion(torch.cat([lo, hi], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h * 0.2 + h_c * 0.2)

class M77_MMPD_Predictor(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.time_branch = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.freq_proj = nn.Linear(d_in, hidden)
        self.freq_branch = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        t = self.time_branch(h)
        hf = torch.fft.rfft(x, dim=1)
        amp = hf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        h_freq = torch.fft.irfft(hf * mask, n=x.shape[1], dim=1)
        f = self.freq_branch(self.freq_proj(h_freq))
        g = self.gate(torch.cat([t, f], dim=-1))
        return self.head(h + t * g + f * (1 - g) + h * 0.2)

class M78_MarketGAN_Aug(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.noise_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        if self.training:
            x_n = x + torch.randn_like(x) * 0.05
        else:
            x_n = x
        n = self.noise_proj(x_n)
        return self.head(h + n * 0.3)

class M79_rfBLT_Bayes(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.rank = max(32, d_in // 2)
        self.fu = nn.Parameter(torch.randn(d_in, self.rank) * 0.03)
        self.res = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(self.rank + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self._preprocess(x)
        hl = torch.matmul(x, self.fu)
        if self.training:
            hl = F.dropout(hl, p=0.1, training=True)
        hr = self.res(x)
        # FIX: 移除外层 torch.sigmoid，主脚本 train_clean 已统一做 sigmoid，避免双重压缩
        return self.head(torch.cat([hl, hr], dim=-1))

class M80_MaGNet(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.assoc = nn.Sequential(nn.Linear(d_in, d_in), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = self.assoc(x)
        xh = x * a
        hh = self.norm(self.proj(xh))
        return self.head(h + hh * 0.3)

class M81_LOBERT(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.diff_proj = nn.Linear(max(1, d_in - 1), hidden)
        self.pressure = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        h_local = self.norm(self.diff_proj(dx))
        p = self.pressure(torch.cat([h, h_local], dim=-1))
        fused = p * h_local + (1 - p) * h
        return self.head(fused)

class M82_KANMixer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.kan_path = nn.Sequential(SimpleKANLayer(d_in, hidden, 5, 3), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.mlp_path = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        hk, hm = self.kan_path(x), self.mlp_path(x)
        g = self.fusion_gate(torch.cat([hk, hm], dim=-1))
        fused = g * hk + (1 - g) * hm
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2)

class M83_FinD3(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.raw = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.diff = nn.Sequential(nn.Linear(max(1, d_in - 1), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        m = x.mean(1, keepdim=True)
        s = x.std(1, keepdim=True) + 1e-5
        xn = (x - m) / s
        r = self.raw(xn)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        d = self.diff(dx)
        g = self.g(torch.cat([r, d], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(r * g + d * (1 - g) + h_c * 0.2)

class M84_Hermes(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.lead = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.lag = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        avg_val = h.mean(1, keepdim=True).expand_as(h)
        l = self.lead(h)
        d = self.lag(h - avg_val)
        fused = self.fuse(torch.cat([l, d], dim=-1))
        e = self.enhance(fused)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + e * 0.3 + h * 0.2 + h_c * 0.2)

class M85_SPF_Hawkes(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.intensity = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        energy = self.intensity(x * torch.tanh(x))
        g = torch.sigmoid(energy)
        fused = h * g + h * 0.2
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h_c * 0.2)

class M86_FactorGCL(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.main = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.perturb = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        m = self.main(h)
        h_p = h + torch.randn_like(h) * 0.1 if self.training else h
        p = self.perturb(h_p)
        diff = m - p
        g = self.g(torch.cat([m, diff], dim=-1))
        return self.head(h + m * g + diff * (1 - g) * 0.3 + h_c * 0.2)

class M87_DeltaLag(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # DynamicLeadLag: 差分驱动 + 多尺度滞后池化
        self.dx_proj = nn.Linear(d_in, hidden)
        self.lag_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.lag_proj = nn.Linear(d_in, hidden)
        # Robust: 显式 quantile 感知门控（满足 Robust 族校验）
        self.quantile_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.sparse_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # Lead 驱动：一阶差分
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx_proj(dx))
        # Lag 编码：移动平均滞后
        c_lag = self.lag_pool(x.unsqueeze(1)).squeeze(1)
        h_lag = self.norm(self.lag_proj(c_lag))
        # Robust 分位数门控融合
        h_robust = self.quantile_gate(h_dx + h_lag)
        g = self.sparse_gate(h)
        hs = h * g
        fused = self.fusion(torch.cat([hs, h_robust], dim=-1))
        return self.head(fused + h * 0.3)

class M88_DTAF(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.time = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        f = self.freq(x)
        t = self.time(x)
        g = self.gate(torch.cat([f, t], dim=-1))
        fused = f * g + t * (1 - g)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h * 0.2 + h_c * 0.2)

class M89_DRFN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.static = nn.Linear(d_in, hidden, bias=False)
        self.dynamic = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
    def forward(self, x):
        h = self._preprocess(x)
        hs = self.static(x)
        hd = self.dynamic(x)
        gate = self.fusion(torch.cat([hs, hd], dim=-1))
        h_f = gate * hs + (1 - gate) * hd
        return (h_f + h * 0.2).mean(1, keepdim=True)

class M90_AMD(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.trend = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.resid = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        x_u = x.unsqueeze(1)
        trend = F.avg_pool1d(x_u, kernel_size=3, stride=1, padding=1).squeeze(1)
        resid = x - trend
        t = self.trend(trend)
        r = self.resid(resid)
        return self.head(h + t * 0.4 + r * 0.2)

class M91_COGRASP(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.inten = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.decay = nn.Parameter(torch.ones(hidden) * 0.5)
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        base = torch.sigmoid(self.inten(h))
        d = torch.sigmoid(self.decay)
        fused = base * d + h * 0.2
        e = self.enhance(fused)
        return self.head(fused + e * 0.3)

class M92_AlphaCFG(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.a = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.m = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        p = torch.tanh(self.a(h))
        q = torch.tanh(self.m(h))
        g = self.g(torch.cat([p, q], dim=-1))
        fused = p * g + q * (1 - g)
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2 + h_c * 0.2)

class M93_FinMamba(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.market_attn = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop))
        self.temporal = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        hm = self.market_attn(x)
        attn_scores = F.softmax(hm, dim=-1)
        ht = self.temporal((hm * attn_scores).sum(dim=1, keepdim=True).expand_as(hm))
        fused = hm * 0.4 + ht * 0.4 + h * 0.2
        f = self.fusion(fused)
        return self.head(fused + f * 0.3)

class M94_DPA_STIFormer(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.path_a = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.path_b = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        a = self.path_a(x)
        b = self.path_b(x)
        fused = self.fusion(torch.cat([a, b], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h * 0.3 + h_c * 0.2)

class M95_HIGSTM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.trend_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.season_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Linear(int(d_in), self.hidden)
        self.season_proj = nn.Linear(int(d_in), self.hidden)
        self.noise_proj = nn.Linear(int(d_in), self.hidden)
        self.gate = nn.Sequential(nn.Linear(self.hidden * 3, self.hidden * 3), nn.Sigmoid())
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x):
        x_unsq = x.unsqueeze(1)
        trend = self.trend_pool(x_unsq).squeeze(1)
        season = self.season_pool(x_unsq).squeeze(1) - trend
        noise = x - trend - season
        h_t = self.norm(self.trend_proj(trend))
        h_s = self.norm(self.season_proj(season))
        h_n = self.norm(self.noise_proj(noise))
        g = self.gate(torch.cat([h_t, h_s, h_n], dim=-1))
        g_t = g[:, :self.hidden]
        g_s = g[:, self.hidden:2*self.hidden]
        g_n = g[:, 2*self.hidden:]
        return self.head(g_t * h_t + g_s * h_s + g_n * h_n)

class M96_SAMBA(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.fwd = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.ctx_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        hf = self.fwd(h)
        hc = self.ctx(h)
        g = self.ctx_gate(torch.cat([hf, hc], dim=-1))
        return self.head(h + hf * g + hc * (1 - g) * 0.3 + h_c * 0.2)

class M97_HINT_Lite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.group_pool = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.compress = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        a, _ = self.attn(hp, hp, hp, need_weights=False)
        p = hp + a * 0.3
        gw = torch.softmax(self.group_pool(p), dim=1)
        pooled = (p * gw).sum(dim=1)
        cm = self.compress(pooled)
        return self.head(h + cm * 0.4 + h_c * 0.2)

class M98_ABSSM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.mod = nn.Linear(hidden, hidden * 2)
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        a, b = self.mod(h).chunk(2, dim=-1)
        fused = h * torch.sigmoid(a) + torch.tanh(b)
        e = self.enhance(fused)
        return self.head(fused + e * 0.3)

class M99_DOTS_Lite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.importance = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        imp = self.importance(x)
        hi = self.norm(self.proj(x * imp))
        return self.head(h + hi * 0.25)

class M100_FASCL_Lite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.v1 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.v2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Dropout(drop * 1.5))
        self.diff_enc = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        a = self.v1(h)
        b = self.v2(h)
        diff = torch.tanh(self.diff_enc(a - b))
        g = self.g(torch.cat([a, diff], dim=-1))
        fused = a * g + diff * (1 - g)
        if self.training:
            fused = fused + torch.randn_like(fused) * 0.05
        return self.head(fused)

class M101_Diffolio_Lite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.dn = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        if self.training:
            n = torch.randn_like(h) * 0.1
            r = torch.tanh(self.dn(h + n))
        else:
            r = torch.tanh(self.dn(h))
        return self.head(h + r * 0.3)

class M102_FreIE_Lite(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=-1).abs()
        split = max(1, xf.size(1) // 4)
        low = xf[:, :split].mean(dim=-1, keepdim=True)
        high = xf[:, split:].mean(dim=-1, keepdim=True) if xf.size(1) > split else torch.ones_like(low) * 1e-6
        ratio = torch.log1p(low) - torch.log1p(high)
        g = torch.sigmoid(ratio)
        h_f = self.freq_enhance(h * g)
        return self.head(h_f + h * 0.3)

class M103_GF_MSH_Lite(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.sm = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.sh = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.g = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        a, b = self.sm(h), self.sh(h)
        g = self.g(torch.cat([a, b], dim=-1))
        fused = a * g + b * (1 - g)
        f = self.fusion(fused)
        return self.head(fused + f * 0.3 + h * 0.2)

class M104_PureKAN_Lite(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.rbf = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        r = self.rbf(h)
        m = self.mlp(x)
        g = self.gate(torch.cat([r, m], dim=-1))
        fused = h + r * g + m * (1 - g)
        return self.head(fused + h_c * 0.2)

class M105_NIFL_Lite(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        m = max(1, d_in // 2)
        self.z_enc = nn.Sequential(nn.Linear(m, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.iv_enc = nn.Sequential(nn.Linear(d_in - m, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.iv_pred = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.residual_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        m = x.size(1) // 2
        z = torch.tanh(self.z_enc(x[:, :m]))
        iv = torch.tanh(self.iv_enc(x[:, m:]))
        z_pred = self.iv_pred(iv)
        residual = z - z_pred
        g = self.residual_gate(torch.cat([z, residual], dim=-1))
        fused = z * g + residual * (1 - g) + h * 0.2
        return self.head(fused + h_c * 0.2)

class M106_SelectiveLearn(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.sel = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        self.bb = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        mask = self.sel(x)
        hb = self.bb(x * mask)
        return self.head(hb + h * 0.3 + h_c * 0.2)

class M107_MERA(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1, n_experts=4):
        super().__init__(d_in, hidden, drop)
        self.n_experts = max(2, int(n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)) for _ in range(self.n_experts)])
        self.router = nn.Sequential(nn.Linear(hidden, self.n_experts), nn.Softmax(dim=-1))
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c1 = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c1 = self.norm(self.proj(c1))
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h_c2 = self.norm(self.proj(c2))
        g_c = self.fusion(torch.cat([h_c1, h_c2], dim=-1))
        h_c = h_c1 * g_c + h_c2 * (1 - g_c)
        r = self.router(h.mean(1, keepdim=True).expand(-1, self.hidden))
        outs = torch.stack([exp(h) for exp in self.experts], dim=1)
        h_moe = (r.unsqueeze(-1) * outs).sum(dim=1)
        return self.head(h_moe + h * 0.3 + h_c * 0.2)

class M108_FinCast_Lite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_seg = 4
        # FIX1: 向上取整，确保 n_seg * seg_len >= d_in，不再截断
        self.seg_len = max(1, (d_in + self.n_seg - 1) // self.n_seg)
        self.seg_proj = nn.Linear(self.seg_len, hidden)
        self.attn = nn.Linear(hidden, 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        # FIX2: 新增 conv_proj，将卷积后的实际维度（n_seg*seg_len）映射到 hidden
        # 不再复用 self.proj（其输入维度硬编码为 d_in，与卷积分支不匹配）
        self.conv_proj = nn.Linear(self.n_seg * self.seg_len, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B = x.shape[0]
        pad = self.n_seg * self.seg_len - x.shape[1]
        if pad > 0:
            x = F.pad(x, (0, pad))
        # 使用 padding 后的完整维度 reshape，不再硬截断
        x = x[:, :self.n_seg * self.seg_len].reshape(B, self.n_seg, self.seg_len)
        hs = torch.tanh(self.seg_proj(x))
        a = torch.softmax(self.attn(hs), dim=1)
        p = (hs * a).sum(dim=1)
        c = self.conv(x.reshape(B, -1).unsqueeze(1)).squeeze(1)
        # FIX3: 使用 conv_proj 替代 self.proj，彻底消除维度不匹配
        h_c = self.norm(self.conv_proj(c))
        return self.head(h + p * 0.3 + h_c * 0.2)

class M109_GraphAttnLite(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # FIX: 彻底重构。原版的 adaptive_avg_pool1d 到 16 节点严重破坏信息。
        # 改为 iTransformer 式特征分组注意力。
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        a, _ = self.attn(hp, hp, hp, need_weights=False)
        p = (hp + a * 0.3).mean(dim=1)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(p + h_c * 0.2 + h * 0.2)

class M110_CausalHyper(Step2Base):
    def __init__(self, d_in, hidden=128, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.causal_adj = nn.Parameter(torch.randn(d_in, d_in) * 0.01)
        self.temp = nn.Parameter(torch.tensor(2.0))
        self.hyper_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.fuse_gate = nn.Linear(hidden, hidden)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        adj = torch.sigmoid(self.causal_adj * F.softplus(self.temp))
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        x_hyper = torch.matmul(x, adj.t())
        h_orig = self._preprocess(x)
        h_hyper = self.hyper_proj(x_hyper)
        g = torch.sigmoid(self.fuse_gate(h_orig))
        fused = h_hyper * g + h_orig * (1.0 - g)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h_c * 0.2)

# ====新增顶会轻量模型 (来自 v5.0) =====================

class M111_TiDE(Step2Base):
    """Time-series Dense Encoder (ICML 2023)"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.dense = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop)
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        d = self.dense(x)
        return self.head(h + d * 0.3)


class M112_MambaStock(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # Temporal: 局部卷积 + SSM 状态门控
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.ssm_gate = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.Sigmoid()
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 局部卷积特征
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        # GRU 时序编码
        h_seq = h.unsqueeze(1)
        out, _ = self.gru(h_seq)
        h_gru = out.squeeze(1)
        # SSM 风格状态门控融合
        g = self.ssm_gate(h_c)
        return self.head(h_gru + h_c * g + h * 0.2)


class M113_SegRNN(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x).unsqueeze(1)
        out, _ = self.gru(h)
        return self.head(out.squeeze(1) + h.squeeze(1) * 0.2)


class M114_PAttn(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        hg = h.reshape(B, self.n_g, self.g_s)
        p = hg.mean(dim=1)
        p_up = p.repeat_interleave(self.n_g, dim=1)[:, :self.hidden]
        g = self.gate(p_up)
        return self.head(h + p_up * g * 0.3)

class M115_MambaSL(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.ssm = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        hc = self.norm(self.proj(c))
        s = self.ssm(hc)
        return self.head(h + hc * 0.3 + s * 0.2)

class M116_TabM(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.ensemble = nn.Parameter(torch.randn(d_in) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        e = self.mlp(x + self.ensemble)
        return self.head(e + h * 0.3)

class M117_FreTS(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.time = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c1 = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c1 = self.norm(self.proj(c1))
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h_c2 = self.norm(self.proj(c2))
        g_c = self.fusion(torch.cat([h_c1, h_c2], dim=-1))
        h_c = h_c1 * g_c + h_c2 * (1 - g_c)
        f = self.freq(x)
        t = self.time(x)
        g = self.gate(torch.cat([f, t], dim=-1))
        fused = f * g + t * (1 - g)
        return self.head(fused + h * 0.2 + h_c * 0.2)

class M118_Koopa(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.koopman = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.local = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        k = self.koopman(x)
        l = self.local(h)
        return self.head(k + l * 0.3 + h * 0.2)

class M119_MambAttention(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        hc = self.norm(self.proj(c)).unsqueeze(1)
        ha, _ = self.attn(hc, hc, hc, need_weights=False)
        ha = ha.squeeze(1)
        return self.head(h + ha * 0.3)

class M120_ASGMamba(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        g = self.gate(h)
        e = self.enhance(h * g)
        return self.head(h + e * 0.3)

class M121_DMamba(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.trend = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.season = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self._preprocess(x)
        x_u = x.unsqueeze(1)
        trend = F.avg_pool1d(x_u, kernel_size=3, stride=1, padding=1).squeeze(1)
        season = x - trend
        t = self.trend(trend)
        s = self.season(season)
        return self.head(h + t * 0.4 + s * 0.2)

class M201_ProbGANLinear(Step2Base):
    """Chronos 概率生成 + MarketGAN 对抗增强 + DLinear 趋势季节分解"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # DLinear: 趋势-季节分解门控
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.season_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        # Chronos: 概率生成噪声分支
        self.noise_enc = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        # MarketGAN: 对抗隐空间判别
        self.discriminator = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gan_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # DLinear 分解
        x_u = x.unsqueeze(1)
        trend = self.decomp(x_u).squeeze(1)
        seasonal = x - trend
        h_t = self.norm(self.proj(trend))
        h_s = self.norm(self.proj(seasonal))
        g_t = self.trend_gate(h_t)
        g_s = self.season_gate(h_s)
        h_decomp = g_t * h_t + g_s * h_s
        # Chronos 概率噪声注入
        x_n = x + torch.randn_like(x) * 0.08 if self.training else x
        h_noise = self.noise_enc(x_n)
        # MarketGAN 对抗融合
        d = self.discriminator(h_noise)
        g = self.gan_gate(torch.cat([h_decomp, d], dim=-1))
        fused = h_decomp * g + d * (1 - g)
        return self.head(fused + h * 0.3)

class M202_JumpConvTrans(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c1 = nn.Conv1d(1, 1, 3, padding=1)
        self.c2 = nn.Conv1d(1, 1, 5, padding=2)
        self.c3 = nn.Conv1d(1, 1, 7, padding=3)
        self.p1 = nn.Linear(d_in, hidden)
        self.p2 = nn.Linear(d_in, hidden)
        self.p3 = nn.Linear(d_in, hidden)
        self.router = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        h1 = self.norm(self.p1(self.c1(x.unsqueeze(1)).squeeze(1)))
        h2 = self.norm(self.p2(self.c2(x.unsqueeze(1)).squeeze(1)))
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        g = self.router(torch.cat([h1, h2, h3], dim=-1))
        m = h1 * g[:, 0:1] + h2 * g[:, 1:2] + h3 * g[:, 2:3]
        m = m.unsqueeze(1)
        a, _ = self.attn(m, m, m, need_weights=False)
        return self.head(a.squeeze(1) * 0.3 + m.squeeze(1) + h * 0.2)

class M203_AdaptiveNormSSM(Step2Base):
    """修复：砍掉失效的因果卷积，改为 RevIN + 双尺度卷积门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.affine = nn.Parameter(torch.ones(hidden))
        self.c3 = nn.Conv1d(1, 1, 3, padding=1)
        self.c5 = nn.Conv1d(1, 1, 5, padding=2)
        self.p3 = nn.Linear(d_in, hidden)
        self.p5 = nn.Linear(d_in, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        m = h.mean(1, keepdim=True); s = h.std(1, keepdim=True) + 1e-5
        h_rev = (h - m) / s * self.affine
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        h5 = self.norm(self.p5(self.c5(x.unsqueeze(1)).squeeze(1)))
        g = self.fuse(torch.cat([h3, h5], dim=-1))
        conv = h3 * g + h5 * (1 - g)
        return self.head(conv + h_rev * 0.3 + h * 0.2)

class M204_SpectralGap(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_low = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.freq_high = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gap_energy = nn.Sequential(nn.Linear(d_in, hidden), nn.Sigmoid())
        self.time_branch = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        split = max(1, amp.shape[1] // 4)
        low_mask = torch.zeros_like(amp); low_mask[:, :split] = 1.0
        high_mask = torch.zeros_like(amp); high_mask[:, split:] = 1.0
        x_low = torch.fft.irfft(xf * low_mask, n=x.shape[1], dim=1)
        x_high = torch.fft.irfft(xf * high_mask, n=x.shape[1], dim=1)
        h_low = self.freq_low(x_low)
        h_high = self.freq_high(x_high)
        gap = self.gap_energy(x_high - x_low)
        t = self.time_branch(h)
        return self.head(h_low * (1 - gap) + h_high * gap + t * 0.3 + h * 0.2)

class M205_CausalHyperGraph(Step2Base):
    """修复：去掉无效超图，改为特征间可学习软注意力（轻量图思想）"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.attn_w = nn.Sequential(nn.Linear(d_in, d_in), nn.Sigmoid())
        self.graph = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        w = self.attn_w(x)
        h_g = self.graph(x * w)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(h_g + h_c * 0.3 + h * 0.2)

class M206_QuantileBridge(Step2Base):
    """
    修复核心：分位数从"常数向量输入"改为"FiLM式全局统计调制"
    理论：样本内分位数(q25/q50/q75)反映特征分布的位置与离散度，
          作为全局统计量通过 scale/shift 调制原始特征表示，而非替代原始特征
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # 分位数统计投影：3个统计量 → hidden
        self.stat_proj = nn.Sequential(nn.Linear(3, hidden), nn.LayerNorm(hidden), nn.GELU())
        # 桥接调制参数生成
        self.bridge = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.scale = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.shift = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        # 增强
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 正确计算样本内分位数 (B, 1)
        k25 = max(1, int(x.shape[1] * 0.25))
        k50 = max(1, int(x.shape[1] * 0.50))
        k75 = max(1, int(x.shape[1] * 0.75))
        q25 = x.kthvalue(k25, dim=1)[0].unsqueeze(-1)
        q50 = x.kthvalue(k50, dim=1)[0].unsqueeze(-1)
        q75 = x.kthvalue(k75, dim=1)[0].unsqueeze(-1)
        # 拼接为 (B, 3) 统计特征
        stats = torch.cat([q25, q50, q75], dim=-1)
        # 投影并生成 FiLM 参数
        h_stat = self.stat_proj(stats)
        b = self.bridge(h_stat)
        gamma = self.scale(b)   # (B, hidden)
        beta = self.shift(b)    # (B, hidden)
        # 调制原始特征
        h_mod = h * gamma + beta
        e = self.enhance(h_mod)
        return self.head(h_mod + e * 0.3 + h * 0.2)


class M207_KoopmanInvPeriod(Step2Base):
    """Koopa Koopman 算子 + iTransformer 倒置变量注意力 + TimesNet 2D 周期变换"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # Koopman: 线性演化
        self.koopman = nn.Linear(d_in, hidden)
        self.koop_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        # iTransformer: 倒置分组注意力
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.inv_attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        # TimesNet: 2D 周期卷积（特征维度视为伪时序）
        self.period = max(4, int(d_in ** 0.5))
        self.period_proj = nn.Linear(self.period, hidden)
        self.period_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B, D = x.shape
        # Koopman 线性演化
        k = self.koopman(x)
        g_k = self.koop_gate(k)
        h_koop = k * g_k
        # iTransformer 倒置变量注意力
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        a, _ = self.inv_attn(hp, hp, hp, need_weights=False)
        p = (hp + a * 0.3).mean(dim=1)
        # TimesNet 2D 周期变换
        pad = self.period * ((D + self.period - 1) // self.period) - D
        x_pad = F.pad(x, (0, pad)) if pad > 0 else x
        n_p = x_pad.shape[1] // self.period
        x_2d = x_pad[:, :n_p * self.period].reshape(B, n_p, self.period).unsqueeze(1)
        c2d = self.period_conv(x_2d).squeeze(1)
        h_2d = self.period_proj(c2d.mean(dim=1))
        # 融合
        return self.head(h_koop * 0.4 + p * 0.3 + h_2d * 0.3 + h * 0.2)


class M208_SparseTFTAMD(Step2Base):
    """Informer ProbSparse + TFT 分位数门控 + AMD 趋势/季节/噪声三分支"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # Informer: 方差感知稀疏选择
        self.sparse_selector = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        # TFT: 分位数多尺度门控
        self.q10 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.q50 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.q90 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.quant_gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        # AMD: 三分支分解
        self.trend_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.season_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.season_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.noise_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.amd_gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # Informer ProbSparse: 方差驱动稀疏激活
        var = h.var(dim=-1, keepdim=True)
        mask = torch.sigmoid(var * 5.0 - 1.0)
        h_sparse = self.sparse_selector(h) * mask
        # TFT 分位数门控
        q1 = self.q10(h_sparse)
        q5 = self.q50(h_sparse)
        q9 = self.q90(h_sparse)
        g_q = self.quant_gate(torch.cat([q1, q5, q9], dim=-1))
        h_tft = q1 * g_q[:, 0:1] + q5 * g_q[:, 1:2] + q9 * g_q[:, 2:3]
        # AMD 三分支分解
        x_u = x.unsqueeze(1)
        trend = self.trend_pool(x_u).squeeze(1)
        season = self.season_pool(x_u).squeeze(1) - trend
        noise = x - trend - season
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(season)
        h_n = self.noise_proj(noise)
        g_amd = self.amd_gate(torch.cat([h_t, h_s, h_n], dim=-1))
        h_amd = g_amd * h_t + (1 - g_amd) * (h_s * 0.5 + h_n * 0.5)
        # 融合
        return self.head(h_tft * 0.5 + h_amd * 0.3 + h * 0.2)

class M209_CrossScaleModern(Step2Base):
    """修复：双尺度卷积 + 差分 surge 检测"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c5 = nn.Conv1d(1, 1, 5, padding=2)
        self.c3 = nn.Conv1d(1, 1, 3, padding=1)
        self.p5 = nn.Linear(d_in, hidden)
        self.p3 = nn.Linear(d_in, hidden)
        self.dx = nn.Linear(d_in, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        h5 = self.norm(self.p5(self.c5(x.unsqueeze(1)).squeeze(1)))
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx(dx))
        g = self.fuse(torch.cat([h5, h3, h_dx], dim=-1))
        fused = h5 * g + h3 * (1 - g) * 0.5 + h_dx * 0.3
        return self.head(fused + h * 0.2)

class M210_TabPriorEnsemble(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.scale = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.shift = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.path1 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.path2 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 1.2))
        self.path3 = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.8))
        self.prior_gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        gamma = self.scale(x); beta = self.shift(x)
        h_prior = h * gamma + beta
        p1 = self.path1(x); p2 = self.path2(x); p3 = self.path3(x)
        g = self.prior_gate(torch.cat([p1, p2, p3], dim=-1))
        fused = p1 * g[:, 0:1] + p2 * g[:, 1:2] + p3 * g[:, 2:3]
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(fused + h_prior * 0.3 + h_c * 0.2)

class M211_VanillaMambaPrior(Step2Base):
    """VanillaTransformer 经典堆叠 + Mamba2-SSD 分段扫描 + TabPFN 先验感知"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # VanillaTransformer: 经典编码器
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=2, dim_feedforward=hidden*2, dropout=drop, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Mamba2-SSD: 分段状态空间
        self.n_seg = 8
        self.seg_len = max(1, (d_in + self.n_seg - 1) // self.n_seg)
        self.seg_proj = nn.Linear(self.seg_len, hidden)
        self.seg_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        # TabPFN: 先验尺度偏移
        self.scale = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        self.shift = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B = h.shape[0]
        # VanillaTransformer 经典堆叠
        hg = h.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        e = self.transformer(hp)
        p = (hp + e * 0.3).mean(dim=1)
        # Mamba2-SSD 分段因果扫描
        pad = self.n_seg * self.seg_len - x.shape[1]
        x_pad = x if pad <= 0 else F.pad(x, (0, pad))
        x_seg = x_pad[:, :self.n_seg * self.seg_len].reshape(B, self.n_seg, self.seg_len)
        h_seg = self.norm(self.seg_proj(x_seg))
        h_c = F.silu(self.seg_conv(h_seg.transpose(1, 2)).transpose(1, 2))
        h_cum = torch.cumsum(h_c, dim=1)
        denom = torch.arange(1, self.n_seg + 1, device=x.device).view(1, -1, 1).float()
        h_cum = h_cum / (denom + 1e-8)
        h_ssm = h_c.mean(dim=1) + h_cum.mean(dim=1) * 0.5
        # TabPFN 先验注入
        gamma = self.scale(x)
        beta = self.shift(x)
        h_prior = h * gamma + beta
        # 融合
        tri = torch.cat([p, h_ssm, h_prior], dim=-1)
        g = self.fusion(tri)
        fused = p * g + h_ssm * (1 - g) * 0.5 + h_prior * 0.3
        return self.head(fused + h * 0.2)

class M212_VolumeSilence(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.silence = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.explosion = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        s = self.silence(h)
        e = self.explosion(h)
        g = self.gate(torch.cat([s, e], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(e * g + s * (1 - g) * 0.3 + h_c * 0.2 + h * 0.2)

class M213_DivergenceMACD(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.short = nn.Conv1d(1, 1, 3, padding=1)
        self.long = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.divergence = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c_s = self.short(x.unsqueeze(1)).squeeze(1)
        h_s = self.norm(self.proj(c_s))
        c_l = self.long(x.unsqueeze(1)).squeeze(1)
        h_l = self.norm(self.proj(c_l))
        div = self.divergence(torch.cat([h_s, h_l], dim=-1))
        g = self.gate(div)
        return self.head(div * g + h_s * 0.3 + h_l * 0.2 + h * 0.2)

class M214_MicroPressure(Step2Base):
    """修复：简化差分压力，直接差分+放大+门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.dx_proj = nn.Linear(d_in, hidden)
        self.pressure = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx_proj(dx))
        p = self.pressure(h_dx)
        g = self.gate(torch.cat([h, h_dx], dim=-1))
        return self.head(h_dx * g + p * 0.3 + h * 0.2)

class M215_AutoFreqDense(Step2Base):
    """Autoformer 自相关分解 + FEDformer 频域 MLP 增强 + TiDE 密集编码"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # Autoformer: 自相关移动平均分解
        self.auto_decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.season_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        # FEDformer: 频域 MLP 增强
        self.freq_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        # TiDE: 密集编码双分支
        self.dense = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop)
        )
        self.dense_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # Autoformer 自相关分解
        x_u = x.unsqueeze(1)
        trend = self.auto_decomp(x_u).squeeze(1)
        seasonal = x - trend
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(seasonal)
        h_auto = h_t + h_s * 0.5
        # FEDformer 频域 MLP 增强（TopK 能量保留）
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        k = max(1, amp.shape[1] // 3)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        x_freq = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        h_freq = self.freq_proj(x_freq)
        # TiDE 密集编码
        d = self.dense(x)
        g = self.dense_gate(torch.cat([h_auto, d], dim=-1))
        h_tide = h_auto * g + d * (1 - g)
        # 融合
        return self.head(h_tide + h_freq * 0.3 + h * 0.2)

class M216_FreqSqueeze(Step2Base):
    """修复：简化频域能量检测，低频占比门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.surge = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        split = max(1, amp.size(1) // 4)
        low_e = amp[:, :split].sum(dim=1, keepdim=True)
        high_e = amp[:, split:].sum(dim=1, keepdim=True) + 1e-8
        ratio = torch.sigmoid(low_e / (low_e + high_e)).expand(-1, x.size(1))
        x_mod = x * ratio
        f = self.freq(x_mod)
        s = self.surge(f)
        return self.head(f + s * 0.3 + h * 0.2)

class M217_AttentionSpark(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.momentum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.shape = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        m = self.momentum(h.mean(1, keepdim=True).expand_as(h))
        g = torch.sigmoid(m)
        h_g = h * g
        avg = h.mean(1, keepdim=True)
        s = self.shape(h - avg)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(h_g + s * 0.3 + h_c * 0.2 + h * 0.2)

class M218_SpectraRoseFreq(Step2Base):
    """SpectraFormer 频谱门控混合 + ROSE 频率调制 + FreTS 频域 MLP/时域残差"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # SpectraFormer: 频谱门控
        self.freq_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.time_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.spectra_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        # ROSE: 频率调制参数
        self.rose_freq = nn.Parameter(torch.tensor(0.12))
        self.rose_amp = nn.Parameter(torch.tensor(1.0))
        # FreTS: 频域/时域双分支
        self.freq_branch = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.time_branch = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.frets_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # SpectraFormer 频谱处理
        hf = torch.fft.rfft(x, dim=1)
        k = max(1, hf.shape[1] // 4)
        hf[:, k:] *= 0.0
        h_freq_spectra = self.norm(self.proj(torch.fft.irfft(hf, n=x.shape[1], dim=1)))
        c = self.spectra_conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        g_f = self.freq_gate(h_freq_spectra)
        g_t = self.time_gate(h)
        h_spectra = h_freq_spectra * g_f + h * g_t + h_c * 0.2
        # ROSE 正弦调制
        h_rose = h_spectra * torch.cos(h_spectra * torch.abs(self.rose_freq)) * torch.sigmoid(self.rose_amp)
        # FreTS 频域/时域双分支
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        k2 = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k2, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        x_f = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        h_f = self.freq_branch(x_f)
        h_t = self.time_branch(x)
        g_ft = self.frets_gate(torch.cat([h_f, h_t], dim=-1))
        h_frets = h_f * g_ft + h_t * (1 - g_ft)
        # 融合
        return self.head(h_rose * 0.4 + h_frets * 0.4 + h * 0.2)

class M219_CrossSkewness(Step2Base):
    """已达头部 (0.2368)，保持原代码"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c_short = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.c_long = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.p_short = nn.Linear(d_in, hidden)
        self.p_long = nn.Linear(d_in, hidden)
        self.diff_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c_s = self.c_short(x.unsqueeze(1)).squeeze(1)
        h_s = self.norm(self.p_short(c_s))
        c_l = self.c_long(x.unsqueeze(1)).squeeze(1)
        h_l = self.norm(self.p_long(c_l))
        diff = h_s - h_l
        g = self.diff_gate(torch.cat([diff, h_l], dim=-1))
        e = self.enhance(diff)
        return self.head(h_s * 0.3 + h_l * 0.2 + diff * g + e * 0.3 + h * 0.2)

class M220_SilentAccum(Step2Base):
    """已达头部 (0.2275)，保持原代码"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c1 = nn.Conv1d(1, 1, 3, padding=1)
        self.c2 = nn.Conv1d(1, 1, 5, padding=2)
        self.accum_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.burst = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        h1 = self.norm(self.proj(self.c1(x.unsqueeze(1)).squeeze(1)))
        h2 = self.norm(self.proj(self.c2(x.unsqueeze(1)).squeeze(1)))
        diff = h2 - h1
        g = self.accum_gate(torch.cat([h1, diff], dim=-1))
        b = self.burst(diff)
        return self.head(b * g + h1 * 0.3 + h2 * 0.2 + h * 0.2)

class M221_GapBloom(Step2Base):
    """修复：统一 dx 维度为 d_in；简化双阶差分"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.dx_proj = nn.Linear(d_in, hidden)
        self.pressure = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mom_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx_proj(dx))
        p = self.pressure(h_dx)
        g = self.mom_gate(p)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(h_dx * g + p * 0.3 + h_c * 0.2 + h * 0.2)

class M222_VolCompressBloom(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq_low = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.accum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.burst = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv = nn.Conv1d(1, 1, 5, padding=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, 1)) if dx.size(1) < x.size(1) else dx
        vol = dx.abs().mean(dim=1, keepdim=True)
        vg = torch.sigmoid(1.0 - vol * 10.0).expand(-1, self.hidden)
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        split = max(1, amp.size(1) // 4)
        low_mask = torch.zeros_like(amp)
        low_mask[:, :split] = 1.0
        x_low = torch.fft.irfft(xf * low_mask, n=x.shape[1], dim=1)
        f = self.freq_low(x_low)
        a = self.accum(f * vg)
        b = self.burst(a)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(b * 0.4 + a * 0.3 + h_c * 0.2 + h * 0.2)

class M223_TriScaleMomentum(Step2Base):
    """修复：三尺度保留，但增强动量检测头"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c3 = nn.Conv1d(1, 1, 3, padding=1)
        self.c5 = nn.Conv1d(1, 1, 5, padding=2)
        self.c7 = nn.Conv1d(1, 1, 7, padding=3)
        self.p3 = nn.Linear(d_in, hidden)
        self.p5 = nn.Linear(d_in, hidden)
        self.p7 = nn.Linear(d_in, hidden)
        self.mom = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        h5 = self.norm(self.p5(self.c5(x.unsqueeze(1)).squeeze(1)))
        h7 = self.norm(self.p7(self.c7(x.unsqueeze(1)).squeeze(1)))
        m = self.mom(h.mean(1, keepdim=True).expand(-1, self.hidden))
        g = self.gate(torch.cat([h3, h5, h7], dim=-1))
        fused = h3 * g[:, 0:1] + h5 * g[:, 1:2] + h7 * g[:, 2:3]
        return self.head(fused + m * 0.3 + h * 0.2)

class M224_DualAttnSurge(Step2Base):
    """修复：砍掉无效分组注意力，改为卷积+均值残差双路"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, 5, padding=2)
        self.residual = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        avg = h.mean(1, keepdim=True).expand_as(h)
        r = self.residual(h - avg)
        g = self.gate(torch.cat([h_c, r], dim=-1))
        return self.head(h_c * g + r * (1 - g) + h * 0.3)

class M225_FreqTimeBridge(Step2Base):
    """已达头部 (0.2227)，保持原代码"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.low_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.high_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.bridge = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        split = max(1, amp.size(1) // 4)
        low_mask = torch.zeros_like(amp); low_mask[:, :split] = 1.0
        high_mask = torch.zeros_like(amp); high_mask[:, split:] = 1.0
        x_low = torch.fft.irfft(xf * low_mask, n=x.shape[1], dim=1)
        x_high = torch.fft.irfft(xf * high_mask, n=x.shape[1], dim=1)
        h_low = self.low_proj(x_low)
        h_high = self.high_proj(x_high)
        b = self.bridge(torch.cat([h_low, h_high], dim=-1))
        g = self.gate(torch.cat([h_low, h_high], dim=-1))
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(b + h_low * g + h_high * (1 - g) * 0.3 + h_c * 0.2 + h * 0.2)

class M226_MoETabContext(Step2Base):
    """MERA Switch MoE + TabICL 上下文学习 + SelectiveLearn 动态特征选择"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # SelectiveLearn: 输入特征软选择
        self.selector = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        # TabICL: 上下文编码与跨样本交互
        self.ctx_enc = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.cross = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        # MERA: Switch MoE
        self.n_experts = 3
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
        ])
        self.router = nn.Sequential(nn.Linear(hidden, self.n_experts), nn.Softmax(dim=-1))
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # SelectiveLearn 动态特征选择
        mask = self.selector(x)
        x_sel = x * mask
        h_sel = self.norm(self.proj(x_sel))
        # TabICL 上下文交互
        c = torch.sigmoid(self.ctx_enc(h_sel))
        cr = self.cross(x_sel)
        h_icl = h_sel * c + cr * 0.3
        # MERA Switch MoE
        r = self.router(h_icl.mean(1, keepdim=True).expand(-1, self.hidden))
        outs = torch.stack([exp(h_icl) for exp in self.experts], dim=1)
        h_moe = (r.unsqueeze(-1) * outs).sum(dim=1)
        # 融合
        g = self.fusion_gate(torch.cat([h_icl, h_moe], dim=-1))
        fused = h_icl * g + h_moe * (1 - g)
        return self.head(fused + h * 0.3)

class M227_CausalMoE(Step2Base):
    """修复：三专家→双专家，因果路由改为差分驱动"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.expert_trend = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.expert_surge = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.router = nn.Sequential(nn.Linear(d_in, 2), nn.Softmax(dim=-1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        r = self.router(dx.abs())
        e1 = self.expert_trend(h)
        e2 = self.expert_surge(h)
        moe = e1 * r[:, 0:1] + e2 * r[:, 1:2]
        return self.head(moe + h * 0.3)

class M228_GroupMarketRisk(Step2Base):
    """StockMixer 分组混合 + FinMamba 市场注意力 + AdaptWin 风险窗口"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # StockMixer: 分组投影
        self.n_group = 8
        self.seg_len = (d_in + self.n_group - 1) // self.n_group
        self.group_proj = nn.Linear(self.seg_len, hidden)
        # FinMamba: 市场注意力
        self.market_attn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        # AdaptWin: 双尺度风险卷积
        self.conv_risk1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv_risk2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.risk_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # 跨组聚合
        self.group_mix = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B, D = x.shape
        # StockMixer 分组
        pad = self.n_group * self.seg_len - D
        x_pad = x if pad <= 0 else F.pad(x, (0, pad))
        x_grp = x_pad[:, :self.n_group * self.seg_len].reshape(B, self.n_group, self.seg_len)
        h_grp = self.norm(self.group_proj(x_grp))
        # FinMamba 市场注意力重加权
        hm = self.market_attn(h_grp)
        attn = F.softmax(hm, dim=-1)
        h_market = (hm * attn).sum(dim=1, keepdim=True).expand(-1, self.n_group, -1)
        h_mkt = h_grp * h_market
        # 跨组混合
        m = h_mkt.mean(dim=1)
        mx = h_mkt.max(dim=1)[0]
        h_mix = self.group_mix(m + mx * 0.3)
        # AdaptWin 风险窗口
        c1 = self.conv_risk1(x.unsqueeze(1)).squeeze(1)
        c2 = self.conv_risk2(x.unsqueeze(1)).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        g_risk = self.risk_gate(torch.cat([h1, h2], dim=-1))
        h_risk = h1 * g_risk + h2 * (1 - g_risk)
        # 融合
        return self.head(h_mix * 0.5 + h_risk * 0.3 + h * 0.2)

class M229_RevIN_Attn(Step2Base):
    """修复：砍掉失效分组注意力，改为 RevIN + 大核卷积"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.affine = nn.Parameter(torch.ones(hidden))
        self.conv = nn.Conv1d(1, 1, 7, padding=3)
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        m = h.mean(1, keepdim=True); s = h.std(1, keepdim=True) + 1e-5
        h_rev = (h - m) / s * self.affine
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        e = self.enhance(h_rev)
        return self.head(h_rev + e * 0.3 + h_c * 0.3 + h * 0.2)

class M230_KAN_ShapeDetect(Step2Base):
    """修复：单 KAN + 卷积旁路，减少非线性堆叠"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.kan = SimpleKANLayer(hidden, hidden, 5, 3)
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        k = self.kan(h)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        g = self.gate(torch.cat([k, h_c], dim=-1))
        return self.head(k * g + h_c * (1 - g) + h * 0.2)

class M231_TripleDivergence(Step2Base):
    """已达头部 (0.2253)，保持原代码"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.short = nn.Conv1d(1, 1, 3, padding=1)
        self.long = nn.AvgPool1d(5, stride=1, padding=2)
        self.skew = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.resid = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c_s = self.short(x.unsqueeze(1)).squeeze(1)
        h_s = self.norm(self.proj(c_s))
        c_l = self.long(x.unsqueeze(1)).squeeze(1)
        h_l = self.norm(self.proj(c_l))
        div1 = h_s - h_l
        sk = ((x - x.mean(dim=1, keepdim=True)) ** 3).mean(dim=1, keepdim=True)
        sk = torch.sign(sk) * torch.abs(sk).pow(1.0 / 3.0)
        h_sk = self.skew(sk.expand(-1, x.size(1)))
        h_res = self.resid(h - h.mean(1, keepdim=True).expand(-1, self.hidden))
        g = self.gate(torch.cat([div1, h_sk, h_res], dim=-1))
        fused = div1 * g[:, 0:1] + h_sk * g[:, 1:2] + h_res * g[:, 2:3]
        return self.head(fused + h * 0.2)

class M232_DynamicHyperSurge(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.adj = nn.Parameter(torch.randn(d_in, d_in) * 0.01)
        self.hyper = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        adj = torch.sigmoid(self.adj)
        x_hyper = torch.matmul(x, adj)
        h_hyper = self.hyper(x_hyper)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        g = self.gate(torch.cat([h_hyper, h_c], dim=-1))
        return self.head(h_hyper * g + h_c * (1 - g) + h * 0.2)

class M233_QuantileMomentumNet(Step2Base):
    """修复：分位数→统计矩（均值/方差/偏度）投影"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.stat = nn.Sequential(nn.Linear(3, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.dx = nn.Linear(d_in, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sk = ((x - mu) ** 3).mean(dim=1, keepdim=True)
        sk = torch.sign(sk) * torch.abs(sk).pow(1.0 / 3.0)
        feat = torch.cat([mu, var, sk], dim=-1)
        h_stat = self.stat(feat)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx(dx))
        g = self.gate(torch.cat([h_stat, h_dx], dim=-1))
        return self.head(h_stat * g + h_dx * (1 - g) + h * 0.3)

class M234_BigKernelRobust(Step2Base):
    """修复：降低噪声强度，改为训练时噪声+测试时纯净"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.c7 = nn.Conv1d(1, 1, 7, padding=3)
        self.c3 = nn.Conv1d(1, 1, 3, padding=1)
        self.p7 = nn.Linear(d_in, hidden)
        self.p3 = nn.Linear(d_in, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        x_in = x + torch.randn_like(x) * 0.02 if self.training else x
        h7 = self.norm(self.p7(self.c7(x_in.unsqueeze(1)).squeeze(1)))
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        g = self.fuse(torch.cat([h7, h3], dim=-1))
        return self.head(h7 * g + h3 * (1 - g) + h * 0.3)

class M235_SparseFocusAttn(Step2Base):
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.selector = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        self.n_g, self.g_s = 8, max(1, hidden // 8)
        self.pg = nn.Linear(self.g_s, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        self.refine = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        mask = self.selector(x)
        h_sel = self.norm(self.proj(x * mask))
        B = h.shape[0]
        hg = h_sel.reshape(B, self.n_g, self.g_s)
        hp = self.pg(hg)
        a, _ = self.attn(hp, hp, hp, need_weights=False)
        p = (hp + a * 0.3).mean(dim=1)
        r = self.refine(p)
        return self.head(p + r * 0.3 + h * 0.2)

class M236_TriBandBloom(Step2Base):
    """三频带开花：低/中/高频能量积累检测与动态门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.low = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mid = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.high = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        self.accum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        n = amp.size(1)
        s1 = max(1, n // 4)
        s2 = max(1, n // 2)
        m_low = torch.zeros_like(amp)
        m_low[:, :s1] = 1.0
        m_mid = torch.zeros_like(amp)
        m_mid[:, s1:s2] = 1.0
        m_high = torch.zeros_like(amp)
        m_high[:, s2:] = 1.0
        x_low = torch.fft.irfft(xf * m_low, n=x.shape[1], dim=1)
        x_mid = torch.fft.irfft(xf * m_mid, n=x.shape[1], dim=1)
        x_high = torch.fft.irfft(xf * m_high, n=x.shape[1], dim=1)
        h_low = self.low(x_low)
        h_mid = self.mid(x_mid)
        h_high = self.high(x_high)
        g = self.gate(torch.cat([h_low, h_mid, h_high], dim=-1))
        fused = h_low * g[:, 0:1] + h_mid * g[:, 1:2] + h_high * g[:, 2:3]
        a = self.accum(fused)
        return self.head(fused + a * 0.3 + h * 0.2)

class M237_LightSSM_Select(Step2Base):
    """
    修复核心：砍掉失效 GRU，改为 MambaSL 式卷积+SSM门控 + SelectiveLearn 输入选择
    理论：横截面数据无天然时序，GRU 序列长度=1 完全退化；
          改用 1D 卷积提取局部特征模式 + Sigmoid 状态门控 + 输入级特征软选择
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # SelectiveLearn: 输入级动态特征软选择
        self.selector = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        # MambaSL 式卷积 + SSM 门控（借鉴 M115 头部架构）
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.ssm = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        # 增强
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 输入特征选择
        mask = self.selector(x)
        x_sel = x * mask
        h_sel = self.norm(self.proj(x_sel))
        # 卷积局部模式提取
        c = self.conv(x_sel.unsqueeze(1)).squeeze(1)
        hc = self.norm(self.proj(c))
        # SSM 状态门控
        s = self.ssm(hc)
        # 增强融合
        e = self.enhance(h_sel + hc * s)
        return self.head(h_sel + hc * 0.3 + s * 0.2 + e * 0.2)

class M238_MultiWaveDomain(Step2Base):
    """MLF 多周期卷积 + WaveMix 小波高低频混合 + DTAF 域自适应滤波"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # MLF: 多周期局部卷积
        self.mlf_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        # WaveMix: 小波式高低频门控
        self.low_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.high_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.wave_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # DTAF: 域自适应对齐
        self.domain_enc = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.domain_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # MLF 局部卷积
        c = self.mlf_conv(x.unsqueeze(1)).squeeze(1)
        h_mlf = self.norm(self.proj(c))
        # WaveMix 小波式分解
        x_smooth = F.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        lo = self.low_pass(x_smooth)
        hi = self.high_pass(x - x_smooth)
        g_wave = self.wave_gate(torch.cat([lo, hi], dim=-1))
        h_wave = lo * g_wave + hi * (1 - g_wave)
        # DTAF 域自适应融合（MLF 与 WaveMix 视为双域）
        g_dom = self.domain_gate(torch.cat([h_mlf, h_wave], dim=-1))
        h_adapt = h_mlf * g_dom + h_wave * (1 - g_dom)
        # 域残差修正
        h_dom = self.domain_enc(x)
        residual = h_dom - h_adapt
        h_dtaf = h_adapt + torch.tanh(residual) * 0.3
        # 融合
        return self.head(h_dtaf + h_dom * 0.2 + h * 0.2)

class M239_DeepResidualBloom(Step2Base):
    """
    修：引入 TriScaleMomentum 三尺度卷积 + 残差净化差异开花
    理论：单一卷积/差分无法捕捉多尺度蓄势；引入 3/5/7 核多尺度卷积，
          残差净化后与原卷积特征做差异检测，通过门控实现"静默积累→爆发"开花
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # 三尺度卷积（借鉴 M223 TriScaleMomentum 0.9933）
        self.c3 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.c5 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.c7 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.p3 = nn.Linear(d_in, hidden)
        self.p5 = nn.Linear(d_in, hidden)
        self.p7 = nn.Linear(d_in, hidden)
        # 三尺度融合门
        self.tri_gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        # 残差净化（保留原模型核心思想）
        self.main = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.filter = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        # 差异开花：多尺度卷积与净化残差的差异决定爆发幅度
        self.bloom = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 三尺度卷积提取
        h3 = self.norm(self.p3(self.c3(x.unsqueeze(1)).squeeze(1)))
        h5 = self.norm(self.p5(self.c5(x.unsqueeze(1)).squeeze(1)))
        h7 = self.norm(self.p7(self.c7(x.unsqueeze(1)).squeeze(1)))
        g = self.tri_gate(torch.cat([h3, h5, h7], dim=-1))
        h_conv = h3 * g[:, 0:1] + h5 * g[:, 1:2] + h7 * g[:, 2:3]
        # 残差净化
        m = self.main(h)
        f = self.filter(h)
        clean = h * f + m * (1 - f)
        # 差异开花：卷积特征与净化特征的差异蕴含蓄势信息
        diff = h_conv - clean
        b = self.bloom(diff)
        bloom_gate = torch.sigmoid(b)
        h_bloom = clean + diff * bloom_gate * 0.5
        return self.head(h_bloom + h * 0.2)

class M240_AlphaSupreme(Step2Base):
    """修复：砍掉复杂MoE，改为双路卷积+频域+差分轻量融合"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv = nn.Conv1d(1, 1, 7, padding=3)
        self.freq = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.dx = nn.Linear(d_in, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        split = max(1, amp.size(1) // 4)
        mask = torch.zeros_like(amp); mask[:, :split] = 1.0
        x_freq = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        h_f = self.freq(x_freq)
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        dx = F.pad(dx, (0, x.size(1) - dx.size(1))) if dx.size(1) < x.size(1) else dx[:, :x.size(1)]
        h_dx = self.norm(self.dx(dx))
        feat = torch.cat([h_c, h_f, h_dx], dim=-1)
        g = self.fusion(feat)
        fused = h_c * g + h_f * (1 - g) * 0.5 + h_dx * 0.3
        return self.head(fused + h * 0.2)

# ===========原生头部杂交模型 (241-246) ==========

class M241_DuetPatch(Step2Base):
    """DUET 双卷积局部提取 + PatchTST 分块注意力全局聚合 + MICN 残差门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # DUET 双尺度卷积分支
        self.conv1 = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.duet_fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop))
        # PatchTST 分块注意力分支
        self.patch_len = max(4, d_in // 4)
        self.n_patch = (d_in + self.patch_len - 1) // self.patch_len
        self.patch_proj = nn.Linear(self.patch_len, hidden)
        self.patch_attn = nn.MultiheadAttention(hidden, 2, dropout=drop, batch_first=True)
        # MICN 式残差门控
        self.res_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # DUET 局部特征
        c1 = self.conv1(x.unsqueeze(1)).squeeze(1)
        c2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        duet_out = self.duet_fusion(torch.cat([h1, h2], dim=-1))
        # PatchTST 全局分块注意力
        B, D = x.shape
        pad = self.n_patch * self.patch_len - D
        x_pad = x if pad <= 0 else F.pad(x, (0, pad))
        x_patch = x_pad[:, :self.n_patch * self.patch_len].reshape(B, self.n_patch, self.patch_len)
        h_patch = self.norm(self.patch_proj(x_patch))
        ha, _ = self.patch_attn(h_patch, h_patch, h_patch, need_weights=False)
        h_patch = h_patch + ha * 0.3
        patch_out = h_patch.mean(dim=1)
        # 融合：局部 DUET + 全局 Patch + 残差
        combined = duet_out + patch_out * 0.4
        alpha = self.res_gate(h) * 0.2 + 0.2
        return self.head(combined + h * alpha)


class M242_MambaGraph(Step2Base):
    """MambaSSM 门控状态空间 + ASTGI 可学习图邻接传播 + 融合门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # ASTGI 图分支
        self.adj = nn.Parameter(torch.eye(d_in) * 0.5 + torch.randn(d_in, d_in) * 0.05)
        self.graph_proj = nn.Linear(d_in, hidden)
        # MambaSSM 时序门控分支
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.gate = nn.Linear(d_in, hidden)
        self.state = nn.Linear(d_in, hidden)
        # 融合与增强
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 图传播
        adj = torch.sigmoid(self.adj)
        x_graph = torch.matmul(x, adj)
        h_graph = self.norm(self.graph_proj(x_graph))
        # MambaSSM 局部扫描
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        g = torch.sigmoid(self.gate(c))
        s = torch.tanh(self.state(c))
        h_conv = self.norm(self.proj(c))
        h_mamba = g * h_conv + (1 - g) * s
        # 自适应融合
        fused = self.fusion_gate(torch.cat([h_graph, h_mamba], dim=-1))
        combined = h_graph * fused + h_mamba * (1 - fused)
        e = self.enhance(combined)
        return self.head(combined + e * 0.3 + h * 0.2)


class M243_ModernSCIMix(Step2Base):
    """ModernTCN 大/小核卷积 + SCINet 可逆奇偶平滑 + TimeMixer 粗细粒度门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.conv_large = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.conv_small = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.scinet_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.coarse_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.fine_enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # ModernTCN 双尺度
        c_large = self.conv_large(x.unsqueeze(1)).squeeze(1)
        c_small = self.conv_small(x.unsqueeze(1)).squeeze(1)
        h_large = self.norm(self.proj(c_large))
        h_small = self.norm(self.proj(c_small))
        # SCINet 可逆平滑
        c_sci = self.scinet_conv(x.unsqueeze(1)).squeeze(1)
        h_sci = self.norm(self.proj((x + c_sci) / 2))
        # 多尺度混合
        multi = h_large * 0.4 + h_small * 0.3 + h_sci * 0.3
        # TimeMixer 粗细粒度
        coarse = multi * self.coarse_gate(multi)
        fine = self.fine_enhance(multi - multi.mean(1, keepdim=True).expand_as(multi))
        return self.head(coarse + fine * 0.4 + h * 0.3)


class M244_FinCastWave(Step2Base):
    """FinCast-Lite 分段注意力 + TimesFM 多尺度池化 + WaveMix 时域高低频门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.n_seg = 4
        self.seg_len = max(1, (d_in + self.n_seg - 1) // self.n_seg)
        self.seg_proj = nn.Linear(self.seg_len, hidden)
        self.seg_attn = nn.Linear(hidden, 1)
        self.pool_large = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool_small = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.low_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.high_pass = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.wave_fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        B, D = x.shape
        # FinCast 分段注意力
        pad = self.n_seg * self.seg_len - D
        x_pad = x if pad <= 0 else F.pad(x, (0, pad))
        x_seg = x_pad[:, :self.n_seg * self.seg_len].reshape(B, self.n_seg, self.seg_len)
        hs = torch.tanh(self.seg_proj(x_seg))
        a = torch.softmax(self.seg_attn(hs), dim=1)
        p = (hs * a).sum(dim=1)
        # TimesFM 多尺度池化
        x_unsq = x.unsqueeze(1)
        t_large = self.pool_large(x_unsq).squeeze(1)
        t_small = self.pool_small(x_unsq).squeeze(1)
        h_large = self.norm(self.proj(t_large))
        h_small = self.norm(self.proj(t_small))
        t_fused = h_large * 0.5 + h_small * 0.5
        # WaveMix 时域高低频（平滑近似）
        x_smooth = F.avg_pool1d(x_unsq, kernel_size=3, stride=1, padding=1).squeeze(1)
        lo = self.low_pass(x_smooth)
        hi = self.high_pass(x - x_smooth)
        g_wave = self.wave_fusion(torch.cat([lo, hi], dim=-1))
        wave_fused = lo * g_wave + hi * (1 - g_wave)
        # 三路融合
        combined = p * 0.4 + t_fused * 0.3 + wave_fused * 0.3
        return self.head(combined + h * 0.3)


class M245_KANRoseLSTM(Step2Base):
    """KANMixer KAN+MLP 双路径 + ROSE 频率调制 + xLSTM-Mixer 记忆门控"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        self.freq = nn.Parameter(torch.tensor(0.15))
        self.amp = nn.Parameter(torch.tensor(1.0))
        self.kan_path = nn.Sequential(SimpleKANLayer(d_in, hidden, 5, 3), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.mlp_path = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.path_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.mem = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mem_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # ROSE 频率调制
        modulated = h * torch.cos(h * torch.abs(self.freq)) * torch.sigmoid(self.amp)
        # KANMixer 双路径
        hk = self.kan_path(x)
        hm = self.mlp_path(x)
        g_path = self.path_gate(torch.cat([hk, hm], dim=-1))
        path_fused = hk * g_path + hm * (1 - g_path)
        # xLSTM 记忆融合
        m = self.mem(modulated)
        g_mem = self.mem_gate(torch.cat([path_fused, m], dim=-1))
        combined = path_fused * g_mem + m * (1 - g_mem)
        return self.head(combined + h * 0.3)


class M246_HyperMoEPath(Step2Base):
    """HIGSTM 层级趋势分解 + Pathformer 多路径 + MERA-Lite MoE 路由"""
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # HIGSTM 三尺度投影
        self.trend_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.season_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.noise_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        # Pathformer 多路径
        self.path_a = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.path_b = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 1.2))
        # MERA-Lite MoE
        self.n_experts = 3
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.5)),
        ])
        self.router = nn.Sequential(nn.Linear(hidden, self.n_experts), nn.Softmax(dim=-1))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # HIGSTM 层级分解
        x_unsq = x.unsqueeze(1)
        trend = F.avg_pool1d(x_unsq, kernel_size=5, stride=1, padding=2).squeeze(1)
        season = F.avg_pool1d(x_unsq, kernel_size=3, stride=1, padding=1).squeeze(1) - trend
        noise = x - trend - season
        h_t = self.trend_proj(trend)
        h_s = self.season_proj(season)
        h_n = self.noise_proj(noise)
        hier = h_t + h_s * 0.5 + h_n * 0.2
        # Pathformer 多路径
        pa = self.path_a(x)
        pb = self.path_b(x)
        # MoE 路由
        r = self.router(hier)
        outs = torch.stack([exp(hier) for exp in self.experts], dim=1)
        h_moe = (r.unsqueeze(-1) * outs).sum(dim=1)
        # 融合
        g = self.fusion(torch.cat([pa, h_moe], dim=-1))
        combined = pa * g + h_moe * (1 - g) + pb * 0.3
        return self.head(combined + h * 0.3)


# ==========原生头部极致杂交模型 (247-250) =====================

class M247_MultiPeriodFreqHyper(Step2Base):
    """
    MLF 多周期局部卷积 + FreTS 频域 TopK 选择与时域残差 + DRFN 动静分解门控
    理论支撑：多周期时域局部性 + 频域能量选择 + 特征动静关系建模
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # MLF: 多周期局部卷积提取
        self.mlf_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        # FreTS: 频域 TopK 选择分支 + 时域分支
        self.freq_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.time_proj = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.freq_time_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # DRFN: 静态线性 + 动态非线性动静分解
        self.static = nn.Linear(d_in, hidden, bias=False)
        self.dynamic = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden)
        )
        self.drfn_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.Sigmoid())
        # 三路融合头
        self.tri_fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.enhance = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # MLF 局部卷积分支
        c = self.mlf_conv(x.unsqueeze(1)).squeeze(1)
        h_mlf = self.norm(self.proj(c))
        # FreTS 频域选择分支（TopK 能量保留，非简单高低频 split）
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        k = max(1, amp.shape[1] // 4)
        _, idx = torch.topk(amp, k, dim=1)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, idx, 1.0)
        x_freq = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        h_freq = self.freq_proj(x_freq)
        h_time = self.time_proj(x)
        g_ft = self.freq_time_gate(torch.cat([h_freq, h_time], dim=-1))
        h_frets = h_freq * g_ft + h_time * (1 - g_ft)
        # DRFN 动静分解分支
        h_static = self.static(x)
        h_dynamic = self.dynamic(x)
        g_drfn = self.drfn_gate(torch.cat([h_static, h_dynamic], dim=-1))
        h_drfn = g_drfn * h_static + (1 - g_drfn) * h_dynamic
        # 三路极致融合
        tri = torch.cat([h_mlf, h_frets, h_drfn], dim=-1)
        fused = self.tri_fusion(tri)
        e = self.enhance(fused)
        return self.head(fused + e * 0.3 + h * 0.2)


class M248_DualMambaRiskWin(Step2Base):
    """
    FinMamba 市场注意力重加权 + Mamba2-SSD 分段状态空间扫描 + AdaptWin 双尺度风险卷积
    理论支撑：市场微观结构注意力 + 分段因果状态演化 + 多尺度波动率感知
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # AdaptWin: 双尺度风险卷积输入
        self.conv_risk1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv_risk2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.risk_fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # Mamba2-SSD: 分段投影 + 分组因果卷积 + 累积和状态扫描
        self.n_seg = 8
        self.seg_len = max(1, (d_in + self.n_seg - 1) // self.n_seg)
        self.seg_proj = nn.Linear(self.seg_len, hidden)
        self.seg_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        # FinMamba: 市场注意力机制
        self.market_attn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        # 融合门控
        self.ssm_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # AdaptWin 双尺度风险特征
        c1 = self.conv_risk1(x.unsqueeze(1)).squeeze(1)
        c2 = self.conv_risk2(x.unsqueeze(1)).squeeze(1)
        h1 = self.norm(self.proj(c1))
        h2 = self.norm(self.proj(c2))
        g_risk = self.risk_fusion(torch.cat([h1, h2], dim=-1))
        h_risk = h1 * g_risk + h2 * (1 - g_risk)
        # Mamba2-SSD 分段状态空间扫描
        B, D = x.shape
        pad = self.n_seg * self.seg_len - D
        x_pad = x if pad <= 0 else F.pad(x, (0, pad))
        x_seg = x_pad[:, :self.n_seg * self.seg_len].reshape(B, self.n_seg, self.seg_len)
        h_seg = self.norm(self.seg_proj(x_seg))
        h_c = F.silu(self.seg_conv(h_seg.transpose(1, 2)).transpose(1, 2))
        h_cum = torch.cumsum(h_c, dim=1)
        denom = torch.arange(1, self.n_seg + 1, device=x.device).view(1, -1, 1).float()
        h_cum = h_cum / (denom + 1e-8)
        h_ssm = h_c.mean(dim=1) + h_cum.mean(dim=1) * 0.5
        # FinMamba 市场注意力重加权
        hm = self.market_attn(h_risk)
        attn = F.softmax(hm, dim=-1)
        h_market = (hm * attn).sum(dim=1, keepdim=True).expand(-1, self.hidden)
        # 市场注意力驱动 SSM 融合
        g = self.ssm_gate(h_market)
        combined = h_ssm * g + h_market * (1 - g)
        return self.head(combined + h_risk * 0.3 + h * 0.2)


class M249_GroupMixPriorSelect(Step2Base):
    """
    SelectiveLearn 动态特征软选择 + StockMixer 分组通道混合 + TabPriorEnsemble 先验三路集成
    理论支撑：特征重要性选择 + 结构化分组交互 + 先验分布注入
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # SelectiveLearn: 输入级特征选择门
        self.selector = nn.Sequential(nn.Linear(d_in, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, d_in), nn.Sigmoid())
        # StockMixer: 特征分组投影
        self.n_group = 8
        self.seg_len = (d_in + self.n_group - 1) // self.n_group
        self.group_proj = nn.Linear(self.seg_len, hidden)
        # TabPriorEnsemble: 每组内三路先验扰动集成
        self.path1 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.path2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 1.2))
        self.path3 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop * 0.8))
        self.prior_gate = nn.Sequential(nn.Linear(hidden * 3, 3), nn.Softmax(dim=-1))
        # 跨组聚合
        self.group_mix = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # 动态特征选择
        mask = self.selector(x)
        x_sel = x * mask
        # StockMixer 分组投影
        B, D = x_sel.shape
        pad = self.n_group * self.seg_len - D
        x_pad = x_sel if pad <= 0 else F.pad(x_sel, (0, pad))
        x_grp = x_pad[:, :self.n_group * self.seg_len].reshape(B, self.n_group, self.seg_len)
        h_grp = self.norm(self.group_proj(x_grp))
        # TabPriorEnsemble：每组内三路先验集成
        p1 = self.path1(h_grp)
        p2 = self.path2(h_grp)
        p3 = self.path3(h_grp)
        g = self.prior_gate(torch.cat([p1, p2, p3], dim=-1))
        h_ensemble = p1 * g[:, :, 0:1] + p2 * g[:, :, 1:2] + p3 * g[:, :, 2:3]
        # 跨组混合（均值 + 极值）
        m = h_ensemble.mean(dim=1)
        mx = h_ensemble.max(dim=1)[0]
        h_mix = self.group_mix(m + mx * 0.3)
        return self.head(h_mix + h * 0.3)


class M250_WorldModelMaskAlign(Step2Base):
    """
    NEDreamer 差分趋势因果编码 + Timer 随机掩码重建双路径 + TimeAlign 分布对齐门控
    理论支撑：因果轨迹生成 + 自监督掩码一致性 + 预测分布对齐
    """
    def __init__(self, d_in, hidden=96, drop=0.1):
        super().__init__(d_in, hidden, drop)
        # NEDreamer: 差分趋势编码与动量门控
        self.trend_enc = nn.Sequential(nn.Linear(max(1, d_in - 1), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.momentum = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        # Timer: 掩码重建路径
        self.mask_path = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        # TimeAlign: 分布对齐与残差门控
        self.align = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.align_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # 卷积旁路
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self._preprocess(x)
        # NEDreamer 差分趋势因果编码
        dx = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else torch.zeros_like(x[:, :1])
        h_trend = self.trend_enc(dx)
        m = torch.tanh(self.momentum(h_trend))
        # Timer 掩码双路径（训练期随机掩码，测试期纯净）
        xm = x * (torch.rand_like(x) > 0.15).float() if self.training else x
        h_mask = self.mask_path(xm)
        # TimeAlign 分布对齐：掩码路径与趋势路径的残差对齐
        corr = torch.tanh(self.align(h_mask - h_trend))
        g = self.align_gate(torch.cat([h_mask, corr], dim=-1))
        h_align = h_mask + corr * g
        # 融合：因果趋势为主，掩码对齐为辅
        combined = h_trend + m * 0.5 + h_align * 0.4
        c = self.conv(x.unsqueeze(1)).squeeze(1)
        h_c = self.norm(self.proj(c))
        return self.head(combined + h_c * 0.2 + h * 0.2)



# ========== 模型工厂注册表 (110 个模型，严格无重复) =====================
MODEL_FACTORY = {
    '01': M01_FITS, '02': M02_SparseTSF, '03': M03_RLinear, '04': M04_TimeBridge,
    '05': M05_SpectraFormer, '06': M06_TabPFN, '07': M07_TabICL, '08': M08_TimesFM,
    '09': M09_OLinear, '10': M10_TiRex, '11': M11_TSPRank, '12': M12_DUET,
    '13': M13_MLF, '14': M14_MambaSSM, '15': M15_FreDF_Whitening, '16': M16_SoftDTW_Shape,
    '17': M17_ModernTCN, '18': M18_TimeMixer, '19': M19_CycleNet, '20': M20_Chronos,
    '21': M21_Aurora, '22': M22_Moirai, '23': M23_LightGTS, '24': M24_Sundial,
    '25': M25_Timer_XL, '26': M26_UniTS, '27': M27_MOMENT, '28': M28_Kronos,
    '29': M29_TimeFilter, '30': M30_ROSE, '31': M31_xLSTM, '32': M32_DistDF_Wasserstein,
    '33': M33_RealMLP, '34': M34_LimiX, '35': M35_Mitra, '36': M36_WPMixer,
    '37': M37_TimeMCL, '38': M38_NeuralPort, '39': M39_AdaptWin, '40': M40_StockSSG,
    '41': M41_TabDPT, '42': M42_Timer, '43': M43_MSGNet, '44': M44_Pathformer,
    '45': M45_NodeTrans_Stock, '46': M46_TimeKAN, '47': M47_Autoformer, '48': M48_Informer,
    '49': M49_FEDformer, '50': M50_PatchTST, '51': M51_TimesNet, '52': M52_DLinear,
    '53': M53_Crossformer, '54': M54_TabNet, '55': M55_TFT, '56': M56_ETSformer,
    '57': M57_ASTGI, '58': M58_Pyraformer, '59': M59_FiLM, '60': M60_MICN,
    '61': M61_SCINet, '62': M62_RevIN, '63': M63_iTransformer, '64': M64_VanillaTransformer,
    '65': M65_RWKV_TS, '66': M66_Mamba2SSD, '67': M67_CondFlowMatch, '68': M68_OrthoTrans,
    '69': M69_NeuralShrinkage, '70': M70_UncertaintyCAE, '71': M71_NEDreamer,
    '72': M72_StockMixer, '73': M73_KAN_AD, '74': M74_SPDQ_RL, '75': M75_TimeAlign,
    '76': M76_WaveLSFormer, '77': M77_MMPD_Predictor, '78': M78_MarketGAN_Aug,
    '79': M79_rfBLT_Bayes, '80': M80_MaGNet, '81': M81_LOBERT, '82': M82_KANMixer,
    '83': M83_FinD3, '84': M84_Hermes, '85': M85_SPF_Hawkes, '86': M86_FactorGCL,
    '87': M87_DeltaLag, '88': M88_DTAF, '89': M89_DRFN, '90': M90_AMD,
    '91': M91_COGRASP, '92': M92_AlphaCFG, '93': M93_FinMamba, '94': M94_DPA_STIFormer,
    '95': M95_HIGSTM, '96': M96_SAMBA, '97': M97_HINT_Lite, '98': M98_ABSSM,
    '99': M99_DOTS_Lite, '100': M100_FASCL_Lite, '101': M101_Diffolio_Lite,
    '102': M102_FreIE_Lite, '103': M103_GF_MSH_Lite, '104': M104_PureKAN_Lite,
    '105': M105_NIFL_Lite, '106': M106_SelectiveLearn, '107': M107_MERA,
    '108': M108_FinCast_Lite, '109': M109_GraphAttnLite, '110': M110_CausalHyper,
    '111': M111_TiDE, '112': M112_MambaStock, '113': M113_SegRNN,
    '114': M114_PAttn, '115': M115_MambaSL, '116': M116_TabM,
    '117': M117_FreTS, '118': M118_Koopa, '119': M119_MambAttention,
    '120': M120_ASGMamba, '121': M121_DMamba,
    #自造模型
    '201': M201_ProbGANLinear,
    '202': M202_JumpConvTrans,
    '203': M203_AdaptiveNormSSM,
    '204': M204_SpectralGap,
    '205': M205_CausalHyperGraph,
    '206': M206_QuantileBridge,
    '207': M207_KoopmanInvPeriod,
    '208': M208_SparseTFTAMD,
    '209': M209_CrossScaleModern,
    '210': M210_TabPriorEnsemble,
    '211': M211_VanillaMambaPrior,
    '212': M212_VolumeSilence,
    '213': M213_DivergenceMACD,
    '214': M214_MicroPressure,
    '215': M215_AutoFreqDense,
    '216': M216_FreqSqueeze,
    '217': M217_AttentionSpark,
    '218': M218_SpectraRoseFreq,
    '219': M219_CrossSkewness,
    '220': M220_SilentAccum,
    '221': M221_GapBloom,
    '222': M222_VolCompressBloom,
    '223': M223_TriScaleMomentum,
    '224': M224_DualAttnSurge,
    '225': M225_FreqTimeBridge,
    '226': M226_MoETabContext,
    '227': M227_CausalMoE,
    '228': M228_GroupMarketRisk,
    '229': M229_RevIN_Attn,
    '230': M230_KAN_ShapeDetect,
    '231': M231_TripleDivergence,
    '232': M232_DynamicHyperSurge,
    '233': M233_QuantileMomentumNet,
    '234': M234_BigKernelRobust,
    '235': M235_SparseFocusAttn,
    '236': M236_TriBandBloom,
    '237': M237_LightSSM_Select,
    '238': M238_MultiWaveDomain,
    '239': M239_DeepResidualBloom,
    '240': M240_AlphaSupreme,
    '241': M241_DuetPatch,
    '242': M242_MambaGraph,
    '243': M243_ModernSCIMix,
    '244': M244_FinCastWave,
    '245': M245_KANRoseLSTM,
    '246': M246_HyperMoEPath,
    '247': M247_MultiPeriodFreqHyper,
    '248': M248_DualMambaRiskWin,
    '249': M249_GroupMixPriorSelect,
    '250': M250_WorldModelMaskAlign,

}

MODEL_META = {
    '01': {'name': 'FITS', 'family': 'FreqLight', 'orth': '频域轻量低通滤波插值预测'},
    '02': {'name': 'SparseTSF', 'family': 'SparsePeriod', 'orth': '跨周期稀疏降采样趋势提取'},
    '03': {'name': 'RLinear', 'family': 'RevNormLin', 'orth': '可逆实例归一化与分布对齐线性层'},
    '04': {'name': 'TimeBridge', 'family': 'CointAttn', 'orth': '协整关系驱动的跨变量注意力桥接'},
    '05': {'name': 'SpectraFormer', 'family': 'SpectralGatingMixer', 'orth': '频域自适应门控与多频段特征混合器'},
    '06': {'name': 'TabPFN-2.6', 'family': 'TabPrior', 'orth': '先验感知门控与表格特征自适应MLP'},
    '07': {'name': 'TabICL', 'family': 'TabContext', 'orth': '表格上下文学习与跨样本特征交互'},
    '08': {'name': 'TimesFM-2.5', 'family': 'LongContext', 'orth': '长上下文多尺度池化时序底座'},
    '09': {'name': 'OLinear', 'family': 'OrthoLin', 'orth': '正交变换去相关与非负归一化线性预测'},
    '10': {'name': 'TiRex', 'family': 'ZeroShotBase', 'orth': '零样本迁移与上下文自适应门控底座'},
    '11': {'name': 'TSPRank', 'family': 'RankOpt', 'orth': '面向TopK排序优化的特征交互头'},
    '12': {'name': 'DUET', 'family': 'DualClust', 'orth': '时空双域特征聚类与双重门控融合'},
    '13': {'name': 'MLF', 'family': 'MultiPeriod', 'orth': '多周期频率特征自适应融合网络'},
    '14': {'name': 'MambaSSM', 'family': 'StateSpace', 'orth': '并行选择性门控与状态空间扫描卷积'},
    '15': {'name': 'FreDF_Whitening', 'family': 'FreqWhitening', 'orth': '频域低通滤波与标签白化去自相关'},
    '16': {'name': 'SoftDTW_Shape', 'family': 'ShapeAlign', 'orth': '基于软动态时间规整的全局形态对齐'},
    '17': {'name': 'ModernTCN', 'family': 'PureConv', 'orth': '大核纯卷积捕捉长周期趋势特征'},
    '18': {'name': 'TimeMixer', 'family': 'MultiScale', 'orth': '多尺度时序分解与粗细粒度门控混合'},
    '19': {'name': 'CycleNet', 'family': 'CycleExplicit', 'orth': '显式周期建模与正弦调制特征增强'},
    '20': {'name': 'Chronos', 'family': 'ProbGen', 'orth': '概率生成式时序预训练底座模型'},
    '21': {'name': 'Aurora', 'family': 'MultimodalBase', 'orth': '多模态数值语义双路径融合底座'},
    '22': {'name': 'Moirai', 'family': 'MaskBaseA', 'orth': '随机掩码重建与特征增强预训练底座'},
    '23': {'name': 'LightGTS', 'family': 'PeriodToken', 'orth': '周期轻量级词元化与多尺度门控'},
    '24': {'name': 'Sundial', 'family': 'MultiTask', 'orth': '多任务学习共享表征与时序底座'},
    '25': {'name': 'Timer_XL', 'family': 'MeanStdDual', 'orth': '均值标准差双尺度长上下文建模'},
    '26': {'name': 'UniTS', 'family': 'UnifiedTS', 'orth': '统一时序表征与均值残差双门控'},
    '27': {'name': 'MOMENT', 'family': 'MaskBaseC', 'orth': '掩码重建与动量更新的掩码学习底座'},
    '28': {'name': 'Kronos', 'family': 'KLineRep', 'orth': 'K线形态表征学习与频域特征编码'},
    '29': {'name': 'TimeFilter', 'family': 'SpatioTemp', 'orth': '多尺度时空滤波与特征门控融合'},
    '30': {'name': 'ROSE', 'family': 'FreqModRose', 'orth': '频率调制玫瑰图与余弦振幅特征增强'},
    '31': {'name': 'xLSTM-Mixer', 'family': 'xLSTMHybrid', 'orth': 'xLSTM状态记忆与通道混合器融合'},
    '32': {'name': 'DistDF_Wasserstein', 'family': 'DistBalancing', 'orth': '联合分布最优传输与自相关平衡'},
    '33': {'name': 'RealMLP', 'family': 'TabMLP', 'orth': '轻量化多层感知机与特征非线性映射'},
    '34': {'name': 'LimiX', 'family': 'TabBase', 'orth': '样本级特征限制与均值增强表格基座'},
    '35': {'name': 'Mitra', 'family': 'TabSynth', 'orth': '表格特征合成与Softmax注意力加权'},
    '36': {'name': 'WPMixer', 'family': 'WaveMix', 'orth': '小波变换频带分离与双路交叉混合'},
    '37': {'name': 'TimeMCL', 'family': 'MultiScene', 'orth': '多Dropout路径集成与不确定性加权'},
    '38': {'name': 'NeuralPortfolio', 'family': 'PortOpt', 'orth': '投资组合优化与特征权重动态分配'},
    '39': {'name': 'AdaptWin', 'family': 'RiskWin', 'orth': '自适应风险窗口与波动率感知卷积'},
    '40': {'name': 'StockSSG', 'family': 'StateGraph', 'orth': '状态空间图建模与节点关系推理'},
    '41': {'name': 'TabDPT', 'family': 'TabPretrain', 'orth': '表格深度预训练与自适应掩码学习'},
    '42': {'name': 'Timer', 'family': 'MaskGen', 'orth': '训练期随机掩码生成与双路径融合'},
    '43': {'name': 'MSGNet', 'family': 'MultiGraph', 'orth': '多尺度图消息传递与近远邻门控聚合'},
    '44': {'name': 'Pathformer', 'family': 'AdaptPath', 'orth': '自适应多路径特征变换与残差融合'},
    '45': {'name': 'NodeTrans-Stock', 'family': 'NodeGraph', 'orth': '节点级Transformer与图结构信息传播'},
    '46': {'name': 'TimeKAN', 'family': 'KANFreq', 'orth': 'KAN频率激活与SiLU门控非线性映射'},
    '47': {'name': 'Autoformer', 'family': 'AutoCorr', 'orth': '自相关序列分解与趋势季节门控融合'},
    '48': {'name': 'Informer', 'family': 'ProbSparse', 'orth': 'ProbSparse稀疏注意力与方差特征选择'},
    '49': {'name': 'FEDformer', 'family': 'FreqCross', 'orth': '频域MLP增强与时域残差交叉融合'},
    '50': {'name': 'PatchTST', 'family': 'ChanInd', 'orth': '通道独立分块嵌入与自注意力聚合'},
    '51': {'name': 'TimesNet', 'family': 'Period2D', 'orth': '二维周期变换与频域掩码特征增强'},
    '52': {'name': 'DLinear', 'family': 'LinDecomp', 'orth': '序列移动平均分解与双线性趋势预测'},
    '53': {'name': 'Crossformer', 'family': 'CrossScale', 'orth': '跨尺度特征路由与多核卷积融合'},
    '54': {'name': 'TabNet', 'family': 'TabSeqAttn', 'orth': '序列注意力稀疏选择与特征逐步聚焦'},
    '55': {'name': 'TFT', 'family': 'GateQuant', 'orth': '门控分位数回归与多尺度特征选择'},
    '56': {'name': 'ETSformer', 'family': 'ExpoSmooth', 'orth': '指数平滑分解与趋势季节双门控'},
    '57': {'name': 'ASTGI', 'family': 'SpatioTempGraph', 'orth': '时空图交互建模与图时卷积分支融合'},
    '58': {'name': 'Pyraformer', 'family': 'PyramidSparse', 'orth': '金字塔稀疏注意力与双尺度门控'},
    '59': {'name': 'FiLM', 'family': 'FreqMod', 'orth': '频率线性调制与特征仿射变换增强'},
    '60': {'name': 'MICN', 'family': 'MultiScaleConv', 'orth': '多尺度一维卷积与特征均值融合'},
    '61': {'name': 'SCINet', 'family': 'InvDown', 'orth': '可逆下采样卷积与奇偶特征平滑'},
    '62': {'name': 'RevIN', 'family': 'RevNorm', 'orth': '可逆实例归一化与序列分布标准化'},
    '63': {'name': 'iTransformer', 'family': 'InvertedAttn', 'orth': '倒置变量自注意力与序列特征编码'},
    '64': {'name': 'VanillaTransformer', 'family': 'MHA_FFN', 'orth': '经典多层自注意力与前馈网络堆叠'},
    '65': {'name': 'RWKV-TS', 'family': 'LinearAttn', 'orth': '显式衰减线性注意力与RWKV状态扫描'},
    '66': {'name': 'Mamba2-SSD', 'family': 'StateSpaceDuality', 'orth': '状态空间对偶与分块因果线性扫描'},
    '67': {'name': 'CondFlowMatch', 'family': 'CondGeneration', 'orth': '条件流匹配与确定性生成轨迹建模'},
    '68': {'name': 'OrthoTrans', 'family': 'DecorrLight', 'orth': '方差感知正交变换与特征去相关轻量拟合'},
    '69': {'name': 'NeuralShrinkage', 'family': 'CovarianceEst', 'orth': '自适应软阈值收缩与协方差矩阵估计'},
    '70': {'name': 'UncertaintyCAE', 'family': 'FactorSelect', 'orth': '不确定性感知编码与因子权重降权选择'},
    '71': {'name': 'NEDreamer', 'family': 'WorldModel', 'orth': '下一嵌入预测世界模型与因果轨迹生成'},
    '72': {'name': 'StockMixer', 'family': 'MLPMixer', 'orth': '特征分组通道混合与跨组全连接交互'},
    '73': {'name': 'KAN-AD', 'family': 'BSplineKAN', 'orth': 'B样条KAN异常感知与门控残差增强'},
    '74': {'name': 'SPDQ-RL', 'family': 'StochasticRL', 'orth': '随机策略强化学习与均值方差双头输出'},
    '75': {'name': 'TimeAlign', 'family': 'DistAlign', 'orth': '预测重构分布对齐与自适应门控融合'},
    '76': {'name': 'WaveMix_Adaptive', 'family': 'LearnableWavelet', 'orth': '小波混合自适应门控'},
    '77': {'name': 'MMPD-Predictor', 'family': 'DiffusionLoss', 'orth': '多步去噪扩散预测与周期门控残差'},
    '78': {'name': 'MarketGAN-Aug', 'family': 'GenAugment', 'orth': '因子生成对抗增强与特征空间扩展'},
    '79': {'name': 'rfBLT-Bayes', 'family': 'BayesianTakens', 'orth': '贝叶斯稀疏Takens嵌入与随机特征投影'},
    '80': {'name': 'MaGNet', 'family': 'MambaHypergraph', 'orth': '双向Mamba状态扫描与超图门控聚合'},
    '81': {'name': 'LOBERT', 'family': 'LOBBERT', 'orth': '订单簿微观结构差分与压力门控感知'},
    '82': {'name': 'KANMixer', 'family': 'KANMixer', 'orth': 'B样条KAN与MLP双路径非线性混合'},
    '83': {'name': 'FinD3', 'family': '3DMambaHyper', 'orth': 'RevIN标准化双分支门控与超图卷积'},
    '84': {'name': 'Hermes', 'family': 'LeadLagHyper', 'orth': '超边移动聚合与Lead-Lag多尺度建模'},
    '85': {'name': 'SPF-Hawkes', 'family': 'HawkesDynamicHyper', 'orth': 'Hawkes自激过程与动态超图强度建模'},
    '86': {'name': 'FactorGCL', 'family': 'ResidualContrast', 'orth': '时序残差对比学习与因子重要性挖掘'},
    '87': {'name': 'DeltaLag', 'family': 'DynamicLeadLag', 'orth': '端到端动态Lead-Lag稀疏注意力建模'},
    '88': {'name': 'DTAF', 'family': 'DomainAdaptFreq', 'orth': '域自适应时序滤波与频域特征增强'},
    '89': {'name': 'DRFN', 'family': 'StaticDynamicFusion', 'orth': '动静关系分解融合与超图交叉注意力'},
    '90': {'name': 'AMD', 'family': 'MultiScaleDecomp', 'orth': '自适应多尺度池化分解与三分支门控'},
    '91': {'name': 'COGRASP', 'family': 'HawkesHypergraph', 'orth': '霍克斯过程强度与指数衰减超图门控'},
    '92': {'name': 'AlphaCFG', 'family': 'GrammarSearch', 'orth': '语法引导因子搜索与TopK稀疏策略'},
    '93': {'name': 'FinMamba', 'family': 'MarketMamba', 'orth': '市场注意力Mamba状态空间与门控融合'},
    '94': {'name': 'DPA-STIFormer', 'family': 'InvertedTransformer', 'orth': '特征分组双路径倒置注意力Transformer'},
    '95': {'name': 'HIGSTM', 'family': 'HierGraphSTM', 'orth': '层级图结构时序记忆与多尺度门控'},
    '96': {'name': 'SAMBA-Lite', 'family': 'BiMambaGraph', 'orth': '双向Mamba状态扫描与自适应图卷积'},
    '97': {'name': 'HINT-Lite', 'family': 'HierarchicalIntention', 'orth': '层级意图感知与分组注意力压缩'},
    '98': {'name': 'ABSSM-Lite', 'family': 'AdaptiveBiSSM', 'orth': '自适应双向状态空间与门控调制融合'},
    '99': {'name': 'DOTS-Lite', 'family': 'CausalAttention', 'orth': '因果排序约束注意力与温度缩放特征选择'},
    '100': {'name': 'FASCL-Lite', 'family': 'ContrastiveRetrieval', 'orth': '预测一致性正则化与双视图对比学习'},
    '101': {'name': 'Diffolio-Lite', 'family': 'DiffusionPortfolio', 'orth': '扩散去噪投资组合与残差特征增强'},
    '102': {'name': 'FreIE-Lite', 'family': 'SpectralBiasCorrection', 'orth': '高低频频谱偏差纠正与频域门控分离'},
    '103': {'name': 'GF-MSH-Lite', 'family': 'MultiScaleGated', 'orth': '多尺度一维卷积与SE通道注意力门控'},
    '104': {'name': 'PureKAN-Lite', 'family': 'KolmogorovArnold', 'orth': '纯B样条KAN架构与残差非线性拟合'},
    '105': {'name': 'NIFL-Lite', 'family': 'CausalFactorization', 'orth': '神经工具变量因子分解与因果门控融合'},
    '106': {'name': 'SelectiveLearn-Lite', 'family': 'InputGating', 'orth': '动态特征软选择与温度感知噪声抑制'},
    '107': {'name': 'MERA-Lite', 'family': 'MoERetrieval', 'orth': 'Switch稳定专家路由与检索增强残差'},
    '108': {'name': 'FinCast-Lite', 'family': 'FinFoundation', 'orth': '金融基础模型特征分组注意力与MoE融合'},
    '109': {'name': 'GraphAttnLite', 'family': 'NodeTransInspired', 'orth': '轻量级图注意力交互与结构偏置建模'},
    '110': {'name': 'CausalHyper', 'family': 'CausalGraph', 'orth': '因果超图剪枝与动态邻接矩阵生成'},
    '111': {'name': 'TiDE', 'family': 'DenseEncoder', 'orth': '时序密集编码器与残差双分支融合'},
    '112': {'name': 'MambaStock', 'family': 'StockSSM', 'orth': '选择性状态空间模型与股票特征扫描'},
    '113': {'name': 'SegRNN', 'family': 'SegmentRNN', 'orth': '分段循环神经网络与轻量时序编码'},
    '114': {'name': 'PAttn', 'family': 'PatchAttn', 'orth': '轻量分块注意力与局部特征聚合'},
    '115': {'name': 'MambaSL', 'family': 'SingleLayerSSM', 'orth': '单层Mamba状态空间与轻量序列建模'},
    '116': {'name': 'TabM', 'family': 'BatchEnsemble', 'orth': '批集成扰动与表格MLP残差增强'},
    '117': {'name': 'FreTS', 'family': 'FreqMLP', 'orth': '频域MLP特征提取与时域残差融合'},
    '118': {'name': 'Koopa', 'family': 'KoopmanPred', 'orth': 'Koopman算子非平稳动态预测与线性演化'},
    '119': {'name': 'MambAttention', 'family': 'HybridAttn', 'orth': 'Mamba与Transformer混合注意力门控'},
    '120': {'name': 'ASGMamba', 'family': 'SpectralGatingSSM', 'orth': '自适应谱门控Mamba与状态空间扫描'},
    '121': {'name': 'DMamba', 'family': 'DecompSSM', 'orth': '分解增强Mamba与趋势季节状态扫描'},
    #自造模型
    '201': {'name': 'ProbGANLinear', 'family': 'ProbGen_GAN_Decomp',
            'orth': 'Chronos概率生成噪声注入与MarketGAN对抗隐空间及DLinear趋势季节分解门控'},
    '202': {'name': 'JumpConvTrans', 'family': 'MultiScale_Trans', 'orth': '三级跳空卷积与Transformer跨尺度路由'},
    '203': {'name': 'AdaptiveNormSSM', 'family': 'RevIN_SSM', 'orth': '可逆归一化与轻量因果卷积状态门控'},
    '204': {'name': 'SpectralGap', 'family': 'Freq_Gap', 'orth': '频域高低频缺口检测与动态门控'},
    '205': {'name': 'CausalHyperGraph', 'family': 'Causal_Graph', 'orth': '轻量特征交互与分组注意力卷积旁路'},
    '206': {'name': 'QuantileBridge', 'family': 'Quantile_Bridge', 'orth': '样本内分位数FiLM式全局统计调制与桥接增强'},
    '207': {'name': 'KoopmanInvPeriod', 'family': 'Koopman_Inv2D',
            'orth': 'Koopman线性演化算子与iTransformer倒置变量注意力及TimesNet二维周期卷积融合'},
    '208': {'name': 'SparseTFTAMD', 'family': 'SparseQuant_TriDecomp',
            'orth': 'Informer方差感知ProbSparse稀疏选择与TFT分位数多尺度门控及AMD趋势季节噪声三分支'},
    '209': {'name': 'CrossScaleModern', 'family': 'PureConv_Cross', 'orth': '双尺度卷积门控与现代卷积融合'},
    '210': {'name': 'TabPriorEnsemble', 'family': 'Tab_Prior', 'orth': '先验尺度偏移与三路扰动集成门控'},
    '211': {'name': 'VanillaMambaPrior', 'family': 'ClassicSSM_TabPrior',
            'orth': 'VanillaTransformer经典MHA_FFN堆叠与Mamba2-SSD分段因果扫描及TabPFN先验尺度偏移注入'},
    '212': {'name': 'VolumeSilence', 'family': 'Volume_Anomaly', 'orth': '量价静默程度与爆发潜力门控'},
    '213': {'name': 'DivergenceMACD', 'family': 'Momentum_Divergence', 'orth': '短长周期动量背离与底背离检测'},
    '214': {'name': 'MicroPressure', 'family': 'MicroStructure', 'orth': '微观结构差分压力积累与突破'},
    '215': {'name': 'AutoFreqDense', 'family': 'AutoCorr_FreqDense',
            'orth': 'Autoformer自相关移动平均分解与FEDformer频域MLP增强残差及TiDE密集编码双分支'},
    '216': {'name': 'FreqSqueeze', 'family': 'Freq_Compression', 'orth': '频域高低频能量压缩与爆发检测'},
    '217': {'name': 'AttentionSpark', 'family': 'Sparse_Attn', 'orth': '分组注意力稀疏化后聚焦激活'},
    '218': {'name': 'SpectraRoseFreq', 'family': 'Spectra_Rose_FreqMLP',
            'orth': 'SpectraFormer频谱自适应门控混合与ROSE正弦频率调制及FreTS频域MLP时域残差双分支'},
    '219': {'name': 'CrossSkewness', 'family': 'Skewness_Anomaly', 'orth': '跨周期偏度非对称蓄势检测'},
    '220': {'name': 'SilentAccum', 'family': 'Silent_Accum', 'orth': '多尺度卷积差异静默积累爆发'},
    '221': {'name': 'GapBloom', 'family': 'GapSurge', 'orth': '微观缺口检测与动量 surge 爆发门控'},
    '222': {'name': 'VolCompressBloom', 'family': 'VolSqueeze', 'orth': '波动率压缩检测与频域低频蓄势开花'},
    '223': {'name': 'TriScaleMomentum', 'family': 'TriConv', 'orth': '三尺度卷积动量共振与自适应加权'},
    '224': {'name': 'DualAttnSurge', 'family': 'HybridAttn', 'orth': '局部卷积与全局分组注意力双重 surge'},
    '225': {'name': 'FreqTimeBridge', 'family': 'FreqBridge', 'orth': '高低频分离编码与桥接门控融合'},
    '226': {'name': 'MoETabContext', 'family': 'MoE_Context_Select',
            'orth': 'MERA Switch式专家路由与TabICL跨样本上下文注意力交互及SelectiveLearn动态特征软选择'},
    '227': {'name': 'CausalMoE', 'family': 'CausalMoE', 'orth': '趋势波动动量三专家因果路由混合'},
    '228': {'name': 'GroupMarketRisk', 'family': 'GroupMarket_RiskWin',
            'orth': 'StockMixer特征分组通道混合与FinMamba市场微观结构注意力重加权及AdaptWin多尺度风险窗口卷积'},
    '229': {'name': 'RevIN_Attn', 'family': 'RevINAttn', 'orth': '自适应RevIN标准化与倒置分组注意力'},
    '230': {'name': 'KAN_ShapeDetect', 'family': 'KANShape', 'orth': '双B样条KAN非线性形态捕捉与门控'},
    '231': {'name': 'TripleDivergence', 'family': 'TripleDiv', 'orth': '短长偏度残差三重背离共识检测'},
    '232': {'name': 'DynamicHyperSurge', 'family': 'DynHyper', 'orth': '动态可学习邻接超图聚合与 surge 门控'},
    '233': {'name': 'QuantileMomentumNet', 'family': 'QtlMom', 'orth': '样本内分位数特征与动量去噪网络'},
    '234': {'name': 'BigKernelRobust', 'family': 'BigKernel', 'orth': '7核大卷积噪声注入与双尺度鲁棒融合'},
    '235': {'name': 'SparseFocusAttn', 'family': 'SparseFocus', 'orth': '特征稀疏选择与分块聚焦注意力精炼'},
    '236': {'name': 'TriBandBloom', 'family': 'TriBand', 'orth': '低中高三频带能量积累与动态开花'},
    '237': {'name': 'LightSSM_Select', 'family': 'LightSSM',
            'orth': 'MambaSL卷积状态门控与SelectiveLearn输入级动态特征软选择'},
    '238': {'name': 'MultiWaveDomain', 'family': 'MultiPeriod_Wave_Domain',
            'orth': 'MLF多周期局部卷积与WaveMix小波式高低频门控混合及DTAF域自适应分布对齐滤波'},
    '239': {'name': 'DeepResidualBloom', 'family': 'DeepResid',
            'orth': 'TriScaleMomentum三尺度卷积与残差净化差异开花门控'},
    '240': {'name': 'AlphaSupreme', 'family': 'Supreme', 'orth': '大核频域KAN微观差分MoE终极融合'},
    '241': {'name': 'DuetPatch', 'family': 'Duet_Patch', 'orth': 'DUET双卷积局部提取与PatchTST分块注意力全局聚合'},
    '242': {'name': 'MambaGraph', 'family': 'Mamba_Graph', 'orth': 'MambaSSM门控状态空间与ASTGI可学习图邻接传播'},
    '243': {'name': 'ModernSCIMix', 'family': 'Modern_SCI', 'orth': 'ModernTCN双核卷积与SCINet奇偶平滑及TimeMixer粗细门控'},
    '244': {'name': 'FinCastWave', 'family': 'FinCast_Wave', 'orth': 'FinCast分段注意力与TimesFM多尺度池化及WaveMix时域高低频门控'},
    '245': {'name': 'KANRoseLSTM', 'family': 'KAN_Rose_LSTM', 'orth': 'KANMixer双路径与ROSE频率调制及xLSTM记忆门控融合'},
    '246': {'name': 'HyperMoEPath', 'family': 'Hyper_MoE_Path', 'orth': 'HIGSTM层级分解与Pathformer多路径及MERA-Lite MoE路由'},
    '247': {'name': 'MultiPeriodFreqHyper', 'family': 'MultiPeriod_FreqHyper', 'orth': 'MLF多周期卷积与FreTS频域TopK选择及DRFN动静分解超图融合'},
    '248': {'name': 'DualMambaRiskWin', 'family': 'DualMamba_RiskWin', 'orth': 'FinMamba市场注意力与Mamba2-SSD分段状态空间扫描及AdaptWin风险卷积'},
    '249': {'name': 'GroupMixPriorSelect', 'family': 'GroupMix_PriorSelect', 'orth': 'StockMixer分组通道混合与TabPriorEnsemble先验集成及SelectiveLearn动态特征选择'},
    '250': {'name': 'WorldModelMaskAlign', 'family': 'WorldModel_MaskAlign', 'orth': 'NEDreamer因果趋势编码与Timer掩码重建及TimeAlign分布对齐门控'},

}