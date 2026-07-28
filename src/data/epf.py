"""EPF five-market data module."""
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from pathlib import Path


MARKET_META = {
    "NP": {"region": "Nord Pool", "years": "2013-2018", "z_cols": ["Grid load forecast", "Wind power forecast"]},
    "PJM": {"region": "PJM/COMED", "years": "2013-2018", "z_cols": ["system load forecast", "zonal load forecast"]},
    "BE": {"region": "Belgium (Elia)", "years": "2011-2016", "z_cols": ["system load forecast", "generation forecast"]},
    "FR": {"region": "France (EPEX)", "years": "2011-2016", "z_cols": ["system load forecast", "generation forecast"]},
    "DE": {"region": "Germany (EPEX)", "years": "2012-2017", "z_cols": ["zonal load forecast", "wind power forecast"]},
}


class EPFDataModule:
    """EPF five-market data module. Protocol identical to original paper."""

    def __init__(self, market: str, lookback: int = 168, horizon: int = 24,
                 splits=(0.7, 0.1, 0.2), seed: int = 2021, data_dir: str = "./dataset/epf",
                 split_mode: str = "chronological"):
        assert market in MARKET_META, f"Unknown market: {market}"
        assert split_mode in ("chronological", "seeded_shuffle"), \
            f"Unknown split_mode: {split_mode}"
        self.market = market
        self.L = lookback
        self.H = horizon
        self.splits = splits
        self.seed = seed
        self.split_mode = split_mode
        self.data_dir = Path(data_dir)
        self.raw_df = None
        self.train_mean = None
        self.train_std = None
        self.windows = {}

    def load_raw(self) -> pd.DataFrame:
        path = self.data_dir / f"{self.market}.csv"
        df = pd.read_csv(path)
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        self.raw_df = df
        return df

    def _get_cols(self, df: pd.DataFrame):
        """Find price/OT column and exogenous columns."""
        cols = list(df.columns)
        # Price column is usually 'OT' or last numeric column
        price_col = "OT" if "OT" in cols else cols[-1]
        # Exogenous columns (all except date and price)
        z_cols = [c for c in cols if c not in ["date", price_col]]
        return price_col, z_cols

    def make_windows(self) -> dict:
        df = self.load_raw()
        price_col, z_cols = self._get_cols(df)

        # Extract series
        x = df[price_col].values.astype(np.float32)
        z_list = [df[c].values.astype(np.float32) for c in z_cols[:2]]  # max 2 exog

        # Pad with zeros if fewer than 2 exogenous
        while len(z_list) < 2:
            z_list.append(np.zeros_like(x))

        # Concatenate: [x; z1; z2] -> each window is (3*L,)
        full = np.stack([x, z_list[0], z_list[1]], axis=1)  # (T, 3)

        windows = {"train": [], "val": [], "test": [], "target": {"train": [], "val": [], "test": []},
                     "all_inp": [], "all_tgt": []}
        for i in range(len(full) - self.L - self.H + 1):
            inp = full[i:i + self.L].flatten()  # (3*L,)
            tgt = x[i + self.L:i + self.L + self.H]  # (H,)
            windows["all_inp"].append(inp)
            windows["all_tgt"].append(tgt)

        n = len(windows["all_inp"])
        n_train = int(n * self.splits[0])
        n_val = int(n * self.splits[1])

        idx = np.arange(n)
        if self.split_mode == "seeded_shuffle":
            # legacy behavior (reviewer-flagged leakage under overlapping windows)
            np.random.RandomState(self.seed).shuffle(idx)
        # "chronological": windows are generated in start-time order, so the
        # sequential index already gives a temporal 0.7/0.1/0.2 split (no seed).

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        for split, sidx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            windows[split] = torch.tensor(np.stack([windows["all_inp"][i] for i in sidx]), dtype=torch.float32)
            windows[f"{split}_tgt"] = torch.tensor(np.stack([windows["all_tgt"][i] for i in sidx]), dtype=torch.float32)

        del windows["all_inp"], windows["all_tgt"]
        self.windows = windows
        return windows

    def normalize(self):
        """z-score normalization using training statistics only."""
        if self.windows is None:
            self.make_windows()
        train_inp = self.windows["train"]
        self.train_mean = train_inp.mean(dim=0, keepdim=True)
        self.train_std = train_inp.std(dim=0, keepdim=True) + 1e-8
        for split in ["train", "val", "test"]:
            self.windows[split] = (self.windows[split] - self.train_mean) / self.train_std

    def get_mask(self, split: str) -> Tensor:
        """Missingness mask (default all ones)."""
        return torch.ones_like(self.windows[split])

    def stats(self) -> dict:
        """Dataset-level stats for probes."""
        x = self.raw_df[self._get_cols(self.raw_df)[0]].values.astype(np.float32)
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "skew": float(pd.Series(x).skew()),
            "kurt": float(pd.Series(x).kurtosis()),
            "length": len(x),
        }
