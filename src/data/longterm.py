"""Long-term benchmark data module (ETTh1, ETTm1, Weather, ECL, Traffic, Exchange)."""
import pandas as pd
import numpy as np
import torch
from pathlib import Path


DATASET_META = {
    "ETTh1": {"file": "ETT-small/ETTh1.csv", "V": 7, "date_col": "date"},
    "ETTm1": {"file": "ETT-small/ETTm1.csv", "V": 7, "date_col": "date"},
    "Weather": {"file": "weather/weather.csv", "V": 21, "date_col": "date"},
    "ECL": {"file": "electricity/electricity.csv", "V": 321, "date_col": "date"},
    "Traffic": {"file": "traffic/traffic.csv", "V": 862, "date_col": "date"},
    "Exchange": {"file": "exchange_rate/exchange_rate.csv", "V": 8, "date_col": "date"},
}


class LongTermDataModule:
    """Unified loader for 6 long-term benchmarks."""

    def __init__(self, name: str, lookback: int = 336, horizon: int = 96,
                 splits=(0.7, 0.1, 0.2), seed: int = 2021, data_dir: str = "./dataset/longterm"):
        assert name in DATASET_META, f"Unknown dataset: {name}"
        self.name = name
        self.L = lookback
        self.H = horizon
        self.splits = splits
        self.seed = seed
        self.data_dir = Path(data_dir)
        self.meta = DATASET_META[name]
        self.windows = {}
        self.train_mean = None
        self.train_std = None

    def load_raw(self) -> pd.DataFrame:
        path = self.data_dir / self.meta["file"]
        df = pd.read_csv(path)
        # Drop date column if present
        if self.meta["date_col"] in df.columns:
            df = df.drop(columns=[self.meta["date_col"]])
        return df.astype(np.float32)

    def make_windows(self, channel_mode: str = "flatten") -> dict:
        df = self.load_raw()
        data = df.values  # (T, V)
        T, V = data.shape

        windows = {"all_inp": [], "all_tgt": []}
        for i in range(T - self.L - self.H + 1):
            if channel_mode == "flatten":
                inp = data[i:i + self.L].flatten()  # (L*V,)
                tgt = data[i + self.L:i + self.L + self.H].flatten()  # (H*V,)
            else:
                inp = data[i:i + self.L]  # (L, V)
                tgt = data[i + self.L:i + self.L + self.H]  # (H, V)
            windows["all_inp"].append(inp)
            windows["all_tgt"].append(tgt)

        n = len(windows["all_inp"])
        n_train = int(n * self.splits[0])
        n_val = int(n * self.splits[1])

        idx = np.arange(n)
        np.random.RandomState(self.seed).shuffle(idx)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        dtype = torch.float32
        for split, sidx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            arr_inp = np.stack([windows["all_inp"][i] for i in sidx])
            arr_tgt = np.stack([windows["all_tgt"][i] for i in sidx])
            self.windows[split] = torch.tensor(arr_inp, dtype=dtype)
            self.windows[f"{split}_tgt"] = torch.tensor(arr_tgt, dtype=dtype)

        return self.windows

    def normalize(self):
        """z-score using training stats."""
        if not self.windows:
            self.make_windows()
        train_inp = self.windows["train"]
        self.train_mean = train_inp.mean(dim=0, keepdim=True)
        self.train_std = train_inp.std(dim=0, keepdim=True) + 1e-8
        for split in ["train", "val", "test"]:
            self.windows[split] = (self.windows[split] - self.train_mean) / self.train_std
