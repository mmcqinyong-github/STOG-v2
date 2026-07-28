"""Unified training engine."""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class UnifiedTrainer:
    """Fair training protocol identical to original paper."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.max_epochs = cfg.get("max_epochs", 10)
        self.patience = cfg.get("patience", 3)
        self.batch_size = cfg.get("batch_size", 256)
        self.lr = cfg.get("lr", 1e-4)
        self.wd = cfg.get("weight_decay", 0.0)
        self.dropout = cfg.get("dropout", 0.1)

    def train_expert(self, expert, datamodule, verbose=False):
        """Train a single expert. Returns dict with metrics."""
        expert = expert.to(self.device)
        train_inp = datamodule.windows["train"].to(self.device)
        train_tgt = datamodule.windows["train_tgt"].to(self.device)
        val_inp = datamodule.windows["val"].to(self.device)
        val_tgt = datamodule.windows["val_tgt"].to(self.device)
        test_inp = datamodule.windows["test"].to(self.device)
        test_tgt = datamodule.windows["test_tgt"].to(self.device)

        # Check target shape and adapt head if needed
        if train_tgt.dim() == 1:
            train_tgt = train_tgt.unsqueeze(-1)
            val_tgt = val_tgt.unsqueeze(-1)
            test_tgt = test_tgt.unsqueeze(-1)

        dataset = TensorDataset(train_inp, train_tgt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(expert.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.MSELoss()

        best_val_mse = float("inf")
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.max_epochs):
            expert.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = expert(xb)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # Validation
            expert.eval()
            with torch.no_grad():
                val_pred = expert(val_inp)
                if val_pred.dim() == 1:
                    val_pred = val_pred.unsqueeze(-1)
                val_mse = criterion(val_pred, val_tgt).item()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Test evaluation
        expert.eval()
        with torch.no_grad():
            test_pred = expert(test_inp)
            if test_pred.dim() == 1:
                test_pred = test_pred.unsqueeze(-1)
            test_mse = criterion(test_pred, test_tgt).item()
            test_mae = (test_pred - test_tgt).abs().mean().item()

        elapsed = time.time() - start_time

        return {
            "val_mse": best_val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "epochs": epoch + 1,
            "time_sec": elapsed,
        }

    def width_selection(self, expert_cls, dm, widths=(128, 256, 512)):
        """Select best hidden width on validation."""
        best_w = widths[0]
        best_mse = float("inf")
        for w in widths:
            expert = expert_cls(d_in=dm.windows["train"].shape[1], hidden=w, drop=self.dropout)
            res = self.train_expert(expert, dm)
            if res["val_mse"] < best_mse:
                best_mse = res["val_mse"]
                best_w = w
        return best_w
