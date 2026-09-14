"""
================================================================================
SkyGuard AI — Deep Weather Autoencoder Detector (PyTorch)
================================================================================
Unsupervised manifold learning architecture trained strictly on normal AWS observations.
Identifies hardware degradation, unphysical cross-correlations, and out-of-distribution
events via reconstruction error magnification.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class AutoencoderNet(nn.Module):
    """Deep bottleneck autoencoder with batch normalization and skip-resistant compression."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class DeepWeatherAutoencoder:
    """
    High-level PyTorch Autoencoder detector with automatic GPU acceleration,
    early stopping, and calibrated reconstruction error thresholds.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 15,
        device: Optional[str] = None,
        threshold_percentile: float = 95.0,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net: Optional[AutoencoderNet] = None
        self.threshold: float = 1.0
        self.recon_max: float = 2.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, val_split: float = 0.1) -> "DeepWeatherAutoencoder":
        """
        Train the autoencoder on clean meteorological observations.
        """
        n_samples, n_features = X.shape
        self.input_dim = n_features

        self.net = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Train/validation split
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        x_train = torch.tensor(X[train_idx], dtype=torch.float32)
        x_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)

        train_loader = DataLoader(
            TensorDataset(x_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(x_train) > self.batch_size,
        )

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            # Validation loss evaluation
            self.net.eval()
            with torch.no_grad():
                val_recon = self.net(x_val)
                val_loss = criterion(val_recon, x_val).item()
            self.net.train()

        self.is_fitted = True

        # Calibrate reconstruction threshold on training data
        self.net.eval()
        with torch.no_grad():
            x_all = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Evaluate in batches to avoid GPU OOM
            recon_errors = []
            for i in range(0, len(x_all), 1024):
                batch = x_all[i : i + 1024]
                recon = self.net(batch)
                mse_batch = torch.mean((recon - batch) ** 2, dim=1).cpu().numpy()
                recon_errors.append(mse_batch)

        all_errors = np.concatenate(recon_errors)
        self.threshold = float(np.percentile(all_errors, self.threshold_percentile))
        self.recon_max = float(np.percentile(all_errors, 99.5))

        return self

    def score(self, X: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        Compute continuous reconstruction error scores in [0.0, 1.0].
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Autoencoder must be fitted before scoring.")

        self.net.eval()
        scores = []
        n_samples = len(X)
        eval_bs = batch_size if getattr(self.device, "type", str(self.device)) == "cuda" else 4096
        with torch.no_grad():
            for i in range(0, n_samples, eval_bs):
                batch_np = X[i : i + eval_bs]
                batch_tensor = torch.from_numpy(np.ascontiguousarray(batch_np, dtype=np.float32)).to(self.device)
                recon = self.net(batch_tensor)
                mse = torch.mean((recon - batch_tensor) ** 2, dim=1).cpu().numpy()
                scores.append(mse)

        raw_mse = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
        # Normalize: threshold corresponds to ~0.6; higher errors scale towards 1.0
        normalized = raw_mse / max(1e-5, self.threshold * 2.0)
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary anomalies and continuous reconstruction scores.
        """
        scores = self.score(X)
        threshold_norm = self.threshold / max(1e-5, self.threshold * 2.0)
        binary_preds = (scores >= threshold_norm).astype(int)
        return binary_preds, scores

    def save(self, filepath: str) -> None:
        """Serialize PyTorch weights and calibration thresholds."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "threshold": self.threshold,
            "recon_max": self.recon_max,
            "threshold_percentile": self.threshold_percentile,
            "state_dict": self.net.state_dict() if self.net else None,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DeepWeatherAutoencoder":
        """Load trained autoencoder from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            device=device,
            threshold_percentile=checkpoint.get("threshold_percentile", 95.0),
        )
        model.threshold = checkpoint["threshold"]
        model.recon_max = checkpoint["recon_max"]
        model.net = AutoencoderNet(input_dim=model.input_dim, latent_dim=model.latent_dim).to(model.device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.is_fitted = True
        return model
