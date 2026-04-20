"""
Factor Autoencoder model for unsupervised latent factor extraction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, List

class FactorAutoencoder(nn.Module):
    """Symmetric autoencoder with configurable hidden layers."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AutoencoderTrainer:
    """Trains a FactorAutoencoder and extracts latent factors."""
    
    def __init__(self, latent_dim: int = 3, hidden_dims: list = [64, 32],
                 epochs: int = 100, batch_size: int = 64, lr: float = 0.001,
                 seed: int = 42):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_etfs = None
        
    def fit(self, data: pd.DataFrame) -> Dict:
        """
        Train autoencoder on the provided data.
        Returns dictionary with model, scaler, and training history.
        """
        self.feature_names = data.columns.tolist()
        
        # Standardize
        X = self.scaler.fit_transform(data.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)
        
        # Train/validation split
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        input_dim = X.shape[1]
        self.model = FactorAutoencoder(input_dim, self.latent_dim, self.hidden_dims)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= train_size
            history['train_loss'].append(train_loss)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    reconstructed, _ = self.model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    val_loss += loss.item() * len(batch_x)
            val_loss /= val_size
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return {'model': self.model, 'scaler': self.scaler, 'history': history}
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Extract latent factors from data."""
        self.model.eval()
        X = self.scaler.transform(data.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            _, latent = self.model(X_tensor)
        return latent.numpy()
    
    def reconstruct(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct data and return errors."""
        self.model.eval()
        X = self.scaler.transform(data.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            reconstructed, latent = self.model(X_tensor)
        reconstructed_np = reconstructed.numpy()
        errors = np.mean((X - reconstructed_np) ** 2, axis=1)
        return reconstructed_np, errors
    
    def get_etf_exposures(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Estimate factor exposures (betas) for each ETF.
        Uses a linear regression of each ETF's returns on the latent factors.
        """
        latent_factors = self.transform(data)
        exposures = {}
        for i, ticker in enumerate(self.feature_names):
            y = data[ticker].values
            X = latent_factors
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            exposures[ticker] = beta
        return exposures
