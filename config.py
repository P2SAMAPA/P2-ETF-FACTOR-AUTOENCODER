"""
Configuration for P2-ETF-FACTOR-AUTOENCODER engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-factor-autoencoder-results"

# --- Universe Definitions (mirroring master data exactly) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]

ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns (using only those available from 2008) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Autoencoder Parameters ---
LATENT_DIM = 3                      # Number of latent factors
HIDDEN_DIMS = [64, 32]              # Encoder layer sizes (reversed for decoder)
EPOCHS = 100                        # Training epochs
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# --- Signal Computation Parameters ---
FACTOR_MOMENTUM_WINDOW = 20         # Days for factor momentum
FACTOR_TREND_WINDOW = 10            # Days for factor trend (slope)
RESIDUAL_ALPHA_WINDOW = 10          # EWMA window for residual alpha
CROSS_SECTIONAL_MOMENTUM_WINDOW = 20  # Days for cross-sectional momentum
ANOMALY_LOOKBACK = 252              # Days for reconstruction error baseline

# --- Signal Weights (must sum to 1.0) ---
SIGNAL_WEIGHTS = {
    "factor_momentum": 0.25,
    "factor_trend": 0.20,
    "reconstruction_error": 0.10,
    "residual_alpha": 0.20,
    "cross_sectional_momentum": 0.25
}

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))  # 2008..2024
MIN_OBSERVATIONS = 252               # Minimum observations required (1 year)

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
