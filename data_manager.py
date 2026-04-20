"""
Data loading and preprocessing for Factor Autoencoder engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from master data.")
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Prepare a wide-format DataFrame of log returns with Date index.
    """
    available_tickers = [t for t in tickers if t in df_wide.columns]
    print(f"Found {len(available_tickers)} ticker columns out of {len(tickers)} expected.")
    
    df_long = pd.melt(
        df_wide,
        id_vars=['Date'],
        value_vars=available_tickers,
        var_name='ticker',
        value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    pivot_returns = df_long.pivot(index='Date', columns='ticker', values='log_return')
    return pivot_returns[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Extract macro columns and return as DataFrame with Date index.
    """
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def prepare_full_feature_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Combine ETF returns and macro features into a single wide DataFrame.
    """
    returns = prepare_returns_matrix(df_wide, tickers)
    macro = prepare_macro_features(df_wide)
    common_idx = returns.index.intersection(macro.index)
    combined = pd.concat([returns.loc[common_idx], macro.loc[common_idx]], axis=1)
    return combined.dropna()
