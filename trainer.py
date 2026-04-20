"""
Main training script for Factor Autoencoder engine.
Computes global model, factor signals, and shrinking-window analysis.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats

import config
import data_manager
from autoencoder_model import AutoencoderTrainer
import push_results

def compute_factor_momentum(factor_returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute momentum for each factor."""
    if len(factor_returns) < window + 1:
        return np.zeros(factor_returns.shape[1])
    return (factor_returns[-1] - factor_returns[-window-1]) / (np.abs(factor_returns[-window-1]) + 1e-6)

def compute_residual_alpha(actual: pd.DataFrame, reconstructed: np.ndarray,
                           scaler, window: int = 10) -> pd.Series:
    """Compute EWMA-smoothed residuals for each ETF."""
    actual_np = scaler.transform(actual.values)
    residuals = actual_np - reconstructed
    residuals_df = pd.DataFrame(residuals, index=actual.index, columns=actual.columns)
    return residuals_df.ewm(span=window, min_periods=1).mean().iloc[-1]

def compute_reconstruction_anomaly(errors: np.ndarray, lookback: int = 252) -> float:
    """Compute anomaly score based on reconstruction error."""
    if len(errors) < lookback:
        return 0.0
    recent_errors = errors[-lookback:]
    median = np.median(recent_errors)
    mad = np.median(np.abs(recent_errors - median))
    if mad < 1e-6:
        return 0.0
    return -abs(errors[-1] - median) / mad

def compute_combined_signals(factor_returns: np.ndarray, exposures: dict,
                             reconstruction_anomaly: float, residual_alpha: pd.Series,
                             tickers: list, weights: dict) -> dict:
    """Compute combined score for each ETF."""
    fm = compute_factor_momentum(factor_returns, config.FACTOR_MOMENTUM_WINDOW)
    
    scores = {}
    for ticker in tickers:
        if ticker not in exposures or ticker not in residual_alpha.index:
            continue
        
        beta = exposures[ticker]
        fm_signal = np.dot(beta, fm)
        ra_signal = residual_alpha[ticker]
        re_signal = reconstruction_anomaly
        
        # Z-score normalization across tickers (done after all computed)
        scores[ticker] = {
            'factor_momentum_raw': fm_signal,
            'residual_alpha_raw': ra_signal,
            'reconstruction_error_raw': re_signal
        }
    
    # Normalize each component to z-scores
    if scores:
        fm_vals = np.array([s['factor_momentum_raw'] for s in scores.values()])
        ra_vals = np.array([s['residual_alpha_raw'] for s in scores.values()])
        
        fm_z = stats.zscore(fm_vals) if fm_vals.std() > 0 else np.zeros_like(fm_vals)
        ra_z = stats.zscore(ra_vals) if ra_vals.std() > 0 else np.zeros_like(ra_vals)
        re_z = reconstruction_anomaly  # Same for all ETFs
        
        for i, ticker in enumerate(scores.keys()):
            scores[ticker]['factor_momentum_z'] = fm_z[i]
            scores[ticker]['residual_alpha_z'] = ra_z[i]
            scores[ticker]['reconstruction_error_z'] = re_z
            
            total = (weights['factor_momentum'] * fm_z[i] +
                     weights['reconstruction_error'] * re_z +
                     weights['residual_alpha'] * ra_z[i])
            scores[ticker]['total_score'] = total
    
    return scores

def run_autoencoder_pipeline():
    print(f"=== P2-ETF-FACTOR-AUTOENCODER Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    
    # Combined universe for global training
    all_tickers = config.ALL_TICKERS
    full_features = data_manager.prepare_full_feature_matrix(df_master, all_tickers)
    print(f"Global training data: {len(full_features)} days, {len(full_features.columns)} features")
    
    # Train global model
    trainer = AutoencoderTrainer(
        latent_dim=config.LATENT_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        seed=config.RANDOM_SEED
    )
    print("\n--- Training Global Autoencoder ---")
    trainer.fit(full_features)
    
    # Extract global factors
    global_factors = trainer.transform(full_features)
    global_reconstructed, global_errors = trainer.reconstruct(full_features)
    global_exposures = trainer.get_etf_exposures(full_features)
    
    # Compute signals
    reconstruction_anomaly = compute_reconstruction_anomaly(global_errors, config.ANOMALY_LOOKBACK)
    residual_alpha = compute_residual_alpha(full_features, global_reconstructed, 
                                            trainer.scaler, config.RESIDUAL_ALPHA_WINDOW)
    
    all_results = {
        "global_model": {
            "factor_returns": global_factors.tolist(),
            "reconstruction_error_today": float(global_errors[-1]),
            "reconstruction_anomaly": reconstruction_anomaly,
            "factor_exposures": {k: v.tolist() for k, v in global_exposures.items()}
        }
    }
    
    # Compute scores per universe and get top picks
    global_top_picks = {}
    global_scores_all = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Computing signals for {universe_name} ---")
        available_tickers = [t for t in tickers if t in global_exposures]
        scores = compute_combined_signals(
            global_factors, global_exposures, reconstruction_anomaly,
            residual_alpha, available_tickers, config.SIGNAL_WEIGHTS
        )
        global_scores_all[universe_name] = scores
        
        if scores:
            top_ticker = max(scores, key=lambda t: scores[t]['total_score'])
            global_top_picks[universe_name] = {
                'ticker': top_ticker,
                'total_score': scores[top_ticker]['total_score'],
                'components': scores[top_ticker]
            }
            print(f"  Top pick: {top_ticker} (score: {scores[top_ticker]['total_score']:.3f})")
    
    all_results['global_model']['signals'] = global_scores_all
    all_results['global_model']['top_picks'] = global_top_picks
    
    # Shrinking windows
    print("\n--- Shrinking Windows ---")
    shrinking_results = {}
    
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        print(f"\n  Window: {window_label}")
        
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            print(f"    Skipping (less than 1 year of data)")
            continue
        
        window_features = data_manager.prepare_full_feature_matrix(df_window, all_tickers)
        if len(window_features) < config.MIN_OBSERVATIONS:
            continue
        
        window_trainer = AutoencoderTrainer(
            latent_dim=config.LATENT_DIM,
            hidden_dims=config.HIDDEN_DIMS,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            seed=config.RANDOM_SEED
        )
        window_trainer.fit(window_features)
        
        win_factors = window_trainer.transform(window_features)
        win_reconstructed, win_errors = window_trainer.reconstruct(window_features)
        win_exposures = window_trainer.get_etf_exposures(window_features)
        win_anomaly = compute_reconstruction_anomaly(win_errors, min(config.ANOMALY_LOOKBACK, len(win_errors)//2))
        win_residual = compute_residual_alpha(window_features, win_reconstructed,
                                              window_trainer.scaler, config.RESIDUAL_ALPHA_WINDOW)
        
        window_top_picks = {}
        for universe_name, tickers in config.UNIVERSES.items():
            available_tickers = [t for t in tickers if t in win_exposures]
            win_scores = compute_combined_signals(
                win_factors, win_exposures, win_anomaly,
                win_residual, available_tickers, config.SIGNAL_WEIGHTS
            )
            if win_scores:
                top_ticker = max(win_scores, key=lambda t: win_scores[t]['total_score'])
                window_top_picks[universe_name] = {
                    'ticker': top_ticker,
                    'total_score': win_scores[top_ticker]['total_score']
                }
        
        shrinking_results[window_label] = {
            'start_year': start_year,
            'start_date': start_date.isoformat(),
            'top_picks': window_top_picks,
            'n_observations': len(window_features)
        }
    
    all_results['shrinking_windows'] = shrinking_results
    
    # Add config and push
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "latent_dim": config.LATENT_DIM,
            "hidden_dims": config.HIDDEN_DIMS,
            "signal_weights": config.SIGNAL_WEIGHTS,
            "factor_momentum_window": config.FACTOR_MOMENTUM_WINDOW,
            "residual_alpha_window": config.RESIDUAL_ALPHA_WINDOW,
            "anomaly_lookback": config.ANOMALY_LOOKBACK
        },
        **all_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_autoencoder_pipeline()
