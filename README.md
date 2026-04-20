# P2-ETF-FACTOR-AUTOENCODER

**Unsupervised Deep Learning for Latent Factor Extraction and ETF Ranking**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-FACTOR-AUTOENCODER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-FACTOR-AUTOENCODER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--factor--autoencoder--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-factor-autoencoder-results)

## Overview

`P2-ETF-FACTOR-AUTOENCODER` uses a deep autoencoder to extract **latent market factors** from ETF returns and macro data. It then combines three signals to rank ETFs:

- **Factor Momentum**: Recent strength of each latent factor
- **Reconstruction Anomaly**: How unusual today's market behavior is
- **Residual Alpha**: ETF-specific return not explained by factors

The engine outputs daily **top picks** for each universe and includes a shrinking‑window analysis to track how factor structure evolves over time.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Methodology

1. **Autoencoder Training**: Learns 3 latent factors that best reconstruct daily returns.
2. **Factor Momentum**: Computes 20‑day return of each factor; ETF projected return = Σ(exposure × factor momentum).
3. **Reconstruction Anomaly**: Measures deviation from normal market patterns.
4. **Residual Alpha**: EWMA‑smoothed difference between actual and reconstructed returns.
5. **Combined Score**: Weighted sum of the three normalized signals.

## File Structure
P2-ETF-FACTOR-AUTOENCODER/
├── config.py # Paths, universes, autoencoder parameters
├── data_manager.py # Data loading and preprocessing
├── autoencoder_model.py # Autoencoder architecture and training
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-FACTOR-AUTOENCODER.git
cd P2-ETF-FACTOR-AUTOENCODER
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
License
MIT License
