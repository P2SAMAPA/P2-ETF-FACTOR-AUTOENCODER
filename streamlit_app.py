"""
Streamlit Dashboard for Factor Autoencoder Engine.
Displays factor exposures, latent factor returns, and daily top picks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
import json
import numpy as np
import config

st.set_page_config(
    page_title="P2Quant Factor Autoencoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem; }
    .hero-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-score { font-size: 2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_hero_card(ticker: str, total_score: float, components: dict):
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🧠 TOP PICK FOR TOMORROW</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-score">Score: {total_score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Signal Breakdown"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Factor Momentum", f"{components.get('factor_momentum_z', 0):.3f}")
        with col2:
            st.metric("Reconstruction Anomaly", f"{components.get('reconstruction_error_z', 0):.3f}")
        with col3:
            st.metric("Residual Alpha", f"{components.get('residual_alpha_z', 0):.3f}")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()
st.sidebar.markdown("### 🧠 Autoencoder Parameters")
st.sidebar.markdown(f"- Latent Factors: **{config.LATENT_DIM}**")
st.sidebar.markdown(f"- Hidden Dims: **{config.HIDDEN_DIMS}**")
st.sidebar.markdown(f"- Signal Weights: FM={config.SIGNAL_WEIGHTS['factor_momentum']:.2f}, RE={config.SIGNAL_WEIGHTS['reconstruction_error']:.2f}, RA={config.SIGNAL_WEIGHTS['residual_alpha']:.2f}")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

st.sidebar.divider()
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**Factor Autoencoder** extracts latent market factors and combines three signals:
- **Factor Momentum**: Recent strength of each factor
- **Reconstruction Anomaly**: How unusual today's market is
- **Residual Alpha**: ETF-specific return not explained by factors
""")

# --- Main Content ---
st.markdown('<div class="main-header">🧠 P2Quant Factor Autoencoder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unsupervised Latent Factor Extraction & Multi‑Signal Ranking</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 Daily Top Picks", "📊 Factor Analysis", "📆 Shrinking Windows"])

with tab1:
    st.markdown("### Today's Target ETFs")
    
    top_picks = data['global_model']['top_picks']
    signals = data['global_model']['signals']
    
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    
    for subtab, ukey in zip(subtabs, universe_keys):
        with subtab:
            if ukey in top_picks:
                pick = top_picks[ukey]
                components = pick.get('components', {})
                display_hero_card(pick['ticker'], pick['total_score'], components)
                
                st.markdown("### All ETF Scores")
                universe_scores = signals.get(ukey, {})
                if universe_scores:
                    rows = []
                    for ticker, s in universe_scores.items():
                        rows.append({
                            'Ticker': ticker,
                            'Total Score': f"{s['total_score']:.3f}",
                            'Factor Mom': f"{s['factor_momentum_z']:.3f}",
                            'Recon Anom': f"{s['reconstruction_error_z']:.3f}",
                            'Resid Alpha': f"{s['residual_alpha_z']:.3f}"
                        })
                    df = pd.DataFrame(rows).sort_values('Total Score', ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Latent Factor Returns")
    
    factor_returns = np.array(data['global_model']['factor_returns'])
    dates = pd.date_range(end=config.TODAY, periods=len(factor_returns))
    
    fig_factors = go.Figure()
    for i in range(factor_returns.shape[1]):
        fig_factors.add_trace(go.Scatter(
            x=dates, y=factor_returns[:, i],
            mode='lines', name=f'Factor {i+1}'
        ))
    fig_factors.update_layout(
        title="Latent Factor Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Factor Value",
        height=400
    )
    st.plotly_chart(fig_factors, use_container_width=True)
    
    st.markdown("### ETF Factor Exposures (Betas)")
    exposures = data['global_model']['factor_exposures']
    if exposures:
        exp_df = pd.DataFrame(exposures).T
        exp_df.columns = [f'Factor {i+1}' for i in range(exp_df.shape[1])]
        fig_heat = px.imshow(
            exp_df.T,
            labels=dict(x="ETF", y="Factor", color="Beta"),
            title="Factor Exposures Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.markdown("### Top Picks Across Historical Windows")
    
    shrinking = data.get('shrinking_windows', {})
    if not shrinking:
        st.warning("No shrinking windows data available.")
        st.stop()
    
    shrink_subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    
    for subtab, ukey in zip(shrink_subtabs, universe_keys):
        with subtab:
            rows = []
            for label, winfo in sorted(shrinking.items(), key=lambda x: x[1]['start_year'], reverse=True):
                top = winfo['top_picks'].get(ukey, {})
                if top:
                    rows.append({
                        'Window': label,
                        'Top Pick': top.get('ticker', 'N/A'),
                        'Score': f"{top.get('total_score', 0):.3f}",
                        'Observations': winfo.get('n_observations', 0)
                    })
            if rows:
                df_win = pd.DataFrame(rows)
                st.dataframe(df_win, use_container_width=True, hide_index=True)
                
                df_chart = df_win.copy()
                df_chart['Score_val'] = df_chart['Score'].astype(float)
                fig = go.Figure(go.Scatter(
                    x=df_chart['Window'], y=df_chart['Score_val'],
                    mode='lines+markers', text=df_chart['Top Pick'],
                    line=dict(color='#667eea', width=3)
                ))
                fig.update_layout(
                    title=f"{ukey} – Top Pick Score by Window",
                    xaxis_title="Window Start Year",
                    yaxis_title="Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
