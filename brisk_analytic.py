import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from streamlit_autorefresh import st_autorefresh
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- INITIALIZE NLP ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- CONFIG & REFRESH ---
st.set_page_config(page_title="Brisk Analytics | AI Terminal", layout="wide", page_icon="⚡")
st_autorefresh(interval=60000, key="auto_refresh")

# --- DATA ENGINES ---
@st.cache_data(ttl=300)
def get_all_indices():
    """Fetches Major and Minor (Sectoral) Indices."""
    # Major Indices
    major = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Nifty Midcap": "NIFTY_MID_100.NS"}
    # Minor / Sectoral Indices
    minor = {"Nifty IT": "^CNXIT", "Nifty Auto": "^CNXAUTO", "Nifty Pharma": "^CNXPHARMA", "Nifty Metal": "^CNXMETAL"}
    
    all_syms = {**major, **minor}
    res = {"major": {}, "minor": {}}
    
    try:
        data = yf.download(list(all_syms.values()), period="2d", progress=False)['Close']
        for name, sym in all_syms.items():
            series = data[sym].dropna()
            if len(series) >= 2:
                curr, prev = series.iloc[-1], series.iloc[-2]
                pct = ((curr - prev) / prev) * 100
                category = "major" if name in major else "minor"
                res[category][name] = (curr, pct)
    except: pass
    return res

@st.cache_data(ttl=3600)
def get_nifty_500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    df['Display_Name'] = df['Symbol'] + " (" + df['Company Name'] + ")"
    return df

# --- 1. TOP MARKET TAPE (MAJOR & MINOR) ---
indices = get_all_indices()

# Major Row
if indices["major"]:
    cols_maj = st.columns(len(indices["major"]))
    for i, (name, (val, pct)) in enumerate(indices["major"].items()):
        cols_maj[i].metric(name, f"{val:,.0f}", f"{pct:+.2f}%")

# Minor Row (Sectorals)
if indices["minor"]:
    cols_min = st.columns(len(indices["minor"]))
    for i, (name, (val, pct)) in enumerate(indices["minor"].items()):
        cols_min[i].metric(name, f"{val:,.0f}", f"{pct:+.2f}%", delta_color="normal")

st.markdown("---")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.markdown("# ⚡ BRISK ANALYTICS")
st.sidebar.info("AI-Powered Equity Intelligence")
st.sidebar.markdown("---")

n500 = get_nifty_500()
selected_display = st.sidebar.selectbox("🔍 Search Script", options=n500['Display_Name'].tolist())
symbol = selected_display.split(" ")[0]
current_sector = n500[n500['Symbol'] == symbol]['Industry'].values[0]

st.sidebar.markdown("---")
horizon = st.sidebar.radio("⏳ Analysis Horizon", ["Short Term", "Medium Term", "Long Term"], index=1)

# --- 3. STRATEGY MAPPING ---
h_map = {
    "Short Term":  {"p": "3mo", "s": -5,  "ma": 20,  "sr": 5,  "lbl": "1-Week Forecast"},
    "Medium Term": {"p": "1y",  "s": -21, "ma": 50,  "sr": 20, "lbl": "1-Month Forecast"},
    "Long Term":   {"p": "3y",  "s": -63, "ma": 200, "sr": 60, "lbl": "1-Quarter Forecast"}
}
cfg = h_map[horizon]

# --- 4. DATA & AI ENGINE ---
ticker_obj = yf.Ticker(f"{symbol}.NS")
df = yf.download(f"{symbol}.NS", period=cfg['p'], progress=False)

if not df.empty and len(df) > 40:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Technical Analysis
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['Trend'] = ta.sma(df['Close'], length=cfg['ma'])
    res = df['High'].rolling(window=cfg['sr']).max().iloc[-1]
    sup = df['Low'].rolling(window=cfg['sr']).min().iloc[-1]

    # AI Forecast (XGBoost)
    df['Target'] = df['Close'].shift(cfg['s'])
    train = df[['RSI', 'Close', 'Target']].dropna()
    model = XGBRegressor(n_estimators=50).fit(train[['RSI', 'Close']], train['Target'])
    pred = model.predict(df[['RSI', 'Close']].iloc[[-1]])[0]
    upside = ((pred/df['Close'].iloc[-1])-1)*100

    # --- 5. MAIN DASHBOARD ---
    st.subheader(f"📈 {symbol} | {horizon} Strategy Engine")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live CMP", f"₹{df['Close'].iloc[-1]:.2f}")
    m2.metric(cfg['lbl'], f"₹{pred:.2f}", f"{upside:+.2f}%")
    m3.metric("Resistance (High)", f"₹{res:.2f}")
    m4.metric("Support (Low)", f"₹{sup:.2f}")

    cl, cr = st.columns([2.5, 1])
    with cl:
        # Main Price Chart with S&R
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label="Price", color='#1f77b4', linewidth=1.5)
        ax.plot(df.index, df['Trend'], color='orange', label=f"{cfg['ma']} MA", linestyle='--')
        ax.axhline(res, color='red', alpha=0.3, label="Resistance")
        ax.axhline(sup, color='green', alpha=0.3, label="Support")
        ax.set_facecolor('#fdfdfd')
        ax.legend(); st.pyplot(fig)

    with cr:
        # Sentiment & Sector Info
        sia = SentimentIntensityAnalyzer()
        news = ticker_obj.news
        headlines = [n.get('content', n).get('title', '') for n in news[:5]]
        avg_s = sum([sia.polarity_scores(h)['compound'] for h in headlines]) / 5 if headlines else 0
        mood = "BULLISH" if avg_s > 0.05 else "BEARISH" if avg_s < -0.05 else "NEUTRAL"
        mood_col = "#d4edda" if mood == "BULLISH" else "#f8d7da" if mood == "BEARISH" else "#fff3cd"

        st.markdown(f'<div style="background-color:{mood_col}; padding:15px; border-radius:10px; text-align:center; font-weight:bold; color:black; border:1px solid #ccc;">Mood: {mood}</div>', unsafe_allow_html=True)
        st.write(f"**Sector:** {current_sector}")
        st.bar_chart(df['Volume'].tail(25))

    with st.expander("📰 View News Intelligence"):
        for h in headlines: st.write(f"• {h}")

else:
    st.error("Data fetch failed. Ensure the ticker symbol is valid for NSE.")