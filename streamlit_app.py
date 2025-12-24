import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
import yfinance as yf
import requests
from openai import OpenAI

# =====================================================
# API CONFIG (OPENROUTER)
# =====================================================
OPENROUTER_API_KEY = "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"
MODEL = "openai/gpt-oss-20b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI BI Dashboard"
    }
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("🤖 AI-Powered BI Dashboard")

# =====================================================
# DARK / LIGHT PREVIEW
# =====================================================
dark_mode = st.sidebar.toggle("🌙 Dark Mode Preview", value=False)

if dark_mode:
    st.markdown("""
    <style>
    .stApp { background:#0f1021; color:white; }
    .card { background:#1c1d3a; border-radius:14px; padding:20px;
            box-shadow:0 4px 20px rgba(0,0,0,.6); margin-bottom:20px; }
    .section-title { color:#e4e4ff; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background:#f6f7fb; }
    .card { background:white; border-radius:14px; padding:20px;
            box-shadow:0 4px 18px rgba(0,0,0,.06); margin-bottom:20px; }
    .section-title { color:#222; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# LLM CALL
# =====================================================
def llm(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error("LLM error")
        st.exception(e)
        return ""

# =====================================================
# SANITIZER
# =====================================================
def sanitize(code: str) -> str:
    if "```" in code:
        code = code.replace("```python", "").replace("```", "")
    for bad in ["trendline", "statsmodels", "sklearn"]:
        code = code.replace(bad, "")
    return code.strip()

# =====================================================
# SESSION STATE
# =====================================================
for k in ["df", "pdf_text", "dashboard_code", "chat"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "chat" else []

# =====================================================
# SIDEBAR – DATA SOURCE
# =====================================================
st.sidebar.header("📊 Data Source")
source = st.sidebar.radio(
    "Choose data source",
    [
        "Upload File",
        "Stock Market Data",
        "Crypto Prices",
        "Weather Data"
    ]
)

# =====================================================
# FILE UPLOAD
# =====================================================
if source == "Upload File":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV / Excel / PDF",
        ["csv", "xlsx", "pdf"]
    )

    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded)
        elif name.endswith(".xlsx"):
            st.session_state.df = pd.read_excel(uploaded)
        elif name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(uploaded) as pdf:
                for p in pdf.pages:
                    text += p.extract_text() or ""
            st.session_state.pdf_text = text[:12000]

# =====================================================
# STOCK MARKET API (Yahoo Finance)
# =====================================================
if source == "Stock Market Data":
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    period = st.sidebar.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y"]
    )
    if st.sidebar.button("📈 Fetch Stock Data"):
        data = yf.download(symbol, period=period)
        data.reset_index(inplace=True)
        st.session_state.df = data

# =====================================================
# CRYPTO API (CoinGecko)
# =====================================================
if source == "Crypto Prices":
    coin = st.sidebar.text_input("Coin ID", "bitcoin")
    days = st.sidebar.selectbox("Days", [7, 30, 90, 365])

    if st.sidebar.button("💱 Fetch Crypto Data"):
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params).json()

        df = pd.DataFrame(r["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        st.session_state.df = df

# =====================================================
# WEATHER API (Open-Meteo)
# =====================================================
if source == "Weather Data":
    city = st.sidebar.text_input("City (lat,lon)", "12.97,77.59")
    days = st.sidebar.selectbox("Days", [3, 7, 14])

    if st.sidebar.button("🌦️ Fetch Weather Data"):
        lat, lon = city.split(",")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "forecast_days": days
        }
        r = requests.get(url, params=params).json()
        df = pd.DataFrame({
            "time": r["hourly"]["time"],
            "temperature": r["hourly"]["temperature_2m"]
        })
        df["time"] = pd.to_datetime(df["time"])
        st.session_state.df = df

# =====================================================
# DATA CLEANING
# =====================================================
if st.session_state.df is not None:
    st.sidebar.header("🧹 Data Cleaning")
    if st.sidebar.button("Clean Dataset"):
        df = st.session_state.df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df.drop_duplicates(inplace=True)
        df.fillna(method="ffill", inplace=True)
        st.session_state.df = df
        st.sidebar.success("Dataset cleaned")

# =====================================================
# SLICERS
# =====================================================
def apply_slicers(df):
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
    filtered = df.copy()

    for col in df.columns[:3]:
        if pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = float(df[col].min()), float(df[col].max())
            rng = st.slider(col, mn, mx, (mn, mx))
            filtered = filtered[(filtered[col] >= rng[0]) & (filtered[col] <= rng[1])]
        elif df[col].nunique() <= 20:
            opts = st.multiselect(col, df[col].unique(), df[col].unique())
            filtered = filtered[filtered[col].isin(opts)]

    return filtered

# =====================================================
# TABS
# =====================================================
summary_tab, dashboard_tab, chat_tab = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chat"]
)

# =====================================================
# SUMMARY
# =====================================================
with summary_tab:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(llm(
            f"Summarize the dataset and suggest 3 business questions.\nColumns: {list(df.columns)}"
        ))

# =====================================================
# DASHBOARD
# =====================================================
with dashboard_tab:
    if st.session_state.df is not None:
        df = st.session_state.df
        filtered_df = apply_slicers(df)

        if st.button("🚀 Generate Dashboard") or st.session_state.dashboard_code is None:
            dashboard_prompt = """
You are a BI dashboard developer.

DATA:
- filtered_df (pandas DataFrame)

RULES:
- Dynamically detect columns
- No trendlines or advanced stats
- Use Plotly Express only
- Use <div class="card">
- Use st.columns layout

TASK:
1. KPI cards (3)
2. Main chart
3. Secondary chart

Output ONLY executable Python code.
"""
            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            exec(
                sanitize(st.session_state.dashboard_code),
                {},
                {"st": st, "filtered_df": filtered_df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("Dashboard execution failed")
            st.code(st.session_state.dashboard_code)
            st.exception(e)

# =====================================================
# CHATBOT
# =====================================================
with chat_tab:
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    q = st.chat_input("Ask a business question")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        ans = llm(f"Answer clearly in human language:\n{q}")
        st.session_state.chat.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").markdown(ans)

# =====================================================
# EMPTY STATE
# =====================================================
if st.session_state.df is None:
    st.info("⬅️ Select a data source to begin.")
