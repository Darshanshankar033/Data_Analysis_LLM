import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
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
st.title("🤖 AI-Generated BI Dashboard")

# =====================================================
# DARK / LIGHT PREVIEW MODE
# =====================================================
dark_mode = st.sidebar.toggle("🌙 Dark Mode Preview", value=False)

if dark_mode:
    st.markdown("""
    <style>
    .stApp { background:#0f1021; color:white; }
    .card { background:#1c1d3a; border-radius:14px; padding:20px;
            box-shadow:0 4px 20px rgba(0,0,0,.6); margin-bottom:20px; }
    .section-title { color:#e4e4ff; font-size:18px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background:#f6f7fb; }
    .card { background:white; border-radius:14px; padding:20px;
            box-shadow:0 4px 18px rgba(0,0,0,.06); margin-bottom:20px; }
    .section-title { color:#222; font-size:18px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SAFE LLM CALL
# =====================================================
def llm(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error("LLM authentication / quota error")
        st.exception(e)
        return ""

# =====================================================
# SANITIZER (CRITICAL)
# =====================================================
def sanitize(code: str) -> str:
    if "```" in code:
        code = code.replace("```python", "").replace("```", "")
    # strip unsupported plotly features
    code = code.replace("trendline='ols'", "")
    code = code.replace('trendline="ols"', "")
    code = code.replace("trendline='lowess'", "")
    code = code.replace('trendline="lowess"', "")
    return code.strip()

# =====================================================
# SESSION STATE
# =====================================================
for k in ["file_type", "df", "pdf_text", "dashboard_code", "chat"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "chat" else []

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded = st.sidebar.file_uploader(
    "Upload CSV / Excel / PDF",
    ["csv", "xlsx", "pdf"]
)

if uploaded:
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.file_type = "tabular"

    elif name.endswith(".xlsx"):
        st.session_state.df = pd.read_excel(uploaded)
        st.session_state.file_type = "tabular"

    elif name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
        st.session_state.pdf_text = text[:12000]
        st.session_state.file_type = "pdf"

# =====================================================
# SYSTEM-CONTROLLED SLICERS (POWER BI STYLE)
# =====================================================
def apply_slicers(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
    filtered = df.copy()

    for col in df.columns[:3]:  # limit slicers for stability
        if pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = float(df[col].min()), float(df[col].max())
            rng = st.slider(col, mn, mx, (mn, mx))
            filtered = filtered[(filtered[col] >= rng[0]) & (filtered[col] <= rng[1])]
        elif df[col].nunique() <= 20:
            opts = st.multiselect(col, df[col].unique(), df[col].unique())
            filtered = filtered[filtered[col].isin(opts)]

    return filtered

# =====================================================
# UI TABS
# =====================================================
summary_tab, dashboard_tab, chat_tab = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chat"]
)

# =====================================================
# SUMMARY TAB
# =====================================================
with summary_tab:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df
        st.markdown(
            llm(
                f"""
You are a data analyst.

Summarize the dataset and suggest exactly 3 business questions.

Columns: {list(df.columns)}
Rows: {len(df)}
"""
            )
        )
    elif st.session_state.file_type == "pdf":
        st.markdown(
            llm(
                f"""
Summarize the document and suggest 3 questions.

Document:
{st.session_state.pdf_text}
"""
            )
        )

# =====================================================
# DASHBOARD TAB (LLM CODE → EXECUTION)
# =====================================================
with dashboard_tab:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df
        filtered_df = apply_slicers(df)

        if filtered_df.empty:
            st.warning("No data available for selected filters.")
        else:
            if st.button("🚀 Generate Dashboard") or st.session_state.dashboard_code is None:

                dashboard_prompt = """
You are a BI dashboard developer.

DATA:
- filtered_df (already filtered pandas DataFrame)

ABSOLUTE RULES:
- DO NOT assume column names
- Detect columns dynamically:
    num_cols = filtered_df.select_dtypes(include='number').columns
    cat_cols = filtered_df.select_dtypes(exclude='number').columns
- DO NOT use trendlines, regression, statsmodels, sklearn
- Use Plotly Express only: bar, line, scatter (no trendline), pie, histogram
- All variables must be defined
- NO markdown fences

STYLE RULES:
- Wrap visuals in <div class="card">
- Use <div class="section-title">
- Use st.columns for layout

TASK:
1. Create 3 KPI cards from numeric columns
2. Create 1 main chart
3. Create 1–2 secondary charts

Output ONLY executable Python code.
"""

                st.session_state.dashboard_code = llm(dashboard_prompt)

            if filtered_df.select_dtypes(include="number").empty:
                st.warning("No numeric columns available for KPIs.")
            else:
                try:
                    exec(
                        sanitize(st.session_state.dashboard_code),
                        {},
                        {
                            "st": st,
                            "filtered_df": filtered_df,
                            "px": px,
                            "pd": pd,
                        },
                    )
                except Exception as e:
                    st.error("Dashboard execution failed")
                    st.code(st.session_state.dashboard_code, language="python")
                    st.exception(e)
    else:
        st.info("Dashboards are not supported for PDFs.")

# =====================================================
# CHAT TAB (HUMAN LANGUAGE ONLY)
# =====================================================
with chat_tab:
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    q = st.chat_input("Ask a business question")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})

        if st.session_state.file_type == "tabular":
            prompt = f"""
You are a business analyst.

Answer the question in clear human language.
Use the dataset internally.
DO NOT show code.

Question:
{q}
"""
        else:
            prompt = f"""
Answer using the document only.

Document:
{st.session_state.pdf_text}

Question:
{q}
"""

        ans = llm(prompt)

        if "```" in ans or "df[" in ans:
            ans = "Here is the answer in simple terms:\n\n" + ans.replace("```", "")

        st.session_state.chat.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").markdown(ans)

# =====================================================
# EMPTY STATE
# =====================================================
if not uploaded:
    st.info("⬅️ Upload a CSV, Excel, or PDF to begin.")
