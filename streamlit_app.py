import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
from openai import OpenAI

# =====================================================
# API CONFIG
# =====================================================
OPENROUTER_API_KEY = "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"
MODEL = "openai/gpt-oss-20b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("🤖 AI-Generated BI Dashboard")

# =====================================================
# THEME PREVIEW TOGGLE
# =====================================================
dark_mode = st.sidebar.toggle("🌙 Dark Mode Preview", value=False)

# =====================================================
# GLOBAL STYLING
# =====================================================
if dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0f1021; color: white; }
    .card { background: #1c1d3a; border-radius: 14px; padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6); margin-bottom: 20px; }
    .card-title { color: #b5b5ff; }
    .card-value { color: white; }
    .section-title { color: #e4e4ff; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #f6f7fb; }
    .card { background: white; border-radius: 14px; padding: 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06); margin-bottom: 20px; }
    .card-title { color: #777; }
    .card-value { color: #222; }
    .section-title { color: #222; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SAFE LLM CALL
# =====================================================
def llm(prompt):
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error("LLM Error")
        st.exception(e)
        return ""

# =====================================================
# SANITIZER
# =====================================================
def sanitize(code):
    if "```" in code:
        code = code.replace("```python", "").replace("```", "")
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
    "Upload CSV / Excel / PDF", ["csv", "xlsx", "pdf"]
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
# TABS
# =====================================================
summary_tab, dashboard_tab, chat_tab = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chat"]
)

# =====================================================
# SUMMARY
# =====================================================
with summary_tab:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df
        st.markdown(llm(f"Summarize the dataset and suggest 3 business questions.\nColumns: {list(df.columns)}"))
    elif st.session_state.file_type == "pdf":
        st.markdown(llm(f"Summarize the document:\n{st.session_state.pdf_text}"))

# =====================================================
# DASHBOARD WITH SLICERS (LLM-GENERATED)
# =====================================================
with dashboard_tab:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.button("🚀 Generate Styled Dashboard with Slicers") or st.session_state.dashboard_code is None:

            dashboard_prompt = f"""
You are a BI dashboard designer.

You MUST generate a styled Streamlit dashboard WITH SLICERS.

AVAILABLE:
- st.selectbox, st.multiselect, st.slider
- df (original dataframe)
- filtered_df (you must create this)
- px for charts

MANDATORY STEPS:
1. Identify 2–3 important slicer columns
2. Create slicers at the TOP of the dashboard
3. Filter df into filtered_df based on slicers
4. Use filtered_df for ALL KPIs and charts

STYLE RULES:
- Wrap visuals in <div class="card">
- Use <div class="section-title"> headings
- Grid layout using st.columns

DATA RULES:
- Valid Python only
- No markdown fences
- Aggregations allowed
- Full dataset access

Output ONLY executable Python code.
"""

            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            exec(
                sanitize(st.session_state.dashboard_code),
                {},
                {"st": st, "df": df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("Dashboard execution failed")
            st.code(st.session_state.dashboard_code, language="python")
            st.exception(e)
    else:
        st.info("Dashboards are not supported for PDFs.")

# =====================================================
# CHAT (HUMAN LANGUAGE ONLY)
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
Do NOT show code.

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
    st.info("⬅️ Upload a file to begin.")
