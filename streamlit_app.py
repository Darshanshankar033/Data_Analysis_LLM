import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
from openai import OpenAI

# =====================================================
# API CONFIG (HARDCODED AS REQUESTED)
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
st.set_page_config(
    page_title="LLM-Powered BI Platform",
    layout="wide"
)
st.title("🤖 LLM-Powered Data Analysis & BI Platform")

# =====================================================
# SAFE LLM CALL
# =====================================================
def llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error("❌ LLM call failed")
        st.exception(e)
        return ""

# =====================================================
# SANITIZER (LLM SAFETY)
# =====================================================
def sanitize_dashboard_code(code: str) -> str:
    code = code.replace(":,.2f", "")
    return code

# =====================================================
# SESSION STATE
# =====================================================
for key in [
    "file_type",
    "df",
    "pdf_text",
    "profile",
    "summary",
    "eda_plan",
    "dashboard_code",
    "chat_history"
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel / PDF", ["csv", "xlsx", "pdf"]
)

# =====================================================
# LOAD FILE
# =====================================================
if uploaded_file:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.file_type = "tabular"

    elif file_name.endswith(".xlsx"):
        st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.file_type = "tabular"

    elif file_name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        st.session_state.pdf_text = text[:12000]  # limit tokens
        st.session_state.file_type = "pdf"

# =====================================================
# DATA PROFILER
# =====================================================
def profile_agent(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

# =====================================================
# UI SECTIONS
# =====================================================
summary_tab, dashboard_tab, chat_tab = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chat"]
)

# =====================================================
# 1️⃣ SUMMARY
# =====================================================
with summary_tab:

    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.session_state.profile is None:
            st.session_state.profile = profile_agent(df)

            summary_prompt = f"""
You are a data analyst.

Summarize the dataset, highlight insights,
and suggest exactly 3 analytical questions.

Dataset Profile:
{st.session_state.profile}
"""
            st.session_state.summary = llm(summary_prompt)

            eda_prompt = f"""
Suggest EDA ideas using the FULL dataset.
Aggregations allowed. No new columns.

Dataset Profile:
{st.session_state.profile}
"""
            st.session_state.eda_plan = llm(eda_prompt)

        st.markdown(st.session_state.summary)

    elif st.session_state.file_type == "pdf":

        pdf_prompt = f"""
You are a document analyst.

Summarize the document content and
suggest 3 questions the user may ask.

Document Text:
{st.session_state.pdf_text}
"""
        st.markdown(llm(pdf_prompt))

# =====================================================
# 2️⃣ DASHBOARD (TABULAR ONLY)
# =====================================================
with dashboard_tab:

    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.button("🚀 Generate / Refresh Dashboard") or st.session_state.dashboard_code is None:

            dashboard_prompt = f"""
You are a BI dashboard developer.

DataFrame name: df
Use valid Python only.
All variables must be defined.

TASK:
- 3–5 KPI cards using st.metric
- 2–3 Plotly charts
- Aggregations allowed
- Output ONLY executable Python code
"""
            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            exec(
                sanitize_dashboard_code(st.session_state.dashboard_code),
                {},
                {"st": st, "df": df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("Dashboard execution failed")
            st.exception(e)

    else:
        st.info("📄 Dashboards are not applicable for PDF documents.")

# =====================================================
# 3️⃣ CHAT
# =====================================================
with chat_tab:

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask questions about the data or document")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        if st.session_state.file_type == "tabular":
            chat_prompt = f"""
You are a data assistant.

You have FULL access to the pandas DataFrame df.

User Question:
{user_input}
"""
        else:
            chat_prompt = f"""
You are a document assistant.

Answer based ONLY on the document text.

Document Text:
{st.session_state.pdf_text}

User Question:
{user_input}
"""

        reply = llm(chat_prompt)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )
        st.chat_message("assistant").markdown(reply)
