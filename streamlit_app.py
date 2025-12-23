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
    page_title="LLM-Powered BI Assistant",
    layout="wide"
)
st.title("🤖 LLM-Powered Data Analysis & BI Assistant")

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
        st.error("LLM call failed")
        st.exception(e)
        return ""

# =====================================================
# SANITIZER (CRITICAL)
# =====================================================
def sanitize_code(code: str) -> str:
    if "```" in code:
        code = code.replace("```python", "").replace("```", "")
    code = code.replace(":,.2f", "")
    return code.strip()

# =====================================================
# SESSION STATE
# =====================================================
for k in [
    "file_type", "df", "pdf_text",
    "profile", "summary", "eda",
    "dashboard_code", "chat"
]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "chat" else []

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Upload Data")
uploaded = st.sidebar.file_uploader(
    "CSV / Excel / PDF", ["csv", "xlsx", "pdf"]
)

# =====================================================
# LOAD FILE
# =====================================================
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
# DATA PROFILER
# =====================================================
def profile_data(df):
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict()
    }

# =====================================================
# UI TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chatbot"]
)

# =====================================================
# SUMMARY TAB
# =====================================================
with tab1:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.session_state.profile is None:
            st.session_state.profile = profile_data(df)

            summary_prompt = f"""
You are a senior data analyst.

Summarize the dataset, highlight insights,
and suggest exactly 3 questions a business user may ask.

Dataset profile:
{st.session_state.profile}
"""
            st.session_state.summary = llm(summary_prompt)

            eda_prompt = f"""
Suggest EDA directions using the full dataset.
Aggregations allowed. No new columns.

Dataset profile:
{st.session_state.profile}
"""
            st.session_state.eda = llm(eda_prompt)

        st.markdown(st.session_state.summary)

    elif st.session_state.file_type == "pdf":
        prompt = f"""
Summarize the document and suggest 3 questions.

Document:
{st.session_state.pdf_text}
"""
        st.markdown(llm(prompt))

# =====================================================
# DASHBOARD TAB (TABULAR ONLY)
# =====================================================
with tab2:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.button("🚀 Generate / Refresh Dashboard") or st.session_state.dashboard_code is None:

            dashboard_prompt = f"""
You are a BI dashboard developer.

You have full access to pandas DataFrame df.

RULES:
- VALID Python only
- No markdown
- No file access
- Every variable must be defined before use
- Use st.metric for KPIs
- Use Plotly Express charts
- Aggregations allowed

Example:
total_sales = df.select_dtypes("number").sum().sum()
st.metric("Total Sales", f"{{total_sales:,.2f}}")

TASK:
- 3–5 KPIs
- 2–3 interactive charts
- Output ONLY Python code
"""
            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            exec(
                sanitize_code(st.session_state.dashboard_code),
                {},
                {"st": st, "df": df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("Dashboard execution failed")
            st.code(st.session_state.dashboard_code, language="python")
            st.exception(e)

    else:
        st.info("Dashboards are not applicable for PDF files.")

# =====================================================
# CHATBOT TAB (HUMAN LANGUAGE ONLY)
# =====================================================
with tab3:
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    user_q = st.chat_input("Ask a question about the data or document")

    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})

        if st.session_state.file_type == "tabular":
            prompt = f"""
You are a business data analyst.

Answer the question in clear human language.
Use the dataset internally but DO NOT show code.
Be specific and concise.

User question:
{user_q}
"""
        else:
            prompt = f"""
Answer using only the document content.
Use clear human language.

Document:
{st.session_state.pdf_text}

Question:
{user_q}
"""

        answer = llm(prompt)

        # Final guardrail
        if "```" in answer or "df[" in answer or "import " in answer:
            answer = "I analyzed the data and here is the answer in simple terms:\n\n" + answer.replace("```", "")

        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

# =====================================================
# EMPTY STATE
# =====================================================
if not uploaded:
    st.info("⬅️ Upload a CSV, Excel, or PDF to begin.")
