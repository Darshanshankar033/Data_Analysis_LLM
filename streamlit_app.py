import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# =====================================================
# ⚠️ API CONFIG (HARDCODED AS REQUESTED)
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
# SANITIZER (FIXES COMMON LLM SYNTAX ERRORS)
# =====================================================
def sanitize_dashboard_code(code: str) -> str:
    # Fix invalid numeric formatting hallucinations
    code = code.replace(":,.2f", "")
    return code

# =====================================================
# SESSION STATE
# =====================================================
for key in [
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
    "Upload CSV / Excel", ["csv", "xlsx"]
)

# =====================================================
# LOAD DATA
# =====================================================
df = None
if uploaded_file:
    df = (
        pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith(".csv")
        else pd.read_excel(uploaded_file)
    )

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
    ["📌 Dataset Summary", "📊 Interactive Dashboard", "💬 Chatbot"]
)

# =====================================================
# 1️⃣ SUMMARY SECTION (SEQUENTIAL PROMPTING)
# =====================================================
if df is not None:
    with summary_tab:

        if st.session_state.profile is None:
            st.session_state.profile = profile_agent(df)

            summary_prompt = f"""
You are a data analyst.

You have FULL access to the dataset.

Tasks:
1. Write a concise dataset summary
2. Highlight key insights
3. Suggest exactly 3 analytical questions users may ask

Dataset Profile:
{st.session_state.profile}
"""
            st.session_state.summary = llm(summary_prompt)

            eda_prompt = f"""
Suggest EDA directions using the FULL dataset.
Aggregations are allowed.
Do not create new columns.

Dataset Profile:
{st.session_state.profile}
"""
            st.session_state.eda_plan = llm(eda_prompt)

        st.markdown(st.session_state.summary)

# =====================================================
# 2️⃣ DASHBOARD SECTION (LLM CODE GENERATOR)
# =====================================================
if df is not None:
    with dashboard_tab:

        if st.button("🚀 Generate / Refresh Dashboard") or st.session_state.dashboard_code is None:

            dashboard_prompt = f"""
You are a BI dashboard developer.

You have FULL access to pandas DataFrame `df`.

STRICT RULES:
- Use VALID Python syntax only
- Use pandas as pd
- Use Plotly Express as px
- Aggregations ARE allowed
- If formatting numbers, use f-strings
  Example:
    total = df['Sales'].sum()
    st.metric("Total Sales", f"{total:,.2f}")

FORBIDDEN:
- File access
- Invalid formatting syntax
- Non-Python expressions

TASK:
1. Create 3–5 KPI cards using st.metric
2. Create 2–3 interactive Plotly charts
3. Use the full dataset
4. Output ONLY executable Python code
"""

            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            safe_code = sanitize_dashboard_code(
                st.session_state.dashboard_code
            )
            exec(
                safe_code,
                {},
                {"st": st, "df": df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("❌ Dashboard execution failed")
            st.exception(e)

# =====================================================
# 3️⃣ CHATBOT SECTION (FULL DATA ACCESS)
# =====================================================
if df is not None:
    with chat_tab:

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])

        user_input = st.chat_input(
            "Ask questions like total sales, inventory, trends, or regenerate dashboard"
        )

        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            chat_prompt = f"""
You are a conversational data assistant.

You have FULL access to the pandas DataFrame `df`.

Chat History:
{st.session_state.chat_history}

User Query:
{user_input}

Rules:
- Use valid Python logic
- Aggregations allowed
- If user asks to regenerate dashboard, return EXACTLY: REGENERATE_DASHBOARD
- Otherwise, explain insights clearly
"""

            reply = llm(chat_prompt)

            if "REGENERATE_DASHBOARD" in reply:
                st.session_state.dashboard_code = None
                reply = "✅ Dashboard regenerated based on your request."

            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )

            st.chat_message("assistant").markdown(reply)

else:
    st.info("⬅️ Upload a dataset to begin.")
