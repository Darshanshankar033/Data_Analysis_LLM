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
# GLOBAL BI STYLING (CRITICAL)
# =====================================================
st.markdown("""
<style>
.stApp { background-color: #f6f7fb; }
.card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.card-title { font-size: 14px; color: #777; }
.card-value { font-size: 28px; font-weight: 700; }
.section-title { font-size: 20px; font-weight: 600; margin: 10px 0; }
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
    code = code.replace(":,.2f", "")
    return code.strip()

# =====================================================
# SESSION STATE
# =====================================================
for k in ["file_type", "df", "pdf_text", "dashboard_code", "chat"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "chat" else []

# =====================================================
# SIDEBAR
# =====================================================
uploaded = st.sidebar.file_uploader(
    "Upload CSV / Excel / PDF", ["csv", "xlsx", "pdf"]
)

# =====================================================
# LOAD DATA
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
        prompt = f"""
Summarize the dataset and suggest exactly 3 business questions.

Columns: {list(df.columns)}
Rows: {len(df)}
"""
        st.markdown(llm(prompt))

    elif st.session_state.file_type == "pdf":
        prompt = f"""
Summarize the document and suggest 3 questions.

Document:
{st.session_state.pdf_text}
"""
        st.markdown(llm(prompt))

# =====================================================
# DASHBOARD (LLM-GENERATED + STYLED)
# =====================================================
with dashboard_tab:
    if st.session_state.file_type == "tabular":
        df = st.session_state.df

        if st.button("🚀 Generate Styled Dashboard") or st.session_state.dashboard_code is None:

            dashboard_prompt = f"""
You are a BI dashboard designer.

You MUST generate a visually styled dashboard using Streamlit.

AVAILABLE:
- st.markdown
- st.columns
- px (Plotly Express)
- df (pandas DataFrame)

STYLE RULES (MANDATORY):
- Wrap all visuals in <div class="card">
- Use <div class="section-title"> for headings
- KPI cards must show title + value
- Use grid layout similar to Power BI / SaaS dashboards

DATA RULES:
- Full dataset access
- Aggregations allowed
- All variables must be defined
- Valid Python only
- NO markdown fences
- NO file access

TASK:
1. Create a KPI row (3–4 cards)
2. Create a main chart section
3. Create a secondary chart section
4. Output ONLY executable Python code
"""

            st.session_state.dashboard_code = llm(dashboard_prompt)

        try:
            exec(
                sanitize(st.session_state.dashboard_code),
                {},
                {"st": st, "df": df, "px": px, "pd": pd}
            )
        except Exception as e:
            st.error("Dashboard failed")
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
Answer clearly in human language.
Do NOT show code.

User question:
{q}
"""
        else:
            prompt = f"""
Answer using document content only.

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
