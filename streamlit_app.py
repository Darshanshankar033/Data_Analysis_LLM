import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="LLM-Powered Interactive BI Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>📊 LLM-Powered Interactive BI Dashboard</h1>
    <p style='text-align:center;color:gray;'>
    Conversational Analytics • AI-Generated BI Dashboard • Cloud Safe
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =================================================
# OPENROUTER CLIENT
# =================================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# =================================================
# SIDEBAR CONTROLS
# =================================================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    ["csv", "xlsx"]
)

auto_build_dashboard = st.sidebar.button("🤖 Auto-Build BI Dashboard")
export_report = st.sidebar.button("📄 Export BI Report")

# =================================================
# LOAD DATA
# =================================================
df = None
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# =================================================
# DATA METADATA
# =================================================
def dataset_metadata(df):
    return f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Types:
{df.dtypes}

Missing Values:
{df.isnull().sum()}

Summary Statistics:
{df.describe(include='all')}
"""

# =================================================
# 🔒 SAFETY: BLOCK FILE ACCESS IN AI CODE
# =================================================
def sanitize_generated_code(code: str) -> str:
    forbidden = [
        "read_csv",
        "read_excel",
        ".csv",
        ".xlsx",
        "open("
    ]
    for f in forbidden:
        if f in code:
            raise ValueError("Unsafe code detected: file access is not allowed.")
    return code

# =================================================
# AI BI DASHBOARD GENERATOR (FIXED)
# =================================================
def generate_bi_dashboard_code(df):
    schema = {
        "rows": df.shape[0],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    prompt = f"""
You are a senior BI dashboard architect.

Dataset schema:
{schema}

STRICT RULES (MUST FOLLOW):
- Dataset is already loaded as pandas DataFrame `df`
- DO NOT read files
- DO NOT use pd.read_csv or pd.read_excel
- DO NOT reference any filenames
- ONLY operate on `df`

TASK:
Generate Streamlit Python code that:
1. Displays 3–5 KPI metrics
2. Creates 2–4 meaningful charts
3. Uses st.columns() for layout
4. Uses matplotlib or seaborn
5. Ends each chart with st.pyplot(plt.gcf())
6. Outputs ONLY executable Python code
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}],
    )

    code = response.choices[0].message.content.strip()
    for fence in ("```python", "```", "`"):
        code = code.replace(fence, "")
    return code

# =================================================
# MAIN APP
# =================================================
if df is not None:

    # ---------------- KPIs ----------------
    st.subheader("📌 KPI Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    # ---------------- DATA PREVIEW ----------------
    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df, use_container_width=True)

    # ---------------- AI INSIGHTS ----------------
    st.subheader("🧠 AI Insights")
    insight_prompt = f"Provide insights for this dataset:\n{dataset_metadata(df)}"
    insights = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": insight_prompt}],
    ).choices[0].message.content
    st.markdown(insights)

    # ---------------- AI BI DASHBOARD ----------------
    if auto_build_dashboard:
        st.subheader("🤖 AI-Generated BI Dashboard")

        if "bi_code" not in st.session_state:
            with st.spinner("AI is building the dashboard..."):
                st.session_state.bi_code = generate_bi_dashboard_code(df)

        try:
            safe_code = sanitize_generated_code(st.session_state.bi_code)
            exec(
                safe_code,
                {},
                {"st": st, "pd": pd, "plt": plt, "sns": sns, "df": df.copy()}
            )
        except Exception as e:
            st.error("Dashboard generation failed due to unsafe AI code.")
            st.exception(e)

    # ---------------- CHAT ----------------
    st.subheader("💬 Conversational Analytics")
    user_query = st.chat_input("Ask a question or request a chart")

    if user_query:
        if any(k in user_query.lower() for k in ["plot", "chart", "graph"]):
            code_prompt = f"""
Generate ONLY Python code using DataFrame `df` to:
{user_query}

Rules:
- Use matplotlib or seaborn
- End with st.pyplot(plt.gcf())
"""
            code = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": code_prompt}],
            ).choices[0].message.content

            code = code.replace("```python", "").replace("```", "")
            st.code(code, language="python")
            exec(code, {}, {"st": st, "df": df, "plt": plt, "sns": sns})

        else:
            reply = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": user_query}],
            ).choices[0].message.content
            st.markdown(reply)

    # ---------------- EXPORT REPORT ----------------
    if export_report:
        report = f"""
LLM-Powered BI Report

Rows: {df.shape[0]}
Columns: {df.shape[1]}

INSIGHTS:
{insights}
"""
        st.download_button(
            "📥 Download BI Report (TXT)",
            report,
            file_name="BI_Report.txt"
        )

        st.download_button(
            "📥 Download Dataset Snapshot (CSV)",
            df.to_csv(index=False),
            file_name="Dataset.csv"
        )

else:
    st.info("⬅️ Upload a dataset to start.")
