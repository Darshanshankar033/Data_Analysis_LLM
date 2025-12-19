import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

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
    Conversational Analytics • AI BI • Exportable Reports
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
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])
auto_build_dashboard = st.sidebar.button("🤖 Auto-Build BI Dashboard")
export_pdf = st.sidebar.button("📄 Export BI Report (PDF)")

# =================================================
# LOAD DATA
# =================================================
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

# =================================================
# UTILITY FUNCTIONS
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
# AI BI DASHBOARD GENERATOR (DEFINED ✔️)
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

STRICT RULES:
- DataFrame is already available as `df`
- DO NOT read files
- DO NOT create dummy data
- Use st.columns() for layout
- Show 3–5 KPIs
- Create 2–4 meaningful charts
- Use matplotlib or seaborn
- End each chart with st.pyplot(plt.gcf())
- Output ONLY executable Python code
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
# LANGCHAIN-STYLE AGENTS
# =================================================
def planner_agent(query):
    prompt = f"""
Classify the user request into one category:
CHAT, VISUALIZATION, EXPORT

Request:
{query}

Return only the category name.
"""
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

def coder_agent(task, df):
    prompt = f"""
Generate ONLY executable Python code using DataFrame `df` to:
{task}

Rules:
- Use matplotlib / seaborn
- End with st.pyplot(plt.gcf())
"""
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}]
    )
    code = r.choices[0].message.content
    return code.replace("```python", "").replace("```", "")

def explainer_agent(context):
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": context}]
    )
    return r.choices[0].message.content

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

    # ---------------- AUTO BI DASHBOARD ----------------
    if auto_build_dashboard:
        st.subheader("🤖 AI-Generated BI Dashboard")

        if "bi_code" not in st.session_state:
            with st.spinner("AI is building dashboard..."):
                st.session_state.bi_code = generate_bi_dashboard_code(df)

        try:
            exec(
                st.session_state.bi_code,
                {},
                {"st": st, "pd": pd, "plt": plt, "sns": sns, "df": df.copy()}
            )
        except Exception as e:
            st.error(f"Dashboard execution error: {e}")

    # ---------------- CHAT ----------------
    st.subheader("💬 Conversational Analytics")
    user_query = st.chat_input("Ask a question, request a chart, or export report")

    if user_query:
        intent = planner_agent(user_query)

        if intent == "VISUALIZATION":
            code = coder_agent(user_query, df)
            st.code(code, language="python")
            exec(code, {}, {"st": st, "df": df, "plt": plt, "sns": sns})

        elif intent == "EXPORT":
            st.info("Use the Export PDF button in the sidebar.")

        else:
            st.markdown(explainer_agent(user_query))

    # ---------------- EXPORT PDF ----------------
    if export_pdf:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("<b>LLM-Powered BI Report</b>", styles["Title"]))
        content.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
        content.append(Paragraph(f"Columns: {', '.join(df.columns)}", styles["Normal"]))
        content.append(Paragraph("<br/><b>Dataset Summary</b><br/>", styles["Heading2"]))
        content.append(Paragraph(dataset_metadata(df).replace("\n", "<br/>"), styles["Normal"]))

        doc.build(content)
        buffer.seek(0)

        st.download_button(
            "📥 Download BI Report (PDF)",
            buffer,
            file_name="BI_Report.pdf",
            mime="application/pdf"
        )

else:
    st.info("⬅️ Upload a dataset to begin.")
