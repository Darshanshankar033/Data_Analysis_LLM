import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="LLM-Powered Interactive Data Analysis & Visualization",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>📊 LLM-Powered Interactive Data Analysis & Visualization</h1>
    <p style='text-align:center;color:gray;'>
    Conversational Analytics • AI-Generated BI Dashboard • Power BI–Style Filters
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
    type=["csv", "xlsx"]
)

mode = st.sidebar.radio(
    "Mode",
    ["💬 Chat", "🧠 Code", "⚙️ Auto"],
    index=2
)

auto_build_dashboard = st.sidebar.button("🤖 Auto-Build BI Dashboard")
clean_data_btn = st.sidebar.button("🧽 Clean Data")

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
# CLEAN DATA
# =================================================
if df is not None and clean_data_btn:
    with st.spinner("Cleaning data..."):
        df = df.drop_duplicates()
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                )
        st.sidebar.success("Data cleaned successfully")

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
# AI SLICER GENERATOR
# =================================================
def generate_slicer_code(df):
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    prompt = f"""
You are a senior BI dashboard developer.

Dataset schema:
{schema}

Generate Streamlit Python code that:
- Creates Power BI–style slicers
- Uses multiselect for categorical columns
- Uses slider for numeric columns
- Uses date_input for datetime columns
- Produces a DataFrame called filtered_df
- Output ONLY executable Python code
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}],
    )

    code = response.choices[0].message.content
    for fence in ("```python", "```", "`"):
        code = code.replace(fence, "")
    return code

# =================================================
# AI BI DASHBOARD GENERATOR
# =================================================
def generate_bi_dashboard_code(filtered_df):
    schema = {
        "rows": filtered_df.shape[0],
        "columns": filtered_df.columns.tolist(),
        "dtypes": filtered_df.dtypes.astype(str).to_dict()
    }

    prompt = f"""
You are a senior BI dashboard architect.

Dataset schema:
{schema}

Generate Streamlit Python code that:
- Builds a professional BI dashboard
- Shows KPI metrics
- Shows 2–4 charts
- Uses st.columns() layout
- Uses matplotlib / seaborn
- Uses filtered_df
- Ends charts with st.pyplot(plt.gcf())
- Output ONLY executable Python code
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}],
    )

    code = response.choices[0].message.content
    for fence in ("```python", "```", "`"):
        code = code.replace(fence, "")
    return code

# =================================================
# MAIN DASHBOARD
# =================================================
if df is not None:

    # ---------------- AI SLICERS ----------------
    st.subheader("🎛️ Interactive Filters (AI-Generated)")

    if "slicer_code" not in st.session_state:
        st.session_state.slicer_code = generate_slicer_code(df)

    try:
        env = {"st": st, "pd": pd, "df": df.copy()}
        exec(st.session_state.slicer_code, {}, env)
        filtered_df = env.get("filtered_df", df)
    except Exception as e:
        st.error(f"Slicer error: {e}")
        filtered_df = df

    # ---------------- KPIs ----------------
    st.subheader("📌 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", filtered_df.shape[0])
    c2.metric("Columns", filtered_df.shape[1])
    c3.metric("Numeric Columns", len(filtered_df.select_dtypes(include="number").columns))
    c4.metric("Missing Values", int(filtered_df.isnull().sum().sum()))

    # ---------------- TABLE ----------------
    with st.expander("🔍 View Filtered Dataset"):
        st.dataframe(filtered_df, use_container_width=True)

    # ---------------- AUTO INSIGHTS ----------------
    st.subheader("🧠 Automated Insights")
    with st.spinner("Generating insights..."):
        prompt = f"""
You are a senior data analyst.

Provide insights from this dataset:
{dataset_metadata(filtered_df)}
"""
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[{"role": "user", "content": prompt}],
        )
        st.markdown(resp.choices[0].message.content)

    # ---------------- AI BI DASHBOARD ----------------
    if auto_build_dashboard:
        st.subheader("🤖 AI-Built BI Dashboard")
        if "bi_code" not in st.session_state:
            st.session_state.bi_code = generate_bi_dashboard_code(filtered_df)

        try:
            exec_env = {
                "st": st,
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "filtered_df": filtered_df.copy(),
            }
            exec(st.session_state.bi_code, {}, exec_env)
        except Exception as e:
            st.error(f"Dashboard error: {e}")

    # ---------------- CHAT ----------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Conversational Analytics")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask questions or request charts...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        viz_words = ["plot", "chart", "graph", "bar", "line", "scatter", "heatmap"]
        run_code = (mode == "🧠 Code") or (mode == "⚙️ Auto" and any(w in user_prompt.lower() for w in viz_words))

        if run_code:
            code_prompt = f"""
Generate ONLY Python code using filtered_df to:
{user_prompt}
End with st.pyplot(plt.gcf())
"""
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": code_prompt}],
            )
            code = resp.choices[0].message.content
            for fence in ("```python", "```", "`"):
                code = code.replace(fence, "")
            st.code(code, language="python")
            try:
                exec(code, {}, {"st": st, "pd": pd, "plt": plt, "sns": sns, "filtered_df": filtered_df.copy()})
            except Exception as e:
                st.error(e)
            st.session_state.chat_history.append({"role": "assistant", "content": code})
        else:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[{"role": "user", "content": dataset_metadata(filtered_df) + user_prompt}],
            )
            answer = resp.choices[0].message.content
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("⬅️ Upload a dataset to start.")
