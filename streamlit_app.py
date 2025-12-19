import streamlit as st
from openai import OpenAI
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
import re

# -------------------------------
#  PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Data_Analysis_LLM", page_icon="📊", layout="wide")
st.title("LLM-Powered Chatbot for Interactive Data Analysis and Visualization")
st.caption("Chat naturally with AI, generate visualizations, or clean your dataset instantly.")

# -------------------------------
#  OPENROUTER API CLIENT
# -------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your key
)

# -------------------------------
#  LAYOUT
# -------------------------------
left_col, right_col = st.columns([1, 1])

# -------------------------------
#  FILE UPLOAD
# -------------------------------
uploaded_file = right_col.file_uploader("📎 Upload a dataset (CSV, TXT, or PDF):", type=["csv", "txt", "pdf"])
dataframe = None
file_content = ""

if uploaded_file:
    if uploaded_file.type == "text/csv":
        dataframe = pd.read_csv(uploaded_file)
        right_col.success(f" CSV '{uploaded_file.name}' uploaded successfully!")
        right_col.dataframe(dataframe.head(), use_container_width=True)
        file_content = dataframe.to_csv(index=False)
    elif uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        right_col.text_area(" File Preview", file_content[:1000])
    elif uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    file_content += text
        right_col.text_area(" Extracted PDF Text", file_content[:1000])

# -------------------------------
#  CLEAN DATA BUTTON
# -------------------------------
if dataframe is not None:
    if st.button(" Clean Data"):
        with st.spinner("Cleaning data..."):
            df = dataframe.copy()
            df = df.drop_duplicates()
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            st.success(" Data cleaned successfully!")
            st.write("###  Cleaned Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            dataframe = df

# -------------------------------
#  AUTO INSIGHTS / SUMMARY
# -------------------------------
with left_col:
    st.subheader(" Auto Insights & Summary")
    if uploaded_file:
        with st.spinner("Generating AI insights..."):
            try:
                summary = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        {"role": "user", "content": f"Summarize this dataset or document:\n\n{file_content[:6000]}"},
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Insight Dashboard",
                    },
                )
                insight_text = summary.choices[0].message.content
                st.success(" Summary Generated")
                st.write(insight_text)
            except Exception as e:
                st.error(f" Error generating insights: {e}")
    else:
        st.info("Upload a file to get AI-generated insights.")

# -------------------------------
# SMART CHAT + CODE EXECUTION
# -------------------------------
right_col.subheader(" Chat with the AI (Ask or Visualize)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with right_col.chat_message(msg["role"]):
        right_col.markdown(msg["content"])

if user_prompt := right_col.chat_input("Ask a question or request a chart (e.g. 'Plot sales vs profit')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with right_col.chat_message("user"):
        right_col.markdown(user_prompt)

    # Detect if user wants visualization
    visualization_keywords = ["plot", "chart", "graph", "visualize", "draw", "heatmap", "histogram", "boxplot"]
    wants_code = any(keyword in user_prompt.lower() for keyword in visualization_keywords)

    if dataframe is not None:
        context = f"The dataset columns are: {', '.join(dataframe.columns)}. Use variable 'df'."
    else:
        context = "No dataset uploaded, respond generally."

    if wants_code and dataframe is not None:
        #  Generate executable Python code
        full_prompt = f"""
        You are an expert Python data analyst using Streamlit, pandas, matplotlib, and seaborn.
        Dataset info: {context}
        Generate **only executable Python code** (no markdown fences, no text).
        Task: {user_prompt}
        Rules:
        - The dataset variable is 'df'.
        - Always import matplotlib.pyplot as plt and seaborn as sns if needed.
        - End code with st.pyplot(plt.gcf()).
        - No explanations or commentary, just code.
        """

        with right_col.chat_message("assistant"):
            with st.spinner(" Generating visualization code..."):
                try:
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-20b:free",
                        messages=[{"role": "user", "content": full_prompt}],
                        extra_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "AI Insight Dashboard",
                        },
                    )
                    generated_code = completion.choices[0].message.content.strip()
                except Exception as e:
                    generated_code = f"# Error generating code: {e}"

                # Clean markdown fences
                for fence in ("```python", "```py", "```", "`"):
                    generated_code = generated_code.replace(fence, "")
                generated_code = re.sub(r"^Python code.*", "", generated_code, flags=re.IGNORECASE)

                st.markdown("###  Generated Python Code:")
                st.code(generated_code, language="python")

                # Execute code safely
                if dataframe is not None:
                    try:
                        df = dataframe.copy()
                        safe_locals = {"st": st, "pd": pd, "plt": plt, "sns": sns, "df": df}
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(generated_code, {}, safe_locals)
                    except Exception as e:
                        st.error(f" Error executing generated code: {e}")
                        st.text_area(" Cleaned Code", generated_code, height=200)
    else:
        #  Normal conversational response
        with right_col.chat_message("assistant"):
            with st.spinner("💬 Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b:free",
                        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {user_prompt}"}],
                        extra_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "AI Insight Dashboard",
                        },
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                except Exception as e:
                    st.error(f" Error generating answer: {e}")

    st.session_state.messages.append({"role": "assistant", "content": user_prompt})
