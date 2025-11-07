import streamlit as st
from openai import OpenAI
import pandas as pd
import pdfplumber
import io
import contextlib
import matplotlib.pyplot as plt

# ---------------------------------
# ⚙️ PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Insight Dashboard", page_icon="📊", layout="wide")
st.title("📊 AI Insight Dashboard with AI Code Generator")
st.caption("Upload your data and ask the AI to write and run Python code for visualization or analysis.")

# ---------------------------------
# 🔑 OpenRouter API Setup
# ---------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# ---------------------------------
# 🧭 PAGE LAYOUT
# ---------------------------------
left_col, right_col = st.columns([1, 1])

# ---------------------------------
# 📂 FILE UPLOAD SECTION
# ---------------------------------
uploaded_file = right_col.file_uploader("📎 Upload a file (CSV, TXT, or PDF):", type=["csv", "txt", "pdf"])
dataframe = None
file_content = ""

if uploaded_file:
    if uploaded_file.type == "text/csv":
        dataframe = pd.read_csv(uploaded_file)
        right_col.success(f"✅ CSV '{uploaded_file.name}' uploaded successfully!")
        right_col.dataframe(dataframe.head(), use_container_width=True)
        file_content = dataframe.to_csv(index=False)
    elif uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        right_col.text_area("📄 File Preview", file_content[:1000])
    elif uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    file_content += text
        right_col.text_area("📄 Extracted PDF Text", file_content[:1000])

# ---------------------------------
# 🧠 AUTO INSIGHTS / SUMMARY
# ---------------------------------
with left_col:
    st.subheader("🧠 Auto Insights & Summary")

    if uploaded_file:
        with st.spinner("Generating AI insights..."):
            try:
                summary = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Provide a brief summary of this dataset or document:\n\n{file_content[:6000]}"
                        }
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Insight Dashboard",
                    },
                )
                insight_text = summary.choices[0].message.content
                st.success("✅ Summary Generated")
                st.write(insight_text)
            except Exception as e:
                st.error(f"⚠️ Error generating summary: {e}")
    else:
        st.info("Upload a file to generate insights.")

# ---------------------------------
# 💬 CHAT / CODE GENERATOR
# ---------------------------------
right_col.subheader("💬 Ask AI to Generate Visualization Code")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with right_col.chat_message(msg["role"]):
        right_col.markdown(msg["content"])

# Chat input
if user_prompt := right_col.chat_input("Ask something like: 'Plot sales vs profit as a bar chart'..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with right_col.chat_message("user"):
        right_col.markdown(user_prompt)

    # Context from file/dataset
    context = ""
    if dataframe is not None:
        context = f"The dataset columns are: {', '.join(dataframe.columns)}.\nThe dataframe variable name is 'df'."
    elif file_content:
        context = "A text document is uploaded."

    # 🔥 Ask LLM to write visualization code
    full_prompt = f"""
You are an expert Python data analyst using Streamlit and matplotlib.

A user uploaded a dataset. You are given this information:
{context}

Write a **Python code snippet only**, no explanations, that performs the user's request:
'{user_prompt}'

Rules:
- Assume the dataset is available as a pandas DataFrame called 'df'.
- Import only necessary libraries (matplotlib, seaborn, pandas).
- Use matplotlib or seaborn to display charts.
- Do not include file uploads or print statements.
- Your code must end with `st.pyplot(plt.gcf())` to render in Streamlit.
"""

    with right_col.chat_message("assistant"):
        with st.spinner("🧠 Generating code with AI..."):
            try:
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[{"role": "user", "content": full_prompt}],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Insight Dashboard",
                    },
                )
                generated_code = completion.choices[0].message.content
            except Exception as e:
                generated_code = f"# Error: {e}"

            # --- Show the code ---
            st.markdown("### 🧩 Generated Python Code:")
            st.code(generated_code, language="python")

            # --- Try executing the code safely ---
            if dataframe is not None:
                try:
                    df = dataframe.copy()  # Available to exec() code
                    safe_locals = {"st": st, "pd": pd, "plt": plt, "df": df}
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(generated_code, {}, safe_locals)
                except Exception as e:
                    st.error(f"⚠️ Error executing generated code: {e}")
            else:
                st.info("⚠️ Please upload a CSV dataset to generate and run visualization code.")

    st.session_state.messages.append({"role": "assistant", "content": generated_code})
