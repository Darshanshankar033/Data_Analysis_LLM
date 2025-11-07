import streamlit as st
from openai import OpenAI
import pandas as pd
import pdfplumber
import io
import re

# ---------------------------------
# ⚙️ PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(page_title="AI Insight Dashboard", page_icon="📊", layout="wide")

st.title("📊 AI Insight Dashboard")
st.caption("Upload your dataset or document, chat with AI, and ask it to visualize insights dynamically!")

# ---------------------------------
# 🔑 OPENROUTER CLIENT
# ---------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Inline key
)

# ---------------------------------
# 🧭 PAGE LAYOUT — TWO COLUMNS
# ---------------------------------
left_col, right_col = st.columns([1, 1])

# ---------------------------------
# 📂 FILE UPLOAD SECTION
# ---------------------------------
uploaded_file = right_col.file_uploader("📎 Upload a file (CSV, TXT, or PDF):", type=["csv", "txt", "pdf"])

dataframe = None
file_content = ""

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "text/csv":
        dataframe = pd.read_csv(uploaded_file)
        right_col.success(f"✅ CSV '{uploaded_file.name}' uploaded successfully!")
        right_col.dataframe(dataframe.head(), use_container_width=True)
        file_content = dataframe.to_csv(index=False)

    elif file_type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        right_col.text_area("📄 File Preview", file_content[:1000])

    elif file_type == "application/pdf":
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
                        {"role": "user", "content": f"Provide a clear summary and top insights from the following data:\n\n{file_content[:6000]}"},
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Insight Dashboard",
                    },
                )
                insight_text = summary.choices[0].message.content
                st.success("✅ Insights Generated")
                st.write(insight_text)
            except Exception as e:
                st.error(f"⚠️ Error while generating insights: {e}")
    else:
        st.info("Upload a file to generate insights.")

# ---------------------------------
# 💬 CHAT INTERFACE (RIGHT COLUMN)
# ---------------------------------
right_col.subheader("💬 Chat with the AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with right_col.chat_message(msg["role"]):
        right_col.markdown(msg["content"])

# Chat input (bottom bar)
if user_input := right_col.chat_input("Ask a question or request a chart (e.g., 'Plot sales vs profit')..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with right_col.chat_message("user"):
        right_col.markdown(user_input)

    context = ""
    if dataframe is not None:
        context = f"The dataset columns are: {', '.join(dataframe.columns)}"
    elif file_content:
        context = f"Here is the uploaded file content:\n{file_content[:4000]}"

    prompt = f"{context}\n\nUser request: {user_input}. If this request sounds like a graph or visualization, respond in the format 'GRAPH: <x_column> vs <y_column> type=<chart_type>' else give text-based answer."

    with right_col.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        *st.session_state.messages[:-1],
                        {"role": "user", "content": prompt},
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Insight Dashboard",
                    },
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"⚠️ Error: {e}"

            # --- Handle visualization requests ---
            if answer.startswith("GRAPH:") and dataframe is not None:
                try:
                    # Parse format like "GRAPH: Sales vs Profit type=bar"
                    pattern = r"GRAPH:\s*(\w+)\s*vs\s*(\w+)\s*type=(\w+)"
                    match = re.search(pattern, answer, re.IGNORECASE)

                    if match:
                        x_col, y_col, chart_type = match.groups()
                        x_col, y_col, chart_type = x_col.strip(), y_col.strip(), chart_type.lower().strip()

                        if x_col in dataframe.columns and y_col in dataframe.columns:
                            st.success(f"📊 Drawing {chart_type} chart for {x_col} vs {y_col}")
                            if chart_type == "line":
                                st.line_chart(dataframe[[x_col, y_col]])
                            elif chart_type == "bar":
                                st.bar_chart(dataframe[[x_col, y_col]])
                            elif chart_type == "area":
                                st.area_chart(dataframe[[x_col, y_col]])
                            else:
                                st.info(f"Unknown chart type '{chart_type}', showing line chart.")
                                st.line_chart(dataframe[[x_col, y_col]])
                        else:
                            st.warning("⚠️ Could not match columns in dataset.")
                    else:
                        st.warning("⚠️ Graph request not clearly understood.")
                except Exception as e:
                    st.error(f"⚠️ Graph generation error: {e}")

            else:
                # Normal text response
                right_col.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
