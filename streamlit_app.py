import streamlit as st
from openai import OpenAI
import pandas as pd
import pdfplumber
import sqlite3
import io

# ---------------------------------
# ⚙️ PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Insight Dashboard", page_icon="📊", layout="wide")

st.title("📊 AI Insight Dashboard")
st.caption("Upload your dataset, connect a database, and chat with AI to explore insights and visualize data.")

# ---------------------------------
# 🔑 OPENROUTER API KEY (INLINE)
# ---------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# ---------------------------------
# 🗃️ DATABASE CONNECTION (Optional)
# ---------------------------------
st.sidebar.header("🗄️ Database Connection (Optional)")
use_db = st.sidebar.checkbox("Enable Database Connection")

conn = None
if use_db:
    db_type = st.sidebar.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"])
    if db_type == "SQLite":
        db_file = st.sidebar.text_input("SQLite DB File Path", "data.db")
        conn = sqlite3.connect(db_file)
        st.sidebar.success(f"✅ Connected to SQLite: {db_file}")

    elif db_type == "MySQL":
        import mysql.connector
        host = st.sidebar.text_input("Host", "localhost")
        user = st.sidebar.text_input("User", "root")
        password = st.sidebar.text_input("Password", "")
        database = st.sidebar.text_input("Database", "")
        if st.sidebar.button("Connect to MySQL"):
            conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
            st.sidebar.success(f"✅ Connected to MySQL: {database}")

    elif db_type == "PostgreSQL":
        import psycopg2
        host = st.sidebar.text_input("Host", "localhost")
        user = st.sidebar.text_input("User", "postgres")
        password = st.sidebar.text_input("Password", "")
        database = st.sidebar.text_input("Database", "")
        if st.sidebar.button("Connect to PostgreSQL"):
            conn = psycopg2.connect(host=host, user=user, password=password, dbname=database)
            st.sidebar.success(f"✅ Connected to PostgreSQL: {database}")

# ---------------------------------
# 🧭 PAGE LAYOUT (Two Columns)
# ---------------------------------
left_col, right_col = st.columns([1, 1])

# ------------------------
# 📂 FILE UPLOAD SECTION
# ------------------------
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

        # Store to DB if connection is active
        if conn is not None:
            table_name = st.text_input("Enter table name to store dataset:", "uploaded_data")
            if st.button("📥 Save to Database"):
                dataframe.to_sql(table_name, conn, if_exists="replace", index=False)
                st.success(f"✅ Data saved to table '{table_name}'")

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

# ------------------------
# 🧠 AUTO SUMMARY (Left Column)
# ------------------------
with left_col:
    st.subheader("🧠 Summary / Insights")

    if uploaded_file:
        with st.spinner("Generating AI insights..."):
            try:
                summary = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        {"role": "user", "content": f"Generate a summary and key insights from the following data:\n\n{file_content[:6000]}"},
                    ],
                )
                insight_text = summary.choices[0].message.content
                st.success("✅ Summary Generated")
                st.write(insight_text)
            except Exception as e:
                st.error(f"⚠️ Error while generating insights: {e}")
    else:
        st.info("Upload a file to generate insights.")

# ------------------------
# 💬 CHAT INTERFACE
# ------------------------
right_col.subheader("💬 Chat with the AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with right_col.chat_message(msg["role"]):
        right_col.markdown(msg["content"])

# Chat input
if user_input := right_col.chat_input("Ask a question about your data or database..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with right_col.chat_message("user"):
        right_col.markdown(user_input)

    # Context: from file or database
    context = ""
    if dataframe is not None:
        context = f"Here is the dataset sample:\n{dataframe.head(10).to_csv(index=False)}"
    elif file_content:
        context = f"Here is the uploaded file content:\n{file_content[:4000]}"

    # If DB connected, let AI know
    if conn is not None:
        context += "\nThe user also has an active database connection for structured queries."

    full_prompt = f"{context}\n\nUser question: {user_input}"

    with right_col.chat_message("assistant"):
        with right_col.spinner("🤔 Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[
                        *st.session_state.messages[:-1],
                        {"role": "user", "content": full_prompt},
                    ],
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Error: {e}"

            right_col.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------------
# 📊 VISUALIZATION
# ------------------------
if dataframe is not None:
    with right_col.expander("📊 Quick Visualization"):
        numeric_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            x_axis = st.selectbox("Select X-axis:", numeric_cols, key="x_axis")
            y_axis = st.selectbox("Select Y-axis:", numeric_cols, key="y_axis")
            st.line_chart(dataframe[[x_axis, y_axis]])
        else:
            st.info("No numeric columns available for plotting.")
