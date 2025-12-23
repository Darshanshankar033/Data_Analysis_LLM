import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="LLM BI Platform", layout="wide")

MODEL = "openai/gpt-oss-20b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["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"]
)

# Email config (SMTP)
SMTP_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = st.secrets.get("SMTP_EMAIL", "")
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", "")

# ======================================================
# LLM HELPER
# ======================================================
def llm(prompt):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# ======================================================
# AGENTS
# ======================================================
def profile_agent(df):
    return {
        "rows": df.shape[0],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict()
    }

def summary_agent(profile):
    prompt = f"""
Summarize the dataset and suggest 3 insightful questions.
Dataset profile:
{profile}
"""
    return llm(prompt)

def eda_agent(profile):
    prompt = f"""
Suggest EDA directions without aggregation or feature engineering.
Dataset profile:
{profile}
"""
    return llm(prompt)

def dashboard_agent(profile, eda_plan):
    prompt = f"""
You are a BI dashboard developer.

Dataset profile:
{profile}

EDA plan:
{eda_plan}

RULES:
- Use Plotly Express
- DataFrame name: df
- No aggregation
- No new columns
- No file access
- Create KPI cards + 2–3 interactive charts
- Output ONLY Python code
"""
    return llm(prompt)

def chat_agent(profile, memory, user_input):
    prompt = f"""
You are a conversational data assistant.

Dataset profile:
{profile}

Chat history:
{memory}

User request:
{user_input}

If user asks for a plot:
- Generate Plotly code using df
If user asks to regenerate dashboard:
- Say: REGENERATE_DASHBOARD
"""
    return llm(prompt)

def email_agent(profile, summary):
    return f"""
Hello,

Here is your AI-generated data summary:

Dataset Overview:
{profile}

Insights:
{summary}

Regards,
AI BI Agent
"""

# ======================================================
# EMAIL SENDER
# ======================================================
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_EMAIL, SMTP_PASSWORD)
    server.send_message(msg)
    server.quit()

# ======================================================
# SESSION STATE
# ======================================================
for key in ["chat", "dashboard_code", "profile", "summary", "eda"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat" else []

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])
email_to = st.sidebar.text_input("📧 Send report to")
send_mail = st.sidebar.button("Send Email")

# ======================================================
# LOAD DATA
# ======================================================
df = None
if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

# ======================================================
# UI TABS
# ======================================================
st.title("🤖 LLM-Powered BI Platform")
summary_tab, dashboard_tab, chat_tab = st.tabs(
    ["📌 Summary", "📊 Dashboard", "💬 Chat"]
)

# ======================================================
# SUMMARY
# ======================================================
if df is not None:
    with summary_tab:
        if st.session_state.profile is None:
            st.session_state.profile = profile_agent(df)
            st.session_state.summary = summary_agent(st.session_state.profile)
            st.session_state.eda = eda_agent(st.session_state.profile)

        st.markdown(st.session_state.summary)

# ======================================================
# DASHBOARD
# ======================================================
if df is not None:
    with dashboard_tab:
        if st.button("Generate / Refresh Dashboard") or st.session_state.dashboard_code is None:
            st.session_state.dashboard_code = dashboard_agent(
                st.session_state.profile,
                st.session_state.eda
            )

        try:
            exec(
                st.session_state.dashboard_code,
                {},
                {"st": st, "df": df, "px": px}
            )
        except Exception as e:
            st.error("Dashboard error")
            st.exception(e)

# ======================================================
# CHAT
# ======================================================
if df is not None:
    with chat_tab:
        for m in st.session_state.chat:
            st.chat_message(m["role"]).markdown(m["content"])

        user_input = st.chat_input("Ask about data, plots, or regenerate dashboard")

        if user_input:
            st.session_state.chat.append({"role": "user", "content": user_input})

            reply = chat_agent(
                st.session_state.profile,
                st.session_state.chat,
                user_input
            )

            if "REGENERATE_DASHBOARD" in reply:
                st.session_state.dashboard_code = dashboard_agent(
                    st.session_state.profile,
                    st.session_state.eda
                )
                reply = "Dashboard regenerated."

            st.session_state.chat.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").markdown(reply)

# ======================================================
# EMAIL
# ======================================================
if send_mail and email_to:
    email_body = email_agent(st.session_state.profile, st.session_state.summary)
    send_email(email_to, "AI BI Dashboard Summary", email_body)
    st.sidebar.success("📧 Email sent successfully")
