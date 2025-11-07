import streamlit as st
import pandas as pd
from openai import OpenAI

# --- Page Config ---
st.set_page_config(page_title="OpenRouter Chat", page_icon="🎈")
st.title("🎈 OpenRouter Chat App with Memory + File Upload")

# --- Initialize Chat Memory ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that can also analyze uploaded files."}
    ]

# --- Initialize OpenRouter Client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"  # Replace with your key
)

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload a file (CSV, TXT, or PDF)", type=["csv", "txt", "pdf"])

file_content = ""
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Handle CSV files
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 CSV Preview")
        st.dataframe(df.head())
        # Convert to string (limit rows to avoid large context)
        file_content = df.head(10).to_csv(index=False)

    # Handle TXT files
    elif uploaded_file.name.endswith(".txt"):
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("📄 File Content", file_content[:1000])

    # Handle PDF files
    elif uploaded_file.name.endswith(".pdf"):
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        file_content = text[:3000]  # Limit to first 3000 chars
        st.text_area("📄 Extracted PDF Text", file_content)

# --- Chat Input ---
user_input = st.text_input("Enter your message:")

# --- Show Chat History ---
st.subheader("💬 Chat History")
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['content']}")

# --- Chat Logic ---
if user_input:
    # Include file info in user message if uploaded
    full_message = user_input
    if file_content:
        full_message += f"\n\n(Attached file content for context):\n{file_content}"

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": full_message})

    try:
        with st.spinner("Thinking..."):
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=st.session_state["messages"],
                extra_headers={
                    "HTTP-Referer": "https://yourappname.streamlit.app",
                    "X-Title": "Chat with Memory and File Upload",
                },
            )

            # Get model reply
            reply = completion.choices[0].message.content

            # Add to memory and display
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.markdown(f"**Assistant:** {reply}")

    except Exception as e:
        st.error(f"Error: {e}")

# --- Optional: Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that can also analyze uploaded files."}
    ]
    st.success("Chat history cleared!")
