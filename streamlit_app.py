
import streamlit as st
from openai import OpenAI

st.title("Chat with OpenRouter 🚀")

# User text input box
user_input = st.text_input("Enter your message:")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your real key or use Secrets
)

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[
                {"role": "user", "content": user_input}
            ],
            extra_headers={
                "HTTP-Referer": "https://example.com",  
                "X-Title": "My Streamlit App"
            }
        )

        st.write("### Response:")
        st.write(response.choices[0].message["content"])
