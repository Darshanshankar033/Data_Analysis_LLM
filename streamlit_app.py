import streamlit as st
from openai import OpenAI

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "Project", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="openai/gpt-oss-20b:free",
  messages=[
              {
                "role": "user",
                "content": st.text_input("Enter your message:")
              }
            ]
)
st.write(completion.choices[0].message.content)
