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
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="openai/gpt-oss-20b:free",
  messages=[
              {
                "role": "user",
                "content": "What NFL team won the Super Bowl in the year Justin Bieber was born?"
              }
            ]
)
st.write(print(completion.choices[0].message.content))
