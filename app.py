import streamlit as st
from transformers import pipeline

st.title("📝 Super-Smart AI Text Summarizer")
st.write("Paste your article or text below and click **Summarize**.")

# Cache the model to avoid reloading
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Single text area with unique key
user_input = st.text_area("Paste your article here:", height=250, key="main_input")

# Summarize button
if st.button("Summarize"):
    if user_input.strip():
        summary = summarizer(user_input, max_length=150, min_length=40, do_sample=False)
        st.subheader("✨ Summary")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please paste some text first!")
