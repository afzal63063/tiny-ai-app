
import streamlit as st
from transformers import pipeline

# --- Setup ---
st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("📝 AI Text Summarizer")

# Load summarization model (HuggingFace)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --- User Input ---
user_input = st.text_area(
    "Enter text to summarize:", 
    height=250, 
    key="summary_text_area_1"
)

# Optional: another input
additional_input = st.text_area(
    "Enter additional text (optional):", 
    height=250, 
    key="summary_text_area_2"
)

# --- Summarization ---
if st.button("Summarize", key="summarize_button"):
    text_to_summarize = user_input.strip()
    
    # Include additional input if present
    if additional_input.strip():
        text_to_summarize += "\n" + additional_input.strip()
    
    if text_to_summarize:
        with st.spinner("Summarizing..."):
            try:
                summary_result = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
                summary_text = summary_result[0]['summary_text']
                st.success("✅ Summary Generated:")
                st.write(summary_text)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
    else:
        st.warning("Please enter text to summarize.")

# --- Optional: Clear Inputs Button ---
if st.button("Clear Inputs", key="clear_button"):
    st.session_state["summary_text_area_1"] = ""
    st.session_state["summary_text_area_2"] = ""

