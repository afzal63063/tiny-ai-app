import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time  # for demo purposes

st.title("📝 Super-Smart AI Text Summarizer with Progress Bar")

# Load model safely
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "sshleifer/distilbart-cnn-12-6",
        device_map=None,
        torch_dtype="auto"
    )
    summarizer_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU; change to device_map="auto" for GPU
    )
    return summarizer_pipeline, tokenizer

summarizer, tokenizer = load_model()

# Split text into chunks for long input
def chunk_text(text, max_tokens=800):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        current_len += len(tokenizer.encode(word, add_special_tokens=False))
        if current_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(tokenizer.encode(word, add_special_tokens=False))
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Multi-pass summarization with progress bar
def smart_summarize_with_progress(text):
    chunks = chunk_text(text, max_tokens=800)
    total_chunks = len(chunks)
    chunk_summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks, start=1):
        status_text.text(f"Summarizing chunk {i} of {total_chunks}...")
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        chunk_summaries.append(result[0]['summary_text'])
        progress_bar.progress(i / total_chunks)
        time.sleep(0.1)  # slight delay for UI update (optional)

    # Final summarization of combined chunks
    status_text.text("Generating final summary...")
    combined_summary = " ".join(chunk_summaries)
    final_result = summarizer(combined_summary, max_length=150, min_length=50, do_sample=False)
    progress_bar.progress(1.0)
    status_text.text("Summarization complete ✅")
    return final_result[0]['summary_text']

# User input
user_input = st.text_area("Enter text to summarize:", height=250)

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text to summarize!")
    else:
        final_summary = smart_summarize_with_progress(user_input)
        st.subheader("Summary:")
        st.write(final_summary)
