from transformers import pipeline

# Explicitly load the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print("---- Simple AI Text Summarizer ----")

# Always read text from article.txt
try:
    with open("article.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("⚠️ Please create a file named 'article.txt' in the same folder as summarizer.py")
    exit()

# Run summarization
summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

print("\n--- Summary ---")
print(summary[0]['summary_text'])
